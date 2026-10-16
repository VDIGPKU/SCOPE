"""M8 public LIBERO boundary. Simulator/perception imports are worker-only.

Generated code receives data, a NumPy facade and recorded public API functions;
it never receives the environment, evaluator, API receiver, or task registry.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from scope.sim.cap.envs.simulators.libero_execution import (
    EpisodeStepLimitError,
    MotionConvergenceError,
)
from scope.sim.cap.utils.ik_validation import IkPoseError

from .diagnostic_graph import PublicEvidence, digest, json_text, require_text
from .motion_evidence import motion_feedback, motion_target
from .protocol import ScopeProtocol
from .records import EvidenceRef, ExecutionResult, ProgramVersion, Termination
from .search import ExecutionReceipt

BOUNDARY_PROFILE = "scope-libero-public-v1"
SIM_ROOT = Path(__file__).resolve().parents[2]
API_FILES = (
    "cap/integrations/franka/libero_reduced.py",
    "cap/integrations/franka/libero_reduced_skill_library.py",
)
PUBLIC_FUNCTIONS = (
    "get_observation",
    "segment_sam3_text_prompt",
    "segment_sam3_point_prompt",
    "plan_grasp",
    "plan_grasp_from_point_clouds",
    "get_oriented_bounding_box_from_3d_points",
    "solve_ik",
    "move_to_joints",
    "open_gripper",
    "close_gripper",
    "goto_pose",
    "goto_home_joint_position",
    "subsample_point_cloud",
    "filter_noise",
    "rotation_matrix_to_quaternion",
    "decompose_transform",
    "depth_to_point_cloud",
    "mask_to_world_points",
    "pixel_to_world_point",
    "transform_points",
    "interpolate_segment",
    "normalize_vector",
    "select_top_down_grasp",
)
# Keyframes follow physical actions rather than spending the image budget on
# hundreds of geometry calls. Priorities favor the grasp transition, then arm
# motion, while a bounded temporal sample retains early and late evidence.
MOTION_FRAME_PRIORITY = {
    "close_gripper": 5,
    "goto_pose": 3,
    "move_to_joints": 3,
    "open_gripper": 2,
    "goto_home_joint_position": 1,
}
SOURCE_CONTRACT = """Return a complete program with exactly one def main(): entry, no arguments.
Only function definitions and docstrings at module scope. No imports, files,
network, clocks, random generators, private attributes, env or evaluator access.
Call public APIs as api["name"](...).
Public API functions are NOT injected as bare names: use api["get_observation"](),
never get_observation(). No identifier may start with an underscore: use
cluster_1d, not _cluster_1d; use named variables instead of the '_' placeholder.
np is preloaded with supported deterministic NumPy functions; do not import numpy.
inputs["task_language"] is the authoritative
public goal. inputs["initial_observation"] is the public observation at this reset.
Use current camera/depth evidence, never task filenames or memorized coordinates.
Use API geometry utilities for rotations/IK. API outputs are ordinary Python and
NumPy data; an API call or program return does not itself prove task success.
Use no print, lambda, decorators, default arguments, annotations or external tools.
Each joint move returns after convergence or the existing at-most-120 control
steps, even if its 0.01 rad joint-error norm tolerance is not met. Joint residuals,
step counts and convergence flags are diagnostic evidence, not fatal errors or
task-success judgments. Subsequent code continues; no host retry is added.
Inspect measured robot-state evidence rather than assuming a normal return proves
arrival. The episode control-step limit and other execution errors still apply.
SCOPE IK starts with measured arm joints, then checks robot-model FK within
0.005 m hand-position and 0.05 rad orientation tolerances, up to five solves per
orientation. ik_pose_not_converged reports MODEL pose residuals, not physical
arrival/collision. Fallbacks can still change the requested orientation.
A motion_error observation reports the requested final TCP endpoint residual,
not the per-waypoint motion error; an exception can occur at the approach pose.
Even converged joints do not prove requested Cartesian orientation/position,
because solve_ik may clamp the target or use an orientation fallback.
"""


def api_specification():
    """Read Python method signatures/docstrings without importing service clients."""
    methods = {}
    for relative in API_FILES:
        tree = ast.parse((SIM_ROOT / relative).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in PUBLIC_FUNCTIONS:
                signature = ast.unparse(node.args)
                signature = signature.removeprefix("self, ").removeprefix("self")
                methods[node.name] = (
                    f'api["{node.name}"]({signature})\n{ast.get_docstring(node) or ""}'
                )
    if set(methods) != set(PUBLIC_FUNCTIONS):
        raise ValueError("public API declaration differs from repository sources")
    from ._replay_policy import NUMPY_NAMES

    return (
        SOURCE_CONTRACT
        + "\nNumPy names: "
        + ", ".join(NUMPY_NAMES)
        + "\n\n"
        + "\n\n".join(methods[name] for name in PUBLIC_FUNCTIONS)
    )


def compile_libero_source(source):
    """Apply the shared replay profile and LIBERO's explicit entry/API contract."""
    from ._replay_policy import ProfileError, compile_source

    code = compile_source(source, "main")
    tree = ast.parse(source)
    entry = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    if (
        entry.args.posonlyargs
        or entry.args.args
        or entry.args.kwonlyargs
        or entry.args.vararg
        or entry.args.kwarg
    ):
        raise ProfileError("main_requires_arguments", entry.lineno)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in PUBLIC_FUNCTIONS
        ):
            raise ProfileError("public_api_requires_mapping", node.lineno)
    return code


def api_fingerprint():
    files = (
        *API_FILES,
        "cap/envs/simulators/libero.py",
        "cap/envs/simulators/libero_execution.py",
        "cap/integrations/franka/common.py",
        "cap/utils/ik_validation.py",
        "cap/integrations/motion/pyroki.py",
        "cap/serving/launch_pyroki_server.py",
        "cap/integrations/libero/__init__.py",
        "cap/scope/libero.py",
        "cap/scope/motion_evidence.py",
        "cap/scope/_replay_policy.py",
        "cap/scope/recording.py",
        "cap/scope/state_codec.py",
    )
    return digest(
        {
            "profile": BOUNDARY_PROFILE,
            "functions": PUBLIC_FUNCTIONS,
            "sources": {
                name: hashlib.sha256((SIM_ROOT / name).read_bytes()).hexdigest() for name in files
            },
        }
    )


@dataclass(frozen=True)
class LiberoSettings:
    max_steps: int = 4000
    control_freq: int = 20
    max_frames: int = 16
    frame_stride: int = 3
    max_api_calls: int = 512
    tape_max_bytes: int = 1024**3

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()) or self.max_frames < 4:
            raise ValueError("invalid LIBERO limits; max_frames must be at least four")


class LiberoBackend:
    """Trusted adapter for the existing reduced public API and host evaluator."""

    def __init__(self, suite, task, seed, settings):
        ScopeProtocol(suite, task).require_development_seed(seed)
        from libero import benchmark

        from scope.sim.cap.envs.simulators.libero import FrankaLiberoEnv
        from scope.sim.cap.integrations.franka.libero_reduced_skill_library import (
            FrankaLiberoApiReducedSkillLibrary,
        )
        from scope.sim.cap.integrations.motion.pyroki import check_pyroki_pose_support

        check_pyroki_pose_support()
        registry = benchmark.get_benchmark_dict()
        task_suite = registry[suite]()
        matches = [i for i in range(task_suite.n_tasks) if task_suite.get_task(i).name == task]
        if len(matches) != 1:
            raise ValueError("task is not uniquely registered in selected suite")
        self.env = FrankaLiberoEnv(
            suite,
            matches[0],
            privileged=False,
            max_steps=settings.max_steps,
            seed=seed,
            control_freq=settings.control_freq,
            viser_debug=False,
            strict_motion=False,
        )
        self.env.reset(seed=seed)
        self.env.motion_reports = []
        self.task_language = self.env.handle.task_language
        require_text(self.task_language, "public task language")
        self.api = FrankaLiberoApiReducedSkillLibrary(self.env)
        self.api.validate_ik_pose = True
        available = self.api.functions()
        self.functions = {name: available[name] for name in PUBLIC_FUNCTIONS}

    def observe(self):
        return self.functions["get_observation"]()

    def take_motion_reports(self):
        reports = self.env.motion_reports
        self.env.motion_reports = []
        return reports

    def completed(self):
        # Only the trusted host consumes evaluator output. Never injected into
        # the generated namespace, frames, API trace or tape public state.
        return bool(self.env.task_completed())

    def start_video(self):
        # Same simulation-step capture used by LIBERO replay_trial.py. This is
        # host-only and independent of the sparse diagnostic frame budget.
        self.env.enable_video_capture(True, clear=True)

    def video_frames(self):
        return self.env.get_video_frames(clear=True)

    def close(self):
        # The worker is the lifetime boundary; BaseEnv exposes close().
        self.env.close()


def write_json(path, value):
    """Atomic progress replacement inside one fresh execution directory."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json_text(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_execution_video(backend, directory, task_completed, termination):
    """Export the environment's rollout frames, never reconstruct from tape."""
    import imageio.v2 as imageio
    import numpy as np

    frames = backend.video_frames()
    result = ExecutionResult(task_completed, termination)
    score = "unknown" if task_completed is None else f"{float(result.success):.3f}"
    suffix = "" if result.termination is Termination.NORMAL else f"_{result.termination.value}"
    filename = f"video_{score}{suffix}.mp4"
    metadata = {
        "camera": "agentview",
        "fps": 30,
        "frame_count": len(frames),
        "task_completed": task_completed,
        "termination": result.termination.value,
        "outcome": result.outcome.value,
    }
    if not frames:
        return {**metadata, "status": "unavailable", "reason": "no_frames"}
    directory = Path(directory)
    path, temporary = directory / filename, directory / "video.partial.mp4"
    if any(directory.glob("video*.mp4")):
        raise FileExistsError("execution video already exists")
    with imageio.get_writer(str(temporary), fps=30, format="FFMPEG", codec="libx264") as writer:
        for frame in frames:
            writer.append_data(np.ascontiguousarray(frame))
    temporary.replace(path)
    return {
        **metadata,
        "status": "saved",
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def evidence_ref(path, root, kind):
    path = Path(path)
    return EvidenceRef(
        kind,
        path.resolve().relative_to(Path(root).resolve()).as_posix(),
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def write_worker_error(directory, error, stage):
    """Host-only traceback locations; omit exception text, locals and source."""
    frames = []
    current = error.__traceback__
    while current is not None:
        frames.append(
            {
                "file": current.tb_frame.f_code.co_filename,
                "line": current.tb_lineno,
                "function": current.tb_frame.f_code.co_name,
            }
        )
        current = current.tb_next
    write_json(
        Path(directory) / "worker_error.json",
        {"error_class": type(error).__name__, "stage": stage, "frames": frames},
    )
    if stage == "program_execution":
        # Only generated-code locations are public. Host paths, exception
        # messages/arguments, locals and simulator frames remain excluded.
        details = {}
        if type(error) is EpisodeStepLimitError:
            details = {
                "reason": "simulation_step_limit",
                "max_steps": error.max_steps,
                "consumed_steps": error.consumed_steps,
            }
        elif type(error) is IkPoseError:
            details = error.details
        elif type(error) is MotionConvergenceError:
            details = {
                "reason": "joint_motion_not_converged",
                "joint_error_norm_rad": public_summary(error.joint_error_norm_rad),
                "tolerance_rad": public_summary(error.tolerance_rad),
                "motion_steps": error.motion_steps,
            }
        write_json(
            Path(directory) / "public_error.json",
            {
                "error_class": type(error).__name__,
                **details,
                "source_frames": [
                    {"line": frame["line"], "function": frame["function"]}
                    for frame in frames
                    if frame["file"] == "<scope-candidate>"
                ],
            },
        )


def public_summary(value, depth=0):
    """Bounded, data-only API summaries for diagnosis; exact values stay in M3."""
    import numpy as np

    kind = type(value)
    if kind in (str,):
        return value[:256]
    if value is None or kind in (bool, int):
        return value if kind is not int or value.bit_length() <= 128 else "large_integer"
    if kind is float:
        return value if math.isfinite(value) else "nonfinite_float"
    if kind is np.ndarray:
        result = {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
        if value.size <= 16 and value.dtype.kind in "biuf":
            result["values"] = public_summary(value.tolist(), depth)
        return result
    if isinstance(value, np.generic) and value.dtype.kind in "biuf":
        item = value.item()
        return (
            public_summary(item, depth)
            if type(item) in (bool, int, float)
            else {"type": kind.__name__}
        )
    if depth < 3 and kind in (tuple, list):
        return [public_summary(item, depth + 1) for item in value[:8]]
    if depth < 3 and kind is dict:
        return {
            key[:128]: public_summary(item, depth + 1)
            for key, item in list(value.items())[:8]
            if type(key) is str
        }
    return {"type": kind.__name__, "detail": "omitted"}


def summary_text(value):
    text = json_text(public_summary(value))
    return text if len(text) <= 4096 else json_text({"detail": "summary_budget_exceeded"})


class PublicJournal:
    def __init__(self, directory, root, execution_id, settings):
        self.directory, self.root = Path(directory), Path(root)
        self.execution_id, self.settings = execution_id, settings
        self.items, self.frames, self.calls, self.issues = [], [], 0, []
        self.frame_priorities = {}
        self.language = None
        self.status = "starting"
        self.publish()

    def publish(self):
        write_json(
            self.directory / "public.json",
            {
                "execution_id": self.execution_id,
                "task_language": self.language,
                "status": self.status,
                "max_step": self.calls,
                "items": [asdict(i) for i in self.items],
                "issues": self.issues,
            },
        )

    def _select_frame_step(self, step, priority):
        """Keep a bounded action-prioritized sample across the whole attempt."""
        capacity = (self.settings.max_frames - 4) // 2
        if not capacity:
            return False
        if step in self.frame_priorities:
            self.frame_priorities[step] = max(priority, self.frame_priorities[step])
            return True
        candidates = {**self.frame_priorities, step: priority}
        if len(candidates) > capacity:
            steps = sorted(candidates)

            def retention_key(candidate):
                index = steps.index(candidate)
                # Preserve temporal endpoints among equally important frames;
                # remove a densely sampled interior point first.
                span = (
                    steps[index + 1] - steps[index - 1]
                    if 0 < index < len(steps) - 1 else math.inf
                )
                return candidates[candidate], span, candidate

            evicted = min(steps, key=retention_key)
            if evicted == step:
                return False
            del self.frame_priorities[evicted]
            removed = [item for item in self.frames if item.step == evicted]
            self.frames = [item for item in self.frames if item.step != evicted]
            removed_ids = {item.evidence_id for item in removed}
            self.items = [item for item in self.items if item.evidence_id not in removed_ids]
            for item in removed:
                (self.directory / (item.evidence_id + ".png")).unlink()
        self.frame_priorities[step] = priority
        return True

    def capture(self, observation, step, *, final=False, priority=0):
        import numpy as np
        from PIL import Image

        # The same public observation already fetched for camera capture also
        # contains measured robot state. Keep it even if the image budget is
        # exhausted; a successful motion call does not imply pose convergence.
        state_id = f"robot_state_{step:06d}"
        if type(observation) is dict and not any(i.evidence_id == state_id for i in self.items):
            robot_state = {
                name: observation[name]
                for name in ("robot_cartesian_pos", "robot_joint_pos")
                if name in observation
            }
            if robot_state:
                self.items.append(
                    PublicEvidence(
                        state_id, self.execution_id, step, "observation", summary_text(robot_state)
                    )
                )
        # Pose records are independent of PNG selection, and must be durable
        # even when no camera pair is admitted at this step.
        if step != 0 and not final and not self._select_frame_step(step, priority):
            self.publish()
            return
        for camera in ("agentview", "robot0_eye_in_hand"):
            frame_id = f"frame_{step:06d}_{camera}"
            if any(item.evidence_id == frame_id for item in self.frames):
                continue
            limit = self.settings.max_frames if final else self.settings.max_frames - 2
            if len(self.frames) >= limit:
                break
            try:
                rgb = observation[camera]["images"]["rgb"]
                if (
                    not isinstance(rgb, np.ndarray)
                    or rgb.dtype != np.uint8
                    or rgb.ndim != 3
                    or rgb.shape[2] != 3
                ):
                    raise ValueError("expected public uint8 RGB")
                path = self.directory / (frame_id + ".png")
                with path.open("xb") as stream:
                    Image.fromarray(rgb).save(stream, format="PNG")
                ref = evidence_ref(path, self.root, "public_frame")
                item = PublicEvidence(
                    frame_id, self.execution_id, step, "frame", artifact=ref, mime_type="image/png"
                )
                self.frames.append(item)
                self.items.append(item)
            except (KeyError, ValueError, TypeError):
                self.issues.append(f"frame_unavailable_{camera}_{step}")
        self.publish()

    def wrap(self, name, function, observe, take_motion_reports=None):
        def call(*args, **kwargs):
            if self.calls >= self.settings.max_api_calls:
                raise RuntimeError("public_api_call_limit")
            priority = MOTION_FRAME_PRIORITY.get(name)
            if priority is not None and take_motion_reports is not None:
                take_motion_reports()  # Discard any report preceding this API call.
            if priority is not None:
                try:
                    self.capture(observe(), self.calls, priority=priority)
                except Exception:
                    self.issues.append(f"pre_action_capture_failed_{self.calls}")
            try:
                target = motion_target(name, args, kwargs)
            except (KeyError, TypeError, ValueError):
                target = None
            self.calls += 1
            step = self.calls
            index = len(self.items)
            arguments = summary_text({"args": args, "kwargs": kwargs})
            self.items.append(
                PublicEvidence(
                    f"api_{step:06d}",
                    self.execution_id,
                    step,
                    "api",
                    json_text({"name": name, "outcome": "pending", "arguments": arguments}),
                )
            )
            self.publish()
            outcome = "return"
            response = None
            try:
                result = function(*args, **kwargs)
                response = summary_text(result)
                return result
            except Exception as error:
                outcome = "exception:" + type(error).__name__
                raise
            finally:
                if priority is not None and take_motion_reports is not None:
                    for segment, report in enumerate(take_motion_reports(), start=1):
                        self.items.append(PublicEvidence(
                            f"motion_joint_{step:06d}_{segment:02d}", self.execution_id, step,
                            "observation", summary_text({**report, "api_outcome": outcome}),
                        ))
                self.items[index] = PublicEvidence(
                    f"api_{step:06d}",
                    self.execution_id,
                    step,
                    "api",
                    json_text(
                        {
                            "name": name,
                            "outcome": outcome,
                            "arguments": arguments,
                            "response": response,
                        }
                    ),
                )
                if priority is not None or step % self.settings.frame_stride == 0:
                    try:
                        observation = observe()
                        if target is not None:
                            try:
                                feedback = motion_feedback(target, observation)
                                feedback["api_outcome"] = outcome
                                self.items.append(PublicEvidence(
                                    f"motion_error_{step:06d}", self.execution_id, step,
                                    "observation", summary_text(feedback),
                                ))
                            except (KeyError, TypeError, ValueError):
                                self.issues.append(f"motion_feedback_unavailable_{step}")
                        self.capture(observation, step, priority=priority or 0)
                    except Exception:
                        self.issues.append(f"capture_failed_{step}")
                self.publish()

        return call


def execute_public_program(request, *, backend_factory=LiberoBackend):
    """Worker entry; fake backends allow complete offline contract testing."""
    from ._replay_policy import ProfileError, execution_globals
    from .recording import BoundarySpec, TapeIdentity, TapeLimits, TapeRecorder
    from .replay import replay_runtime_fingerprint

    root, directory = Path(request["root"]), Path(request["directory"])
    settings = LiberoSettings(**request["settings"])
    protocol = ScopeProtocol(request["suite"], request["task"])
    protocol.require_development_seed(request["seed"])
    journal = PublicJournal(directory, root, request["execution_id"], settings)
    backend, recorder = None, None
    video = {
        "status": "unavailable" if request["mode"] == "execute" else "not_requested",
        "reason": "backend_not_started" if request["mode"] == "execute" else "probe",
    }
    termination, task_completed, error_code = Termination.INFRASTRUCTURE_ERROR, None, None
    stage, error_stage = "fingerprint_validation", None
    try:
        if (
            request["api_sha256"] != api_fingerprint()
            or request["runtime_sha256"] != replay_runtime_fingerprint()
        ):
            raise ValueError("worker_fingerprint_mismatch")
        stage = "source_validation"
        program = None if request["mode"] == "probe" else ProgramVersion(**request["program"])
        code = compile_libero_source(program.source) if program else None
        stage = "backend_initialization"
        backend = backend_factory(request["suite"], request["task"], request["seed"], settings)
        if program is not None:
            try:
                backend.start_video()
                video = {"status": "recording"}
            except Exception as error:
                video = {
                    "status": "error",
                    "stage": "capture_start",
                    "error_code": type(error).__name__,
                }
        stage = "public_observation"
        journal.language = backend.task_language
        require_text(journal.language, "task language")
        if (
            request.get("expected_language") is not None
            and journal.language != request["expected_language"]
        ):
            raise ValueError("public_task_language_changed")
        observation = backend.observe()
        journal.capture(observation, 0)
        if program is None:
            journal.status = "probe_complete"
            journal.publish()
            return {"mode": "probe", "task_language": journal.language, "public": "public.json"}
        stage = "program_setup"
        inputs = {"task_language": journal.language, "initial_observation": observation}
        public = {}
        functions = {
            name: journal.wrap(name, fn, backend.observe, getattr(backend, "take_motion_reports", None))
            for name, fn in backend.functions.items()
            if name in PUBLIC_FUNCTIONS
        }
        if set(functions) != set(PUBLIC_FUNCTIONS):
            raise ValueError("runtime_public_api_mismatch")
        recorder = TapeRecorder(
            directory / "tape",
            identity=TapeIdentity(
                request["execution_id"],
                program.sha256,
                request["api_sha256"],
                request["runtime_sha256"],
            ),
            boundary=BoundarySpec(BOUNDARY_PROFILE, True),
            functions=functions,
            inputs=inputs,
            public_state=public,
            limits=TapeLimits(max_storage_bytes=settings.tape_max_bytes),
        )

        def unsupported(reason, **details):
            recorder.mark_unavailable(reason)
            raise RuntimeError(reason)

        namespace = execution_globals(
            recorder.functions, recorder.readers, inputs, public, unsupported
        )
        exec(code, namespace)
        if inspect.signature(namespace["main"]).parameters:
            raise ProfileError("main_requires_arguments")
        journal.status = "running"
        journal.publish()
        stage = "program_execution"
        try:
            recorder.run(namespace["main"])
        except Exception as error:
            termination = Termination.EXCEPTION
            error_code = (
                "simulation_step_limit"
                if type(error) is EpisodeStepLimitError
                else "joint_motion_not_converged"
                if type(error) is MotionConvergenceError
                else "ik_pose_not_converged"
                if type(error) is IkPoseError
                else type(error).__name__
            )
            error_stage = stage
            write_worker_error(directory, error, stage)
        else:
            termination = Termination.NORMAL
        # Evaluation remains host-only, after program termination.
        stage = "task_evaluation"
        task_completed = backend.completed()
        try:
            journal.capture(backend.observe(), journal.calls, final=True)
        except Exception:
            journal.issues.append("final_capture_failed")
    except ProfileError as error:
        termination, error_code = Termination.EXCEPTION, error.reason
        error_stage = stage
        write_worker_error(directory, error, stage)
    except Exception as error:
        termination, task_completed, error_code = (
            Termination.INFRASTRUCTURE_ERROR,
            None,
            type(error).__name__,
        )
        error_stage = stage
        write_worker_error(directory, error, stage)
    finally:
        if backend is not None:
            if video["status"] == "recording":
                try:
                    video = save_execution_video(backend, directory, task_completed, termination)
                except Exception as error:
                    video = {
                        "status": "error",
                        "stage": "export",
                        "error_code": type(error).__name__,
                    }
            try:
                backend.close()
            except Exception:
                journal.issues.append("backend_close_failed")
        if request["mode"] == "execute":
            # These host-only artifacts are not added to PublicEvidence or M3,
            # so export errors cannot affect task success or the model prompt.
            video.update(
                execution_id=request["execution_id"],
                seed=request["seed"],
                suite=request["suite"],
                task=request["task"],
                source_sha256=request["program"]["sha256"],
            )
            try:
                write_json(directory / "video.json", video)
            except Exception:
                video["metadata_write_failed"] = True
    journal.status = "finished"
    journal.publish()
    return {
        "mode": request["mode"],
        "termination": termination.value,
        "task_completed": task_completed,
        "error_code": error_code,
        "error_stage": error_stage,
        "task_language": journal.language,
        "public": "public.json",
        "tape_replay_ready": recorder.manifest["replay_ready"] if recorder else False,
        "video": video,
    }


def load_public_items(data):
    items = []
    for value in data["items"]:
        row = dict(value)
        if row["artifact"] is not None:
            row["artifact"] = EvidenceRef(**row["artifact"])
        items.append(PublicEvidence(**row))
    return tuple(items)


def receipt_from_files(directory, root, execution_id, result):
    from .diagnostic_graph import parse_json, read_public_artifact
    from .recording import RecordedTape
    from .state_codec import StateUnavailable

    directory, root = Path(directory), Path(root)
    path = directory / "public.json"
    data = (
        parse_json(path.read_text())
        if path.exists()
        else {"execution_id": execution_id, "max_step": 0, "items": []}
    )
    if data["execution_id"] != execution_id:
        raise ValueError("worker public evidence identity mismatch")
    items = load_public_items(data)
    refs = [i.artifact for i in items if i.artifact]
    for ref in refs:
        read_public_artifact(root, ref)
    raw = []
    if path.exists():
        raw.append(evidence_ref(path, root, "public_trace"))
    error_path = directory / "public_error.json"
    if error_path.exists():
        ref = evidence_ref(error_path, root, "public_error")
        error_text = read_public_artifact(root, ref).decode("utf-8")
        items += (
            PublicEvidence("program_error", execution_id, data["max_step"], "error", error_text),
        )
        raw.append(ref)
    if (directory / "tape" / "tape.json").exists():
        try:
            tape = RecordedTape.load(directory / "tape")
            raw.append(tape.evidence_ref(root))
        except StateUnavailable:
            pass  # Independently recorded public trace is still diagnostic evidence.
    refs.extend(raw)
    return ExecutionReceipt(
        ExecutionResult(
            result["task_completed"], result["termination"], tuple(refs), result.get("error_code")
        ),
        data["max_step"],
        items,
        tuple(raw),
    )
