"""Single-task SCOPE demo for LIBERO-Pro."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shlex
import socket
import subprocess
import sys
from contextlib import nullcontext, suppress
from dataclasses import asdict
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import ProxyHandler, build_opener

from .critic import _request, plan_graph
from .diagnostic_graph import GraphContext, digest, json_text, parse_json, save_artifact
from .libero import (
    BOUNDARY_PROFILE,
    SIM_ROOT,
    LiberoSettings,
    compile_libero_source,
    load_public_items,
    write_json,
)
from .libero_runner import LiberoExecutor, worker_environment, worker_facts
from .model_transport import ModelClient, ModelSettings, generate_initial
from .progress import LearningProgress, action_label, print_progress, render_preflight
from .protocol import LIBERO_PRO_SUITES, ScopeProtocol
from .records import CampaignIdentity
from .replay import ReplayEnvironment
from .search import RecoveryRequiredError, SearchCampaign


def parser():
    result = argparse.ArgumentParser(description="SCOPE single-task LIBERO-Pro demo")
    result.add_argument("--env-file", type=Path)
    result.add_argument("--gpu", dest="task_gpu", default="0", help="Physical GPU index")
    result.add_argument("--seed", type=int, choices=range(51, 66), default=51)
    result.add_argument("--output", type=Path, default=Path("outputs/demo"))
    result.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    result.add_argument("--model", default=os.environ.get("SCOPE_AGENT_MODEL", ""))
    result.add_argument("--worker-python", type=Path, default=Path(sys.executable))
    result.add_argument("--max-tokens", type=int, default=16384)
    result.add_argument("--reasoning-effort", default="low")
    result.add_argument("--response-format", choices=("provider", "json_object"), default="provider")
    result.add_argument("--max-actions", type=int, default=8,
                        help="Number of search actions in this invocation (default: 8)")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--check", action="store_true", help="Check dependencies and configuration")
    result.set_defaults(
        suite="libero_goal_swap", task="open_the_middle_drawer_of_the_cabinet",
        config=SIM_ROOT / "env_configs/libero/demo.yaml", temperature=0,
        api_timeout_seconds=3600, max_api_retries=2, max_empty_responses=3,
        max_initial_program_retries=2, execution_timeout_seconds=600,
        sampling_seed=0, max_frames=16, frame_stride=3, log_format="human",
        progress_interval_seconds=30,
    )
    return result


def load_settings(args):
    import yaml

    path = args.config.absolute()
    data = yaml.safe_load(path.read_text())
    cfg = data["env"]["cfg"]
    low = cfg["low_level"]
    if cfg.get("privileged") is not False or low.get("privileged") is not False:
        raise ValueError("SCOPE requires the nonprivileged public LIBERO API")
    if low["_target_"] != "scope.sim.cap.envs.simulators.libero.FrankaLiberoEnv":
        raise ValueError("config is not the existing LIBERO environment")
    if cfg["apis"] not in (
        ["FrankaLiberoApiReducedSkillLibraryTraced"],
        ["FrankaLiberoApiReducedSkillLibrary"],
    ):
        raise ValueError("config does not select the reduced public skill API")
    return LiberoSettings(
        max_steps=low.get("max_steps", 4000),
        control_freq=low.get("control_freq", 20),
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
    )


def model_settings(args):
    return ModelSettings(
        args.base_url,
        args.model,
        args.max_tokens,
        args.temperature,
        args.api_timeout_seconds,
        args.max_api_retries,
        args.max_empty_responses,
        args.reasoning_effort,
        args.response_format,
    )


def default_output(args):
    """Match SCOPE's flat directory naming, with a SCOPE method suffix."""
    name = (
        "-".join(
            quote(value, safe="._-")
            for value in (args.suite, args.task or "all-tasks", args.model, args.reasoning_effort)
        )
        + "-scope"
    )
    # Match SCOPE's portable component limit and deterministic resume naming.
    if len(name) > 240:
        name = f"{name[:217]}-{digest(name)[:16]}-scope"
    return SIM_ROOT / "outputs" / "scope" / name


def preflight(args):
    if args.output is None:
        args.output = default_output(args)
    protocol = ScopeProtocol(args.suite, args.task, development_seeds=tuple(args.development_seeds))
    settings = load_settings(args)
    if not math.isfinite(args.progress_interval_seconds) or args.progress_interval_seconds <= 0:
        raise ValueError("progress interval must be positive")
    if not 0 <= args.max_initial_program_retries <= 10:
        raise ValueError("initial program retry limit must be within 0..10")
    if args.max_actions is not None and args.max_actions < 1:
        raise ValueError("max-actions must be positive")
    if not math.isfinite(args.execution_timeout_seconds) or args.execution_timeout_seconds <= 0:
        raise ValueError("execution timeout must be positive")
    ports = {
        name: int(os.environ.get(name) or str(default))
        for name, default in (("SAM3_PORT", 8114), ("GRASPNET_PORT", 8115), ("PYROKI_PORT", 8116))
    }
    blockers = []
    model = None
    try:
        model = model_settings(args)
    except ValueError:
        blockers.append("model_or_endpoint_missing_or_invalid")
    facts = None
    try:
        facts = worker_facts(args.worker_python.absolute(), gpu=args.task_gpu, service_ports=ports)
        if not facts.get("runtime_sha256"):
            raise ValueError("worker description unavailable")
    except (OSError, ValueError, RuntimeError):
        blockers.append("worker_environment_unavailable")
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.used",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        gpu_rows = [
            row for row in gpu.stdout.splitlines() if row.split(",")[0].strip() == args.task_gpu
        ]
        if gpu.returncode or len(gpu_rows) != 1:
            blockers.append("selected_gpu_not_visible")
    except (OSError, subprocess.TimeoutExpired):
        gpu_rows = []
        blockers.append("gpu_probe_unavailable")
    services = {}
    opener = build_opener(ProxyHandler({}))
    for name, port in ports.items():
        try:
            with opener.open(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                services[name] = response.status == 200
        except HTTPError as error:
            # Repository perception servers do not define /health; their
            # documented liveness check accepts 404 (not weight readiness).
            services[name] = error.code == 404
        except Exception:
            services[name] = False
        if not services[name]:
            blockers.append(name.lower() + "_unavailable")
    output = args.output.absolute()
    if output.exists() and not args.resume:
        blockers.append("output_already_exists")
    if args.resume and not (output / "run.json").is_file():
        blockers.append("resume_manifest_missing")
    if not Path.home().joinpath(".libero/config.yaml").exists() and not os.environ.get(
        "LIBERO_CONFIG_PATH"
    ):
        blockers.append("libero_path_config_missing")
    n = len(protocol.development_seeds)
    report = {
        "format": "scope-libero-preflight",
        "host": socket.gethostname(),
        "suite": args.suite,
        "task": args.task,
        "development_seeds": list(protocol.development_seeds),
        "held_out_seeds": list(protocol.held_out_seeds),
        "gpu_physical": args.task_gpu,
        "gpu_mapping": gpu_rows,
        "worker_cuda_logical": "0",
        "worker_egl_physical": args.task_gpu,
        "services": services,
        "service_ports": ports,
        "credentials": {
            "OPENAI_API_KEY_present": bool(os.environ.get("OPENAI_API_KEY")),
            "HF_TOKEN_present": bool(os.environ.get("HF_TOKEN")),
            "gated_weights": "not downloaded or checked; service liveness does not prove weight readiness",
        },
        "model": model.public_config() if model else None,
        "worker": {k: v for k, v in (facts or {}).items() if k != "api_specification"},
        "worker_python": str(args.worker_python.absolute()),
        "settings": asdict(settings),
        "paper_budgets": {
            "main_repairs_per_seed": 3,
            "followup_repairs_per_seed": 1,
            "replay_sample_size": 3,
            "diagnostics_per_accepted_update": 3,
        },
        "upper_bounds": {
            "search_executions": 6 * n,
            "diagnostics": 6 * n,
            "environment_executions_including_bootstrap": 12 * n + 1,
            "environment_timeout_hours": (12 * n + 1) * args.execution_timeout_seconds / 3600,
        },
        "runtime_estimate": "Uncalibrated: depends on simulator/service/model latency; timeout bound excludes model time.",
        "output": str(output),
        "sampling_seed": args.sampling_seed,
        "environment_timeout_seconds": args.execution_timeout_seconds,
        "max_initial_program_retries": args.max_initial_program_retries,
        "execution_video": {
            "enabled": True,
            "camera": "agentview",
            "fps": 30,
            "path": "campaign/executions/<execution_id>/video_<success_score>[_<termination>].mp4",
            "capture": "environment simulation-step frames (same as SCOPE replay_trial)",
            "hard_timeout": "SIGKILL may prevent video export",
        },
        "invocation_action_limit": args.max_actions,
        "invocation_environment_upper_bound": min(12 * n, args.max_actions)
        + (0 if args.resume else 1)
        if args.max_actions is not None
        else 12 * n + (0 if args.resume else 1),
        "old_fix_loop_options_not_used": ["max-agent-turns", "skill-extraction-mode"],
        "blockers": blockers,
        "ready": not blockers,
    }
    return report, protocol, settings, facts, ports


def check_self_matches(campaign, executor, progress=None):
    """Resume missing checks before advancing, retaining unavailable diagnostics."""
    state = campaign.snapshot()
    counts = {"certified": 0, "unavailable": 0, "diverged": 0}
    for record in state.executions.values():
        if record.result is None or not record.result.success or not record.role.is_search:
            continue
        path = campaign.directory / f"self_match_{record.execution_id}.json"
        new_match = not path.exists()
        if new_match:
            phase = (
                progress.phase(f"seed={record.seed:03d} | Successful trajectory self-match")
                if progress
                else nullcontext()
            )
            with phase:
                executor.self_match(record, state.programs[record.source_sha256]).save(path)
        match = parse_json(path.read_text())
        if (
            match["candidate_sha256"] != record.source_sha256
            or match["witness_execution_id"] != record.execution_id
            or match["witness_source_sha256"] != record.source_sha256
            or match["tape_sha256"] != (record.result.tape.sha256 if record.result.tape else None)
            or match["api_sha256"] != executor.facts["api_sha256"]
            or match["runtime_sha256"] != executor.facts["runtime_sha256"]
        ):
            raise ValueError("self-match report identity mismatch")
        counts[match["status"]] += 1
        if progress and new_match:
            progress.emit(
                "self_match_result",
                f"Self-match result | seed={record.seed:03d} | {match['status']} | {match['reason']}",
            )
        if match["status"] == "diverged":
            return {
                "status": "self_match_diverged",
                "execution_id": record.execution_id,
                "search": str(campaign.path),
                "frozen": False,
            }
    return {"self_matches": counts}


def learn(args, report, protocol, settings, facts, ports):
    output = args.output.absolute()
    model_config = model_settings(args)
    configuration = {
        "max_initial_program_retries": args.max_initial_program_retries,
        "model": model_config.public_config(),
        "settings": asdict(settings),
        "service_ports": ports,
        "sampling_seed": args.sampling_seed,
        "execution_timeout_seconds": args.execution_timeout_seconds,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "worker_python": str(args.worker_python.absolute()),
        "gpu": args.task_gpu,
        "adapter_sources": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in (
                "libero_learning.py",
                "libero_runner.py",
                "_libero_worker.py",
                "model_transport.py",
                "progress.py",
                "critic.py",
                "repair.py",
                "retrieval.py",
                "search.py",
            )
        },
    }
    identity = CampaignIdentity(
        "scope-libero",
        protocol,
        facts["api_sha256"],
        facts["runtime_sha256"],
        digest(configuration),
    )
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
        save_artifact(output / "preflight.json", report)
        save_artifact(
            output / "run.json", {"identity": asdict(identity), "configuration": configuration}
        )
    else:
        stored = parse_json((output / "run.json").read_text())
        if (
            stored["sha256"] != digest(stored["data"])
            or stored["data"]["configuration"] != configuration
        ):
            raise ValueError("resume configuration changed")
    progress = LearningProgress(
        output, log_format=args.log_format, interval=args.progress_interval_seconds,
        context=getattr(args, "progress_context", None),
    )
    progress.emit(
        "run_start",
        f"{'Resuming' if args.resume else 'Starting'} learning | {args.suite}/{args.task}"
        f" | seeds={list(protocol.development_seeds)} | GPU={args.task_gpu} | model={args.model}",
    )
    progress.emit("output", f"Output | {output} | summary: summary.md | timeline: learning.log")
    try:
        return _learn_campaign(args, protocol, settings, facts, ports, identity, progress)
    except BaseException as error:
        status = "recovery_required" if isinstance(error, RecoveryRequiredError) else "interrupted"
        progress.summary["status"] = status
        try:
            if (output / "campaign/search.json").exists():
                campaign = SearchCampaign.open(output / "campaign", expected_identity=identity)
                progress.sync(campaign, status=status)
        except (OSError, ValueError):
            pass
        with suppress(OSError):
            write_json(
                output / "result.json",
                {
                    "status": status,
                    "error_code": type(error).__name__,
                    "frozen": progress.summary.get("frozen", False),
                    "search": str(output / "campaign/search.json")
                    if (output / "campaign/search.json").exists()
                    else None,
                },
            )
        progress.emit(
            "run_error",
            f"Learning interrupted | {type(error).__name__} | inspect {output / 'progress.json'}"
            f" and {output / 'campaign/search.json'}",
        )
        raise


def _learn_campaign(args, protocol, settings, facts, ports, identity, progress):
    output = args.output.absolute()
    model = progress.model(ModelClient(model_settings(args), output / "model_calls"))
    campaign_root = output / "campaign"
    executor = LiberoExecutor(
        python=args.worker_python,
        root=campaign_root,
        suite=args.suite,
        task=args.task,
        gpu=args.task_gpu,
        settings=settings,
        facts=facts,
        service_ports=ports,
        timeout_seconds=args.execution_timeout_seconds,
    )
    if args.resume:
        campaign = SearchCampaign.open(campaign_root, expected_identity=identity)
    else:
        # Bootstrap has its own evidence root and no search/budget slot. It uses
        # the explicitly selected first development seed and public language.
        bootstrap_executor = LiberoExecutor(
            python=args.worker_python,
            root=output,
            suite=args.suite,
            task=args.task,
            gpu=args.task_gpu,
            settings=settings,
            facts=facts,
            service_ports=ports,
            timeout_seconds=args.execution_timeout_seconds,
        )
        with progress.phase(f"Initial observation | seed={protocol.development_seeds[0]:03d}"):
            probe = bootstrap_executor.probe(protocol.development_seeds[0], output / "bootstrap")
        public = parse_json((output / "bootstrap/public.json").read_text())
        context = GraphContext(
            protocol,
            "scope_bootstrap",
            protocol.development_seeds[0],
            facts["api_sha256"],
            probe["task_language"],
            facts["api_specification"],
            load_public_items(public),
        )
        graph_plan = plan_graph(context, archive_root=output, model=model)
        graph_plan.save(output / "graph_planning")
        if graph_plan.graph is None:
            raise RuntimeError(
                f"graph planning failed ({graph_plan.issue}); "
                f"inspect {output / 'graph_planning/planning.json'} "
                f"and {output / 'model_calls'}"
            )
        program, request, raw = generate_initial(
            _request(context, output),
            model,
            max_retries=args.max_initial_program_retries,
            attempt_directory=output / "initial_program_attempts",
        )
        save_artifact(
            output / "initial_program.json",
            {"program": asdict(program), "request": request.to_dict(), "raw_response": raw},
        )
        campaign = SearchCampaign.create(
            campaign_root,
            identity,
            program,
            graph=graph_plan.graph,
            task_language=probe["task_language"],
            api_specification=facts["api_specification"],
            random_seed=args.sampling_seed,
            replay_environment=ReplayEnvironment(
                facts["api_sha256"], facts["runtime_sha256"], BOUNDARY_PROFILE
            ),
        )

    def finish(result):
        summary = progress.sync(campaign, status=result["status"], result=result)
        labels = {
            "search_complete": "Learning finished",
            "action_limit": "Invocation action limit reached; resume to continue",
            "self_match_diverged": "Self-match diverged; stopped",
        }
        progress.emit(
            "run_end",
            f"{labels.get(result['status'], result['status'])} | completed executions={summary['completed_executions']}"
            f" | historically successful search seeds={summary['historically_successful_seeds']}"
            f" | frozen={result['frozen']} | summary: {output / 'summary.md'}",
        )
        if result["frozen"] and not summary["historically_successful_seeds"]:
            progress.emit(
                "no_success",
                "No successful search executions were recorded; the frozen source is still the initial program.",
            )
        write_result = {
            **result,
            "completed_executions": summary["completed_executions"],
            "historically_successful_seeds": summary["historically_successful_seeds"],
        }
        write_json(output / "result.json", write_result)
        return write_result

    executor.task_language = campaign.state()["task_language"]
    progress.sync(campaign)
    completed_actions = 0
    while True:
        matches = check_self_matches(campaign, executor, progress)
        if "status" in matches:
            return finish(matches)
        action = campaign.next_action()
        progress.sync(campaign)
        if action is None:
            break
        if action.status != "ready":
            progress.emit(
                "recovery_required",
                f"Pending action has no confirmed receipt; explicit recovery is required | {action.action_id} | {action.status}",
            )
            raise RecoveryRequiredError("pending action requires explicit recovery")
        if args.max_actions is not None and completed_actions >= args.max_actions:
            return finish({"status": "action_limit", "search": str(campaign.path), "frozen": False})
        label = action_label(action, campaign.state())
        with progress.phase(
            label,
            execution_id=action.payload.get("execution_id") if action.kind == "execute" else None,
        ):
            action = campaign.run_next(
                executor=executor,
                critic_model=model,
                planner=model,
                generator=model,
                replayer=executor.replayer,
                candidate_validator=compile_libero_source,
            )
        completed_actions += 1
        progress.action_finished(action, campaign)
    frozen = output / "frozen.json"
    if not frozen.exists():
        campaign.export_frozen(frozen)
    else:
        stored = parse_json(frozen.read_text())
        if (
            stored["sha256"] != digest(stored["data"])
            or stored["data"]["program"] != asdict(campaign.frozen_program)
            or stored["data"]["campaign_id"] != campaign.snapshot().identity.campaign_id
        ):
            raise ValueError("frozen export identity mismatch")
    return finish(
        {
            "status": "search_complete",
            "search": str(campaign.path),
            "frozen": True,
            **matches,
            "source_sha256": campaign.frozen_program.sha256,
        }
    )


def load_environment(path):
    """Read assignments and unset directives; never execute the environment file."""
    from dotenv import dotenv_values
    from dotenv.parser import parse_stream

    path = Path(path)
    if not path.is_file():
        raise ValueError("environment file does not exist")
    lines, unset = [], []
    for line in path.read_text().splitlines(keepends=True):
        if line.lstrip().startswith("unset "):
            names = shlex.split(line, comments=True)[1:]
            if not names or any(not name.isidentifier() or not name.isascii() for name in names):
                raise ValueError("unsupported environment unset directive")
            unset.extend(names)
            lines.append("\n")
        else:
            lines.append(line)
    text = "".join(lines)
    if any(binding.error for binding in parse_stream(io.StringIO(text))):
        raise ValueError("environment file contains unsupported shell syntax")
    allowed = (
        "OPENAI_BASE_URL",
        "SCOPE_AGENT_MODEL",
        "OPENAI_API_KEY",
        "HF_TOKEN",
        "SAM3_PORT",
        "GRASPNET_PORT",
        "PYROKI_PORT",
        "LIBERO_CONFIG_PATH",
        "HF_HUB_OFFLINE",
        "ROBOT_DESCRIPTIONS_CACHE",
    )
    for name in unset:
        if name in allowed or name.lower() in (
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "no_proxy",
        ):
            os.environ.pop(name, None)
    values = dotenv_values(stream=io.StringIO(text))
    for name in allowed:
        if values.get(name):
            os.environ[name] = values[name]


def prepare_task(args):
    prepared = preflight(args)
    print_progress(
        json.dumps(prepared[0], ensure_ascii=False, indent=2)
        if args.log_format == "json"
        else render_preflight(prepared[0]),
    )
    return prepared


def run_task(args, *, prepared=None):
    report, protocol, settings, facts, ports = prepare_task(args) if prepared is None else prepared
    if not args.yes:
        if args.log_format == "human" and report["ready"]:
            print_progress("Configuration ready. Run without --check to start the demo.")
        return 0 if report["ready"] else 2
    if not report["ready"]:
        raise SystemExit("preflight has blockers; no environment trial started")
    try:
        result = learn(args, report, protocol, settings, facts, ports)
    except RecoveryRequiredError:
        raise SystemExit(
            f"pending action requires explicit recovery; inspect {args.output / 'campaign/search.json'}"
        ) from None
    if args.log_format == "json":
        print_progress(json.dumps({**getattr(args, "progress_context", {}), **result}, indent=2))
    return 0 if result["status"] in ("search_complete", "action_limit") else 3



def main(argv=None):
    env_parser = argparse.ArgumentParser(add_help=False)
    env_parser.add_argument("--env-file", type=Path)
    env_args, _ = env_parser.parse_known_args(argv)
    if env_args.env_file is not None:
        load_environment(env_args.env_file)
    args = parser().parse_args(argv)
    if not args.task_gpu.isascii() or not args.task_gpu.isdigit():
        raise ValueError("--gpu must be a nonnegative physical GPU index")
    args.task_gpu = str(int(args.task_gpu))
    args.development_seeds = [args.seed]
    args.yes = not args.check
    return run_task(args)
