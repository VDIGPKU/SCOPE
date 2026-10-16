"""Human-facing SCOPE progress and rebuildable views of the authoritative ledger.

No scheduling decisions, evidence mutation, simulator access or model prompts.
Attempt aliases point to the original execution directories; tapes aren't copied.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path

from .libero import write_json

# The coordinator also runs on Python 3.10, before datetime.UTC was added.
UTC = timezone.utc  # noqa: UP017
_CONSOLE_LOCK = threading.Lock()


def print_progress(message):
    """Keep concurrent tasks' console records intact."""
    with _CONSOLE_LOCK:
        print(message, flush=True)

ROLE_NAMES = {
    "main_entry": ("initial", "Main entry"),
    "main_repair": ("fix", "Main repair"),
    "followup_entry": ("followup_initial", "Follow-up entry"),
    "followup_repair": ("followup_fix", "Follow-up repair"),
    "diagnostic": ("diagnostic", "Historical diagnostic execution"),
}
PURPOSE_NAMES = {
    "graph": "Diagnostic graph planning",
    "initial_program": "Initial program generation/correction",
    "critic": "Execution diagnosis",
    "repair_plan": "Repair planning",
    "repair_generate": "Repair code generation",
}
OUTCOME_NAMES = {
    "success": "Success",
    "task_failure": "Task failed",
    "exception": "Program exception",
    "timeout": "Timeout",
    "infrastructure_error": "Infrastructure error",
    "reserved": "Awaiting execution receipt",
    "interrupted": "Recovery required",
}
END_REASONS = {
    "search_success_same_source": "Execution succeeded",
    "accepted_update_complete": "Source update accepted and historical validation processed",
    "repair_budget_exhausted": "Repair budget exhausted",
    "infrastructure_error_no_task_outcome": "Infrastructure error; task outcome unknown",
    "repair_model_error": "Repair proposal failed validation",
}


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_text(path, text):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("derived report must not replace a symlink")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def attempt_rows(state):
    counts, rows = Counter(), []
    for record in sorted(
        state["ledger"]["executions"].values(), key=lambda r: r["started_revision"]
    ):
        seed, role = record["seed"], record["role"]
        counts[seed, role] += 1
        index = counts[seed, role]
        name = f"seed_{seed:03d}_{ROLE_NAMES[role][0]}_{index:02d}"
        result = record["result"]
        outcome = record["status"]
        if result is not None:
            outcome = result["termination"]
            if outcome == "normal":
                outcome = "success" if result["task_completed"] else "task_failure"
        rows.append(
            {
                "name": name,
                "path": f"attempts/{name}",
                "execution_id": record["execution_id"],
                "seed": seed,
                "role": role,
                "attempt": index,
                "outcome": outcome,
                "status": record["status"],
                "source_sha256": record["source_sha256"],
                "error_code": result.get("error_code") if result else None,
            }
        )
    return rows


def action_label(action, state):
    rows = attempt_rows(state)
    payload = action.payload
    target = (
        payload.get("execution_id")
        or payload.get("failure_id")
        or payload.get("witness_execution_id")
    )
    row = next((r for r in rows if r["execution_id"] == target), None)
    seed = payload.get("seed", row["seed"] if row else None)
    protocol = state["ledger"]["identity"]["protocol"]
    seed_text = f"seed={seed:03d}" if seed is not None else ""
    position = (
        f"Development seed {protocol['development_seeds'].index(seed) + 1}/{len(protocol['development_seeds'])}"
        if seed in protocol["development_seeds"]
        else ""
    )
    if action.kind == "execute":
        role = row["role"]
        label = ROLE_NAMES[role][1]
        if role.endswith("repair"):
            limit = (
                protocol["max_main_repairs"]
                if role == "main_repair"
                else protocol["max_followup_repairs"]
            )
            label += f" {row['attempt']}/{limit}"
        return f"{position} | {seed_text} | {label} | {row['path']}"
    if action.kind == "repair":
        main = state["visit"]["queue"] == "main"
        role = "main_repair" if main else "followup_repair"
        count = sum(r["seed"] == seed and r["role"] == role for r in rows) + 1
        limit = protocol["max_main_repairs"] if main else protocol["max_followup_repairs"]
        return f"{position} | {seed_text} | {ROLE_NAMES[role][1]} {count}/{limit}"
    label = (
        "Execution diagnosis"
        if action.kind == "critic"
        else "Historical trajectory replay certification"
    )
    return f"{position} | {seed_text} | {label}"


def render_preflight(report):
    model = report.get("model") or {}
    lines = [
        "SCOPE / LIBERO-Pro learning preflight",
        f"Task: {report['suite']}/{report['task']}",
        f"Model: {model.get('model', 'not configured')} | reasoning={model.get('reasoning_effort', '-')}",
        f"Response format: {model.get('response_format', 'provider')}",
        f"Development seeds: {', '.join(map(str, report['development_seeds']))} | GPU: {report['gpu_physical']}",
        f"Host: {report['host']} | worker Python: {report.get('worker', {}).get('python', '-')}",
        "GPU mapping: " + "; ".join(report.get("gpu_mapping", [])),
        "Held-out seeds are excluded; model credentials stay in the coordinator.",
        "Budgets: 3 main repairs and 1 follow-up repair per seed; up to 3 historical samples and 3 diagnostics per update",
        f"Environment timeout: {report.get('environment_timeout_seconds', '-')}s | Model timeout: {model.get('timeout_seconds', '-')}s",
        "Services: "
        + ", ".join(
            f"{k}={'reachable' if v else 'unreachable'}" for k, v in report["services"].items()
        ),
        "Credentials: "
        + ", ".join(
            f"{k}={'set' if v else 'not set'}"
            for k, v in report["credentials"].items()
            if k.endswith("_present")
        ),
        "Reuse existing services without downloading weights; reachable services may not be ready for inference.",
        f"Conservative environment execution bound for this invocation: {report['invocation_environment_upper_bound']}; time bound excludes model calls/retries.",
        f"Output: {report['output']}",
        "Browse: summary.md / progress.json / learning.log / attempts/",
    ]
    if report.get("invocation_action_limit") is not None:
        lines.append(
            f"Invocation limited to {report['invocation_action_limit']} scheduler actions; unfinished searches will not be frozen"
        )
    lines.append(
        "Preflight passed." if report["ready"] else "Blockers: " + ", ".join(report["blockers"])
    )
    return "\n".join(lines)


class LearningProgress:
    def __init__(self, output, *, log_format="human", interval=30, context=None):
        self.output = Path(output)
        self.log_format, self.interval = log_format, interval
        self.context = dict(context or {})
        self.lock = threading.RLock()
        self.active, self.summary = None, {"status": "initializing"}
        self.io_error_reported = False
        if not (self.output / "summary.md").exists():
            write_text(
                self.output / "summary.md",
                "# SCOPE learning progress\n\nInitializing. See the current stage in "
                "[progress.json](progress.json) and the timeline in [learning.log](learning.log).\n",
            )

    def _persist(self):
        active = dict(self.active) if self.active else None
        if active:
            active["elapsed_seconds"] = round(time.monotonic() - active.pop("started"), 1)
        write_json(
            self.output / "progress.json",
            {
                **self.summary,
                "updated_at": datetime.now(UTC).isoformat(),
                "active": active,
            },
        )

    def emit(self, event, message, **details):
        # Observability failures must never turn a completed paid call into a retry.
        with self.lock:
            stamp = datetime.now(UTC).isoformat(timespec="seconds")
            row = {"time": stamp, "event": event, "message": message, **self.context, **details}
            line = f"[{stamp[11:19]} UTC] {message}"
            prefix = (
                f"[{self.context['suite']}/{self.context['task']} | GPU={self.context['gpu']}] "
                if self.context else ""
            )
            with suppress(OSError):
                print_progress(
                    json.dumps(row, ensure_ascii=False) if self.log_format == "json" else prefix + line,
                )
            try:
                with (self.output / "learning.log").open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
                with (self.output / "events.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                self._persist()
            except OSError:
                self.io_error_reported = True

    @contextmanager
    def phase(self, label, *, execution_id=None):
        previous = self.active
        active = {"label": label, "started": time.monotonic(), "execution_id": execution_id}
        with self.lock:
            self.active = active
        self.emit("stage_start", f"Started | {label}")
        stopped = threading.Event()

        def heartbeat():
            while not stopped.wait(self.interval):
                with self.lock:
                    if self.active is not active:
                        continue
                    extra = ""
                    if execution_id:
                        public = read_json(
                            self.output / "campaign/executions" / execution_id / "public.json", {}
                        )
                        if "max_step" in public:
                            extra = f" | Public API calls={public['max_step']}"
                    elapsed = round(time.monotonic() - active["started"])
                    self.emit(
                        "heartbeat",
                        f"Running | {label} | elapsed={elapsed}s{extra}",
                        elapsed_seconds=elapsed,
                    )

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            yield
        except BaseException as error:
            self.emit(
                "stage_error",
                f"Interrupted | {label} | {type(error).__name__}",
                error_class=type(error).__name__,
            )
            raise
        else:
            elapsed = round(time.monotonic() - active["started"], 1)
            self.emit(
                "stage_end", f"Finished | {label} | elapsed={elapsed}s", elapsed_seconds=elapsed
            )
        finally:
            stopped.set()
            thread.join()
            with self.lock:
                self.active = previous
                try:
                    self._persist()
                except OSError:
                    self.io_error_reported = True

    def model(self, client):
        if hasattr(client, "observer"):
            client.observer = self.model_event

        def call(request):
            parent = self.active["label"] + " / " if self.active else ""
            label = PURPOSE_NAMES.get(request.purpose, request.purpose)
            if request.purpose == "initial_program":
                number = request.payload.get("initial_program_feedback", {}).get("attempt", 0) + 1
                label += f" (version {number})"
            with self.phase(parent + "Model: " + label):
                return client(request)

        return call

    def model_event(self, event, **details):
        purpose = PURPOSE_NAMES.get(details["purpose"], details["purpose"])
        if event == "model_retry":
            message = f"Model retry | {purpose} | {details['reason']} | {details['used']}/{details['limit']}"
        else:
            message = f"Model response | {purpose} | tokens={details['usage'].get('total_tokens', 'not provided')}"
        self.emit(event, message, **details)

    def sync(self, campaign, *, status="learning", result=None):
        """Rebuild views from committed state; never invoke next_action()."""
        state = campaign.state()
        ledger = state["ledger"]
        protocol = ledger["identity"]["protocol"]
        rows = attempt_rows(state)
        attempts = self.output / "attempts"
        attempts.mkdir(exist_ok=True)
        for row in rows:
            target = self.output / "campaign/executions" / row["execution_id"]
            alias = self.output / row["path"]
            relative = os.path.relpath(target, alias.parent)
            if alias.is_symlink():
                if os.readlink(alias) != relative:
                    raise ValueError("attempt alias conflicts with execution identity")
            elif alias.exists():
                raise ValueError("attempt alias would overwrite existing output")
            else:
                alias.symlink_to(relative, target_is_directory=True)
            worker = read_json(target / "result.json", {})
            video = read_json(target / "video.json", worker.get("video", {}))
            # New recordings encode the outcome in their name. Older artifacts
            # used video.mp4; retain that fallback without changing old files.
            filename = video.get("path", "video.mp4")
            video_path = None
            if (
                isinstance(filename, str)
                and Path(filename).name == filename
                and filename.endswith(".mp4")
                and not filename.endswith(".partial.mp4")
                and (target / filename).is_file()
            ):
                video_path = f"{row['path']}/{filename}"
            tape = read_json(target / "tape/tape.json", {})
            row.update(
                tape_replay_ready=worker.get("tape_replay_ready", tape.get("replay_ready")),
                video_status=video.get("status", "not_saved"),
                video_path=video_path,
            )
            if target.is_dir():
                write_text(
                    target / "program.py", ledger["programs"][row["source_sha256"]]["source"]
                )
                write_json(target / "attempt_summary.json", row)
                assessment = state["assessments"].get(row["execution_id"])
                if assessment:
                    write_json(
                        target / "diagnosis.json",
                        {
                            k: assessment[k]
                            for k in (
                                "status",
                                "usable",
                                "frontier",
                                "issues",
                                "nodes",
                            )
                        },
                    )
        for name, key in (
            ("initial_code.py", "initial_source_sha256"),
            ("working_code.py", "working_source_sha256"),
            ("accepted_code.py", "accepted_source_sha256"),
        ):
            write_text(self.output / name, ledger["programs"][ledger[key]]["source"])
        if state["frozen"]:
            write_text(self.output / "frozen_code.py", state["frozen"]["source"])
        completed_main = {
            e["seed"] for e in state["events"] if e["kind"] == "visit_end" and e["queue"] == "main"
        }
        successes = {
            r["seed"] for r in rows if r["outcome"] == "success" and r["role"] != "diagnostic"
        }
        summary = {
            "status": status,
            "suite": protocol["suite"],
            "task": protocol["task"],
            "checkpoint_revision": state["revision"],
            "main_seeds_completed": len(completed_main),
            "main_seeds_total": len(protocol["development_seeds"]),
            "completed_main_seeds": sorted(completed_main),
            "historically_successful_seeds": sorted(successes),
            "completed_executions": sum(r["status"] == "completed" for r in rows),
            "followups_pending": len(state["followups"]),
            "pending_action": state["pending"],
            "attempts": rows,
            "frozen": state["frozen"] is not None,
            "frozen_is_initial": bool(
                state["frozen"] and state["frozen"]["sha256"] == ledger["initial_source_sha256"]
            ),
            "result": result,
        }
        responses = [
            read_json(p, {}) for p in (self.output / "model_calls").glob("*.response.json")
        ]
        summary["model_calls_completed"] = len(responses)
        summary["model_calls_with_token_usage"] = sum(
            "total_tokens" in r.get("data", {}).get("usage", {}) for r in responses
        )
        summary["model_usage"] = {
            name: sum(r.get("data", {}).get("usage", {}).get(name, 0) for r in responses)
            for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        previous_completed = set(self.summary.get("completed_main_seeds", []))
        with self.lock:
            self.summary = summary
            self._persist()
        lines = [
            "# SCOPE learning progress",
            "",
            f"Task: `{protocol['suite']}/{protocol['task']}`",
            "",
            f"Status: **{status}**; main queue completed {len(completed_main)}/{len(protocol['development_seeds'])} seeds; "
            f"historically successful search seeds: {sorted(successes)}; pending follow-ups: {len(state['followups'])}.",
            "",
            "Historical success does not establish that the current program succeeds on all seeds; diagnostic executions are excluded from search successes.",
            "",
            "See the current stage and heartbeat in [progress.json](progress.json) and the timeline in [learning.log](learning.log).",
            "The authoritative checkpoint is [campaign/search.json](campaign/search.json); this page and attempts are rebuildable views.",
            "",
            "Source: [initial](initial_code.py) / [working](working_code.py) / [accepted](accepted_code.py).",
            "The accepted program may still be the initial program without a successful execution.",
            "",
        ]
        lines += [
            f"Model responses received: {summary['model_calls_completed']}; "
            f"total tokens: {summary['model_usage']['total_tokens']}"
            f" ({summary['model_calls_with_token_usage']} requests reported usage).",
            "",
        ]
        if state["frozen"]:
            lines += [
                "Search finished; the [frozen program](frozen_code.py) is exported. See task outcomes below.",
                "",
            ]
            if not successes:
                lines += [
                    "**No successful search executions were recorded; the frozen source is still the initial program.**",
                    "",
                ]
        lines += [
            "| Attempt | Seed | Role | Outcome | Replay-ready tape | Video |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in rows:
            video_link = (
                f"[Play]({row['video_path']})" if row["video_path"] else row["video_status"]
            )
            tape = (
                "unknown"
                if row["tape_replay_ready"] is None
                else ("yes" if row["tape_replay_ready"] else "no")
            )
            lines.append(
                f"| [{row['name']}]({row['path']}/) | {row['seed']} | {ROLE_NAMES[row['role']][1]} | "
                f"{OUTCOME_NAMES.get(row['outcome'], row['outcome'])} | {tape} | {video_link} |"
            )
        write_text(self.output / "summary.md", "\n".join(lines) + "\n")
        write_json(self.output / "attempts.json", rows)
        for seed in sorted(completed_main - previous_completed):
            event = next(
                e
                for e in state["events"]
                if e["kind"] == "visit_end" and e["queue"] == "main" and e["seed"] == seed
            )
            self.emit(
                "seed_finished",
                f"Main queue | seed={seed:03d} finished | {END_REASONS.get(event['reason'], event['reason'])}"
                f" | completed={len(completed_main)}/{len(protocol['development_seeds'])}",
            )
        return summary

    def action_finished(self, action, campaign):
        summary = self.sync(campaign)
        if action.kind == "execute":
            row = next(
                r
                for r in summary["attempts"]
                if r["execution_id"] == action.payload["execution_id"]
            )
            tape = (
                "unknown"
                if row["tape_replay_ready"] is None
                else ("ready" if row["tape_replay_ready"] else "unavailable")
            )
            error = f" | error={row['error_code']}" if row["error_code"] else ""
            self.emit(
                "execution_result",
                f"Result | seed={row['seed']:03d} | {OUTCOME_NAMES.get(row['outcome'], row['outcome'])}"
                f" | tape={tape} | video={row['video_path'] or row['video_status']}{error}",
                **row,
            )
        elif action.kind == "repair":
            attempt = campaign.state()["attempts"].get(action.action_id, {})
            self.emit(
                "repair_result",
                "Repair candidate | "
                + (
                    f"Rejected: {attempt['issue']}"
                    if attempt.get("issue")
                    else "Generated; awaiting environment validation"
                ),
            )
        elif action.kind == "critic":
            report = campaign.state()["assessments"][action.payload["execution_id"]]
            self.emit(
                "critic_result",
                f"Diagnosis | usable={report['usable']} | repair frontier={report['frontier']}",
            )
        elif action.kind == "replay":
            report = campaign.state()["update"]["reports"][-1]
            self.emit(
                "replay_result", f"Historical replay | {report['status']} | {report['reason']}"
            )
        self.emit(
            "progress",
            f"Progress | main queue completed={summary['main_seeds_completed']}/{summary['main_seeds_total']}"
            f" | historically successful search seeds={summary['historically_successful_seeds']}"
            f" | completed executions={summary['completed_executions']} | pending follow-ups={summary['followups_pending']}",
        )
