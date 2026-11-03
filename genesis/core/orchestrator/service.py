"""Run objectives apart from the chat that created them."""

from __future__ import annotations

import logging
import threading
from dataclasses import replace

from genesis.capabilities.manifest import ModuleError
from genesis.capabilities.service import ModuleService
from genesis.development.service import DevelopmentService
from genesis.evolution.service import ArchiveService
from genesis.cognition.learning import (
    LearningError,
    LearningService,
    gap_message,
    procedure_text,
)
from genesis.cognition.planner import PlanError, Planner, step_messages
from genesis.tools.execution.service import ExecutionService
from genesis.tools.research.actions import ResearchAction, parse_action
from genesis.tools.research.service import ResearchService
from genesis.core.configuration.settings import Settings
from genesis.core.model_interface.llama_cpp_backend import ModelLoadError
from genesis.core.model_interface.service import ModelNotReady, ModelService
from genesis.monitoring.logging import log_event
from genesis.storage.database import Database, Objective, ObjectiveEvent, StepRecord

log = logging.getLogger("genesis")
ACTION_RETRY = "That reply was not one JSON action. Return one JSON object and nothing else."


def _incomplete_action(reply: str) -> bool:
    stripped = reply.strip()
    if stripped.startswith("{") or stripped.startswith("```"):
        return parse_action(reply) is None
    return False


WEB_ACTIONS = {"search", "repositories", "package", "fetch", "read", "download", "browse"}
STUB_VALUES = {
    "directory",
    "folder",
    "module-id",
    "text",
    "path",
    "name",
    "task",
    "...",
    "the tool output",
    "the step result",
    "the module output",
    "that directory",
    "the module folder",
    "the stated task",
}
PLACEHOLDER_RESULTS = {
    "the tool output",
    "the step result",
    "the module output",
    "module output",
    "result",
    "...",
}


def _copied_instruction(value: str) -> bool:
    lowered = value.casefold().strip()
    return lowered in STUB_VALUES or "in the objective" in lowered or "earlier observation" in lowered


def _option(name: str) -> dict[str, object]:
    if name == "invoke":
        properties: dict[str, object] = {
            "action": {"const": "invoke"},
            "name": {"type": "string"},
            "task": {"type": "string"},
        }
        required = ["action", "name", "task"]
    elif name == "finish":
        properties = {"action": {"const": "finish"}, "result": {"type": "string"}}
        required = ["action", "result"]
    else:
        properties = {"action": {"const": name}, "path": {"type": "string"}}
        required = ["action", "path"]
    return {"type": "object", "properties": properties, "required": required}


def _action_format(request: str, phase: str) -> dict[str, object]:
    if not _project_module(request):
        return {"type": "json_object"}
    allowed = {"look": ["inspect"], "acquire": ["inspect", "install"]}.get(
        phase, ["inspect", "install", "invoke", "finish"]
    )
    options = [_option(name) for name in allowed]
    schema = options[0] if len(options) == 1 else {"oneOf": options}
    return {"type": "json_object", "schema": schema}


def _module_folders(listing: str, parent: str) -> list[str]:
    base = parent.strip().strip("/").replace("\\", "/")
    if not base:
        return []
    folders: list[str] = []
    prefix = base + "/"
    for line in listing.splitlines():
        path = line.strip().replace("\\", "/")
        if not path.startswith(prefix):
            continue
        rest = path[len(prefix):]
        if "/" not in rest:
            continue
        child = rest.split("/", 1)[0]
        if not child:
            continue
        folder = f"{base}/{child}"
        if folder not in folders:
            folders.append(folder)
        if len(folders) >= 8:
            break
    return folders


def _with_next_step(action: ResearchAction, observation: str, earlier: list[str]) -> str:
    if "has no manifest" in observation:
        listing = next((item for item in reversed(earlier) if "/" in item), "")
        folders = _module_folders(listing, action.path)
        if folders:
            return observation + " Install one of these folders: " + ", ".join(folders) + "."
    if action.name == "invoke" and observation.startswith("Module failed:"):
        installed = [item for item in earlier if item.startswith("Installed ")]
        if installed:
            return observation + " Use the id from this line: " + installed[-1]
    return observation


def _project_module(request: str) -> bool:
    lowered = request.casefold()
    return any(
        part in lowered
        for part in ("not installed", "under tests", "in this project", "inside this project")
    )


def _signature(action: ResearchAction) -> tuple[str, ...]:
    return (
        action.name,
        action.path.casefold(),
        action.query.casefold(),
        action.package.casefold(),
        action.task.casefold(),
        action.url.casefold(),
        action.command.casefold(),
    )


def _steer(request: str, action: ResearchAction, seen: set[tuple[str, ...]]) -> str | None:
    signature = _signature(action)
    if signature in seen:
        return "That action already ran. Choose a different path from the file list, or finish with the invoke output."
    filled = [action.path, action.package, action.task]
    if any(_copied_instruction(value) for value in filled if value):
        return (
            "That value is an instruction word, not a real path or module id. "
            "Copy the directory and the task from the objective or from an earlier observation."
        )
    if _project_module(request) and action.name in WEB_ACTIONS:
        return "The module is already in this project. Use inspect. Do not search the web."
    return None


def _defers_to_user(reply: str) -> bool:
    lowered = reply.casefold()
    return any(
        needle in lowered
        for needle in (
            "please provide",
            "necessary details",
            "more information",
            "more details",
            "i am ready to help",
            "i'm ready to help",
        )
    )


PENDING = "pending"
PLANNING = "planning"
EXECUTING = "executing"
WAITING = "waiting"
PAUSED = "paused"
COMPLETED = "completed"
FAILED = "failed"


class UnknownObjective(KeyError):
    pass


class ObjectiveStateError(RuntimeError):
    pass


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        models: ModelService,
        research: ResearchService,
        modules: ModuleService,
        development: DevelopmentService,
        archive: ArchiveService,
        execution: ExecutionService,
    ) -> None:
        self.settings = settings
        self.db = db
        self.models = models
        self.research = research
        self.modules = modules
        self.development = development
        self.archive = archive
        self.execution = execution
        self.planner = Planner(models)
        self.learning = LearningService(settings, db, models, modules)
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="genesis-objectives", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)

    def notify_model(self) -> None:
        if self.models.snapshot().loaded:
            self.db.requeue_waiting_objectives()
        self._wake.set()

    def submit(self, request: str) -> Objective:
        text = request.strip()
        if not text:
            raise ValueError("Objective is empty.")
        objective = self.db.create_objective(text)
        self._event(objective.id, "created", text)
        self._wake.set()
        return objective

    def get(self, objective_id: str) -> tuple[Objective, list[ObjectiveEvent]]:
        objective = self._require(objective_id)
        return objective, self.db.list_objective_events(objective_id)

    def list_objectives(self) -> list[Objective]:
        return self.db.list_objectives()

    def pause(self, objective_id: str) -> Objective:
        objective = self._require(objective_id)
        if objective.status in {COMPLETED, FAILED}:
            raise ObjectiveStateError(f"Cannot pause an objective that is {objective.status}.")
        if objective.status != PAUSED:
            objective = self._mark(objective, status=PAUSED)
            self._event(objective.id, "paused", "Paused.")
        return objective

    def resume(self, objective_id: str) -> Objective:
        objective = self._require(objective_id)
        if objective.status == COMPLETED:
            raise ObjectiveStateError("Cannot resume a completed objective.")
        if objective.status != PENDING:
            objective = self._mark(objective, status=PENDING, error=None, protect_pause=False)
            self._event(objective.id, "resumed", "Resumed.")
        self._wake.set()
        return objective

    def run_available(self) -> None:
        while not self._stop.is_set() and self._tick():
            continue

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._tick():
                continue
            self._wake.wait(0.5)
            self._wake.clear()

    def _tick(self) -> bool:
        if self._stop.is_set():
            return False
        objective = self.db.next_runnable_objective()
        if objective is None:
            return False
        self._execute(objective)
        return True

    def _execute(self, objective: Objective) -> None:
        if self._stop.is_set():
            return
        current = self.db.get_objective(objective.id)
        if current is None or current.status in {PAUSED, COMPLETED, FAILED, WAITING}:
            return
        if not self.models.snapshot().loaded:
            detail = self.models.snapshot().detail or "No GGUF model is loaded."
            self._mark(current, status=WAITING, error=detail)
            self._event(current.id, "waiting", detail)
            return
        try:
            current = self._prepare(current)
            if current.status == PAUSED or self._stop.is_set():
                return
            while current.current_step < len(current.steps):
                if self._stop.is_set():
                    return
                fresh = self.db.get_objective(current.id)
                if fresh is None or fresh.status == PAUSED:
                    return
                index = current.current_step
                step = current.steps[index]
                self._event(current.id, "step", step.title)
                text, _observations = self._perform_step(
                    current.id, current.request, step, index, len(current.steps)
                )
                updated = list(current.steps)
                updated[index] = StepRecord(step.title, step.instruction, text)
                current = self._mark(
                    current,
                    steps=tuple(updated),
                    current_step=index + 1,
                    status=EXECUTING,
                )
                if current.status == PAUSED:
                    return
            if self._stop.is_set():
                return
            final = self.planner.conclude(current.request, current.steps)
            if _project_module(current.request):
                for step in reversed(current.steps):
                    value = (step.result or "").strip()
                    if value and value not in final and "/" not in value and "\n" not in value and not value.endswith("."):
                        final = value
                        break
            self._mark(current, status=COMPLETED, result=final, error=None, current_step=len(current.steps))
            self._event(current.id, "completed", final[:500])
        except (PlanError, LearningError) as exc:
            self._fail(current, str(exc))
        except ModelNotReady as exc:
            self._mark(current, status=WAITING, error=exc.detail)
            self._event(current.id, "waiting", exc.detail)
        except ModelLoadError as exc:
            self._fail(current, str(exc))
        except Exception as exc:
            log.exception("objective failed")
            self._fail(current, str(exc))

    def _perform_step(
        self, objective_id: str, request: str, step: StepRecord, index: int, count: int
    ) -> tuple[str, tuple[str, ...]]:
        context = self.learning.context_for(request, objective_id)
        messages = step_messages(request, step, index, count, context)
        limit = self.settings.research.max_actions
        observations: list[str] = []
        seen: set[tuple[str, ...]] = set()
        tools = 0
        last_reply = ""
        invoke_output = ""
        phase = "look" if _project_module(request) else "call"
        for _ in range(limit + 9):
            reply = self.models.generate(
                messages,
                temperature=0.0,
                max_tokens=self.settings.model.max_tokens,
                response_format=_action_format(request, phase),
            ).content.strip()
            last_reply = reply
            action = parse_action(reply)
            deferred = _defers_to_user(reply if action is None else action.result)
            placeholder = (
                action is not None
                and action.name == "finish"
                and action.result.strip().casefold() in PLACEHOLDER_RESULTS
            )
            if placeholder or (action is None and (_incomplete_action(reply) or deferred)) or (
                action is not None and action.name == "finish" and deferred
            ):
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": ACTION_RETRY})
                continue
            if action is None or action.name == "finish":
                text = reply if action is None else action.result
                if invoke_output and invoke_output not in text:
                    text = invoke_output
                if not text:
                    raise PlanError(f"Step '{step.title}' returned an empty result.", tuple(observations))
                return text, tuple(observations)
            if tools >= limit:
                break
            signature = _signature(action)
            steer = _steer(request, action, seen)
            seen.add(signature)
            if steer is not None:
                self._event(objective_id, action.name, steer)
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": f"Observation:\n{steer}"})
                continue
            tools += 1
            if action.name in {"inventory", "install", "invoke", "remember", "skills"}:
                observation = self.modules.execute(objective_id, action)
            elif action.name in {"inspect", "test_candidate", "compare", "archive"}:
                observation = self.development.execute(objective_id, action)
            elif action.name in {"propose", "retain", "lineage", "evaluate", "select", "revisit"}:
                observation = self.archive.execute(objective_id, action)
            elif action.name == "run":
                observation = self.execution.execute(objective_id, action)
            else:
                observation = self.research.execute(objective_id, action)
            observation = _with_next_step(action, observation, observations)
            observations.append(observation)
            self._event(objective_id, action.name, observation)
            if action.name == "inspect" and not observation.startswith("Modification failed") and phase == "look":
                phase = "acquire"
            if action.name == "install" and observation.startswith("Installed "):
                phase = "call"
            if action.name == "invoke" and not observation.startswith("Module failed"):
                invoke_output = observation.strip()
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"Observation:\n{observation}"})
        raise PlanError(
            f"Step '{step.title}' used more than {limit} research actions. Last reply: {last_reply[:300]}",
            tuple(observations),
        )

    def _learn(self, current: Objective) -> None:
        if self.db.list_gaps(current.id) or self.db.list_experiments(current.id):
            return
        try:
            assessment = self.learning.assess(current.request)
        except LearningError as exc:
            self.db.save_gap(current.id, [], [])
            self._event(current.id, "gap", f"Learning assessment was not usable. {exc}"[:500])
            return
        self.db.save_gap(current.id, list(assessment.missing), list(assessment.known))
        self._event(current.id, "gap", gap_message(assessment))
        for spec in assessment.experiments:
            if self._stop.is_set():
                return
            fresh = self.db.get_objective(current.id)
            if fresh is None or fresh.status == PAUSED:
                return
            try:
                text, observations = self._perform_step(
                    current.id,
                    current.request,
                    StepRecord(spec.name, spec.probe),
                    0,
                    1,
                )
            except PlanError as exc:
                text = str(exc)
                observations = exc.observations
            outcome = "\n".join(part for part in (text, *observations) if part)
            success = spec.expect.casefold() in outcome.casefold()
            self.db.save_experiment(
                current.id,
                spec.name,
                spec.hypothesis,
                spec.probe,
                spec.expect,
                outcome[:4000],
                success,
            )
            self._event(current.id, "experiment", f"{spec.name} {'succeeded' if success else 'failed'}.")
            if not success:
                continue
            try:
                self.modules.remember(spec.name, procedure_text(spec, outcome[:4000]))
            except ModuleError as exc:
                self._event(current.id, "retain-failed", str(exc))
                continue
            self._event(current.id, "retained", spec.name)

    def _prepare(self, current: Objective) -> Objective:
        if current.steps:
            return self._mark(current, status=EXECUTING, error=None)
        current = self._mark(current, status=PLANNING, error=None)
        self._event(current.id, "planning", "Planning steps.")
        if self._stop.is_set():
            return current
        self._learn(current)
        fresh = self.db.get_objective(current.id)
        if fresh is None or fresh.status == PAUSED or self._stop.is_set():
            return fresh or current
        context = self.learning.context_for(fresh.request, fresh.id)
        steps = self.planner.plan(fresh.request, context)
        fresh = self.db.get_objective(fresh.id)
        status = PAUSED if fresh is not None and fresh.status == PAUSED else EXECUTING
        current = self._mark(current, steps=steps, current_step=0, status=status)
        self._event(current.id, "planned", f"{len(steps)} steps.")
        return current

    def _fail(self, objective: Objective, message: str) -> None:
        self._mark(objective, status=FAILED, error=message)
        self._event(objective.id, "failed", message)

    def _mark(self, objective: Objective, *, protect_pause: bool = True, **changes: object) -> Objective:
        fresh = self._require(objective.id)
        if protect_pause and fresh.status == PAUSED and changes.get("status") in {PENDING, PLANNING, EXECUTING}:
            changes["status"] = PAUSED
        updated = replace(fresh, **changes)
        return self.db.save_objective(updated)

    def _require(self, objective_id: str) -> Objective:
        objective = self.db.get_objective(objective_id)
        if objective is None:
            raise UnknownObjective(objective_id)
        return objective

    def _event(self, objective_id: str, kind: str, message: str) -> None:
        self.db.add_objective_event(objective_id, kind, message)
        log_event(logging.INFO, "objective", objective_id=objective_id, kind=kind, detail=message[:500])
