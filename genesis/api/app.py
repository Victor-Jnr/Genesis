"""FastAPI surface. JSON routes and the HTML workstation."""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from genesis import PHASE, __version__
from genesis.api.schemas import (
    CandidateResponse,
    CapabilityResponse,
    EvaluateRequest,
    GenerationResponse,
    ChatRequest,
    CommandRequest,
    ConfigResponse,
    ChatResponse,
    ConversationResponse,
    ExperimentResponse,
    HealthResponse,
    InvokeRequest,
    MessageResponse,
    ModelStatusResponse,
    ObjectiveEventResponse,
    ModuleRequest,
    ModuleResponse,
    ObjectiveRequest,
    ObjectiveResponse,
    ProposalRequest,
    ResourceResponse,
    SkillRequest,
    SkillResponse,
    SelectModelRequest,
    StepResponse,
)
from genesis.capabilities.manifest import ModuleError
from genesis.capabilities.service import BUILTIN_CAPABILITIES, ModuleService
from genesis.development.service import DevelopmentService, ModificationError
from genesis.evolution.service import ArchiveError, ArchiveService
from genesis.core.conversation import ConversationService, UnknownConversation
from genesis.core.model_interface.llama_cpp_backend import ModelLoadError
from genesis.core.model_interface.service import ConfigPinned, ModelNotReady, ModelService, ModelSnapshot
from genesis.core.orchestrator.service import ObjectiveStateError, Orchestrator, UnknownObjective
from genesis.core.runtime import repo_root
from genesis.monitoring.logging import log_event
from genesis.storage.database import (
    Candidate,
    Database,
    Experiment,
    Generation,
    ModuleRecord,
    Objective,
    ObjectiveEvent,
    Resource,
    Skill,
)
from genesis.tools.execution.service import ExecutionError, ExecutionService
from genesis.tools.research.service import ResearchService
from genesis.workstation.pages import register_workstation

log = logging.getLogger("genesis")


def create_app(settings_root: PathContext) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.context.models.startup()
        app.state.context.modules.startup()
        app.state.context.orchestrator.start()
        yield
        app.state.context.orchestrator.stop()
        app.state.context.models.shutdown()

    app = FastAPI(title="Genesis", version=__version__, lifespan=lifespan)
    app.state.context = settings_root

    @app.middleware("http")
    async def log_request(request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        log_event(
            logging.INFO,
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return response

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", venv=sys.prefix, phase=PHASE)

    @app.get("/config", response_model=ConfigResponse)
    def config_snapshot() -> ConfigResponse:
        settings = app.state.context.settings
        return ConfigResponse(
            identity_name=settings.identity.name,
            server_host=settings.server.host,
            server_port=settings.server.port,
            model_directory=settings.model.directory,
            model_filename=settings.model.filename,
            context_length=settings.model.context_length,
            n_gpu_layers=settings.model.n_gpu_layers,
            temperature=settings.model.temperature,
            top_p=settings.model.top_p,
            max_tokens=settings.model.max_tokens,
            database_path=settings.database.path,
            logging_level=settings.logging.level,
            logging_directory=settings.logging.directory,
            research_download_directory=settings.research.download_directory,
            research_max_actions=settings.research.max_actions,
            modules_directory=settings.modules.directory,
            skills_directory=settings.modules.skills_directory,
            candidates_directory=settings.development.candidates_directory,
            test_timeout_seconds=settings.development.test_timeout_seconds,
            evolution_benchmark=settings.evolution.benchmark,
            learning_max_experiments=settings.learning.max_experiments,
            execution_enabled=settings.execution.enabled,
            lab_enabled=settings.lab.enabled,
        )

    @app.get("/model", response_model=ModelStatusResponse)
    def model_status() -> ModelStatusResponse:
        return _status(app.state.context.models.snapshot())

    @app.post("/model/reload", response_model=ModelStatusResponse)
    def model_reload() -> ModelStatusResponse:
        status = _status(app.state.context.models.reload())
        app.state.context.orchestrator.notify_model()
        return status

    @app.post("/model/select", response_model=ModelStatusResponse)
    def model_select(body: SelectModelRequest) -> ModelStatusResponse:
        try:
            status = _status(app.state.context.models.select(body.filename))
        except ConfigPinned as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ModelLoadError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        app.state.context.orchestrator.notify_model()
        return status

    @app.post("/objectives", response_model=ObjectiveResponse)
    def create_objective(body: ObjectiveRequest) -> ObjectiveResponse:
        try:
            objective = app.state.context.orchestrator.submit(body.request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _objective_response(objective, app.state.context.db.list_objective_events(objective.id))

    @app.get("/objectives", response_model=list[ObjectiveResponse])
    def list_objectives() -> list[ObjectiveResponse]:
        return [_objective_response(item) for item in app.state.context.orchestrator.list_objectives()]

    @app.get("/objectives/{objective_id}", response_model=ObjectiveResponse)
    def get_objective(objective_id: str) -> ObjectiveResponse:
        try:
            objective, events = app.state.context.orchestrator.get(objective_id)
        except UnknownObjective as exc:
            raise HTTPException(status_code=404, detail="Objective not found.") from exc
        return _objective_response(objective, events)

    @app.post("/objectives/{objective_id}/pause", response_model=ObjectiveResponse)
    def pause_objective(objective_id: str) -> ObjectiveResponse:
        return _objective_action(app, objective_id, "pause")

    @app.post("/objectives/{objective_id}/resume", response_model=ObjectiveResponse)
    def resume_objective(objective_id: str) -> ObjectiveResponse:
        return _objective_action(app, objective_id, "resume")

    @app.get("/objectives/{objective_id}/experiments", response_model=list[ExperimentResponse])
    def objective_experiments(objective_id: str) -> list[ExperimentResponse]:
        if app.state.context.db.get_objective(objective_id) is None:
            raise HTTPException(status_code=404, detail="Objective not found.")
        return [_experiment_response(item) for item in app.state.context.db.list_experiments(objective_id)]

    @app.get("/objectives/{objective_id}/resources", response_model=list[ResourceResponse])
    def objective_resources(objective_id: str) -> list[ResourceResponse]:
        if app.state.context.db.get_objective(objective_id) is None:
            raise HTTPException(status_code=404, detail="Objective not found.")
        return [_resource_response(item) for item in app.state.context.db.list_resources(objective_id)]

    @app.get("/resources", response_model=list[ResourceResponse])
    def list_resources() -> list[ResourceResponse]:
        return [_resource_response(item) for item in app.state.context.db.list_resources()]

    @app.get("/modules", response_model=list[ModuleResponse])
    def list_modules() -> list[ModuleResponse]:
        return [_module_response(item) for item in app.state.context.db.list_modules()]

    @app.post("/modules/install", response_model=ModuleResponse)
    def install_module(body: ModuleRequest) -> ModuleResponse:
        try:
            record = app.state.context.modules.install_path(body.path)
        except ModuleError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _module_response(record)

    @app.post("/modules/{module_id}/invoke", response_model=dict[str, str])
    def invoke_module(module_id: str, body: InvokeRequest) -> dict[str, str]:
        try:
            result = app.state.context.modules.invoke(None, module_id, body.task)
        except ModuleError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"module": module_id, "result": result}

    @app.get("/source")
    def read_source(path: str) -> dict[str, str]:
        try:
            text = app.state.context.development.inspect(path)
        except ModificationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"path": path, "text": text}

    @app.post("/proposals", response_model=CandidateResponse)
    def create_proposal(body: ProposalRequest) -> CandidateResponse:
        try:
            _proposal, candidate, generation = app.state.context.archive.propose(
                None,
                body.path,
                body.rationale,
                body.old,
                body.text,
                body.parent,
            )
        except (ModificationError, ArchiveError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        response = _candidate_response(app.state.context.db, candidate)
        return response.model_copy(update={"generation_id": generation.id})

    @app.get("/candidates", response_model=list[CandidateResponse])
    def list_candidates() -> list[CandidateResponse]:
        database = app.state.context.db
        return [_candidate_response(database, item) for item in database.list_candidates()]

    @app.get("/candidates/{candidate_id}", response_model=CandidateResponse)
    def get_candidate(candidate_id: str) -> CandidateResponse:
        candidate = app.state.context.db.get_candidate(candidate_id)
        if candidate is None:
            raise HTTPException(status_code=404, detail="Candidate not found.")
        return _candidate_response(app.state.context.db, candidate)

    @app.post("/candidates/{candidate_id}/test", response_model=CandidateResponse)
    def test_candidate(candidate_id: str) -> CandidateResponse:
        try:
            candidate = app.state.context.development.test(candidate_id)
        except ModificationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _candidate_response(app.state.context.db, candidate)

    @app.post("/candidates/{candidate_id}/retain", response_model=CandidateResponse)
    def retain_candidate(candidate_id: str) -> CandidateResponse:
        try:
            app.state.context.archive.retain(candidate_id)
        except (ModificationError, ArchiveError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        candidate = app.state.context.db.get_candidate(candidate_id)
        if candidate is None:
            generation = app.state.context.db.get_generation(candidate_id)
            if generation is None or generation.candidate_id is None:
                raise HTTPException(status_code=404, detail="Candidate not found.")
            candidate = app.state.context.db.get_candidate(generation.candidate_id)
        if candidate is None:
            raise HTTPException(status_code=404, detail="Candidate not found.")
        return _candidate_response(app.state.context.db, candidate)

    @app.post("/candidates/{candidate_id}/evaluate", response_model=GenerationResponse)
    def evaluate_candidate(candidate_id: str) -> GenerationResponse:
        try:
            app.state.context.archive.evaluate(candidate_id)
        except (ModificationError, ArchiveError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        generation = app.state.context.db.get_generation_for_candidate(candidate_id)
        if generation is None:
            generation = app.state.context.db.get_generation(candidate_id)
        if generation is None:
            raise HTTPException(status_code=404, detail="Generation not found.")
        return _generation_response(
            app.state.context.db,
            generation,
            app.state.context.settings.evolution.benchmark,
        )

    @app.get("/archive", response_model=list[GenerationResponse])
    def list_archive() -> list[GenerationResponse]:
        database = app.state.context.db
        benchmark = app.state.context.settings.evolution.benchmark
        return [_generation_response(database, item, benchmark) for item in database.list_generations()]

    @app.get("/archive/{generation_id}", response_model=GenerationResponse)
    def get_generation(generation_id: str) -> GenerationResponse:
        generation = app.state.context.db.get_generation(generation_id)
        if generation is None:
            raise HTTPException(status_code=404, detail="Generation not found.")
        return _generation_response(
            app.state.context.db,
            generation,
            app.state.context.settings.evolution.benchmark,
        )

    @app.get("/archive/{generation_id}/source")
    def revisit_generation(generation_id: str, path: str) -> dict[str, str]:
        try:
            text = app.state.context.archive.revisit(generation_id, path)
        except (ModificationError, ArchiveError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"path": path, "text": text}

    @app.post("/archive/evaluate", response_model=GenerationResponse)
    def evaluate_archive(body: EvaluateRequest) -> GenerationResponse:
        name = body.benchmark or app.state.context.settings.evolution.benchmark
        try:
            app.state.context.archive.evaluate(body.target, name)
        except (ModificationError, ArchiveError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        generation = app.state.context.archive._resolve(body.target)
        return _generation_response(app.state.context.db, generation, name)

    @app.post("/archive/select", response_model=GenerationResponse)
    def select_parent(benchmark: str | None = None) -> GenerationResponse:
        try:
            app.state.context.archive.select(benchmark)
        except ArchiveError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        name = benchmark or app.state.context.settings.evolution.benchmark
        chosen = app.state.context.db.latest_selection(name)
        if chosen is None:
            raise HTTPException(status_code=404, detail="No parent was selected.")
        generation = app.state.context.db.get_generation(chosen.generation_id)
        if generation is None:
            raise HTTPException(status_code=404, detail="Generation not found.")
        return _generation_response(app.state.context.db, generation, name)

    @app.post("/commands", response_model=dict[str, str])
    def run_command(body: CommandRequest) -> dict[str, str]:
        try:
            output = app.state.context.execution.run(body.command)
        except ExecutionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"output": output}

    @app.get("/capabilities", response_model=list[CapabilityResponse])
    def capabilities() -> list[CapabilityResponse]:
        return _capabilities(app.state.context.modules)

    @app.get("/skills", response_model=list[SkillResponse])
    def list_skills() -> list[SkillResponse]:
        return [_skill_response(item) for item in app.state.context.db.list_skills()]

    @app.post("/skills", response_model=SkillResponse)
    def remember_skill(body: SkillRequest) -> SkillResponse:
        try:
            skill = app.state.context.modules.remember(body.name, body.body)
        except ModuleError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _skill_response(skill)

    @app.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest) -> ChatResponse:
        try:
            turn = app.state.context.conversations.send(
                body.message,
                body.conversation_id,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            )
        except UnknownConversation as exc:
            raise HTTPException(status_code=404, detail="Conversation not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ModelNotReady as exc:
            raise HTTPException(
                status_code=409,
                detail={"message": exc.detail, "discovered": exc.discovered},
            ) from exc
        except ModelLoadError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return ChatResponse(
            conversation_id=turn.conversation_id,
            message=turn.result.content,
            model=turn.result.model_filename,
            prompt_tokens=turn.result.prompt_tokens,
            completion_tokens=turn.result.completion_tokens,
            duration_ms=turn.result.duration_ms,
        )

    @app.post("/chat/stream")
    def chat_stream(body: ChatRequest) -> StreamingResponse:
        try:
            events = app.state.context.conversations.stream(
                body.message,
                body.conversation_id,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            )
            first = next(events)
        except UnknownConversation as exc:
            raise HTTPException(status_code=404, detail="Conversation not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ModelNotReady as exc:
            raise HTTPException(
                status_code=409,
                detail={"message": exc.detail, "discovered": exc.discovered},
            ) from exc
        except ModelLoadError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except StopIteration:
            raise HTTPException(status_code=502, detail="Model returned no stream.") from None

        def encoded() -> Iterator[str]:
            yield _sse(first)
            for event in events:
                yield _sse(event)

        return StreamingResponse(encoded(), media_type="text/event-stream")

    @app.get("/conversations", response_model=list[ConversationResponse])
    def list_conversations() -> list[ConversationResponse]:
        return [
            ConversationResponse(
                id=item.id,
                title=item.title,
                created_at=item.created_at,
                updated_at=item.updated_at,
            )
            for item in app.state.context.conversations.list_conversations()
        ]

    @app.get("/conversations/{conversation_id}", response_model=ConversationResponse)
    def get_conversation(conversation_id: str) -> ConversationResponse:
        try:
            conversation, messages = app.state.context.conversations.get_conversation(conversation_id)
        except UnknownConversation as exc:
            raise HTTPException(status_code=404, detail="Conversation not found.") from exc
        return ConversationResponse(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            messages=[
                MessageResponse(id=item.id, role=item.role, content=item.content, created_at=item.created_at)
                for item in messages
            ],
        )

    register_workstation(app)
    return app


class PathContext:
    def __init__(
        self,
        root,
        settings,
        db: Database,
        models: ModelService,
        research: ResearchService,
        modules: ModuleService,
        development: DevelopmentService,
        archive: ArchiveService,
        execution: ExecutionService,
    ) -> None:
        self.root = root
        self.settings = settings
        self.db = db
        self.models = models
        self.research = research
        self.modules = modules
        self.development = development
        self.archive = archive
        self.execution = execution
        self.conversations = ConversationService(settings, db, models)
        self.orchestrator = Orchestrator(settings, db, models, research, modules, development, archive, execution)


def build_context(root, settings, backend, http=None, module_runner=None, candidate_runner=None, command_runner=None) -> PathContext:
    db_path = settings.resolve_path(settings.database.path, root)
    db = Database(db_path)
    db.migrate()
    models = ModelService(settings, db, backend, root)
    research = ResearchService(settings, db, root, http)
    modules = ModuleService(settings, db, root, module_runner)
    development = DevelopmentService(settings, db, root, candidate_runner)
    archive = ArchiveService(settings, db, development)
    execution = ExecutionService(settings, root, command_runner)
    return PathContext(root, settings, db, models, research, modules, development, archive, execution)


def build_runtime_context(backend=None) -> PathContext:
    from genesis.core.configuration.settings import load_settings
    from genesis.core.model_interface.service import default_backend

    root = repo_root()
    settings = load_settings(root=root)
    return build_context(root, settings, backend or default_backend())


def _status(snapshot: ModelSnapshot) -> ModelStatusResponse:
    return ModelStatusResponse(
        loaded=snapshot.loaded,
        directory=snapshot.directory,
        filename=snapshot.filename,
        discovered=snapshot.discovered,
        context_length=snapshot.context_length,
        n_gpu_layers=snapshot.n_gpu_layers,
        detail=snapshot.detail,
    )


def _objective_action(app: FastAPI, objective_id: str, action: str) -> ObjectiveResponse:
    try:
        orchestrator = app.state.context.orchestrator
        objective = orchestrator.pause(objective_id) if action == "pause" else orchestrator.resume(objective_id)
    except UnknownObjective as exc:
        raise HTTPException(status_code=404, detail="Objective not found.") from exc
    except ObjectiveStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _objective_response(objective, app.state.context.db.list_objective_events(objective.id))


def _objective_response(objective: Objective, events: list[ObjectiveEvent] | None = None) -> ObjectiveResponse:
    return ObjectiveResponse(
        id=objective.id,
        request=objective.request,
        status=objective.status,
        current_step=objective.current_step,
        steps=[
            StepResponse(title=step.title, instruction=step.instruction, result=step.result)
            for step in objective.steps
        ],
        result=objective.result,
        error=objective.error,
        created_at=objective.created_at,
        updated_at=objective.updated_at,
        events=None
        if events is None
        else [
            ObjectiveEventResponse(id=event.id, kind=event.kind, message=event.message, created_at=event.created_at)
            for event in events
        ],
    )


def _experiment_response(experiment: Experiment) -> ExperimentResponse:
    return ExperimentResponse(
        id=experiment.id,
        objective_id=experiment.objective_id,
        name=experiment.name,
        hypothesis=experiment.hypothesis,
        probe=experiment.probe,
        expected=experiment.expected,
        outcome=experiment.outcome,
        success=experiment.success,
        created_at=experiment.created_at,
    )


def _resource_response(resource: Resource) -> ResourceResponse:
    try:
        metadata = json.loads(resource.metadata_json)
    except json.JSONDecodeError:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    return ResourceResponse(
        id=resource.id,
        objective_id=resource.objective_id,
        kind=resource.kind,
        title=resource.title,
        url=resource.url,
        path=resource.path,
        sha256=resource.sha256,
        metadata=metadata,
        retrieved_at=resource.retrieved_at,
    )


def _module_response(record: ModuleRecord) -> ModuleResponse:
    return ModuleResponse(
        id=record.id,
        name=record.name,
        version=record.version,
        description=record.description,
        status=record.status,
        path=record.path,
        error=record.error,
        installed_at=record.installed_at,
        updated_at=record.updated_at,
    )


def _skill_response(skill: Skill) -> SkillResponse:
    return SkillResponse(
        id=skill.id,
        name=skill.name,
        body=skill.body,
        created_at=skill.created_at,
        updated_at=skill.updated_at,
    )


def _is_selected(database: Database, benchmark: str, generation_id: str) -> bool:
    chosen = database.latest_selection(benchmark)
    return chosen is not None and chosen.generation_id == generation_id


def _generation_response(
    database: Database,
    generation: Generation,
    benchmark: str | None = None,
) -> GenerationResponse:
    name = benchmark or "tests"
    latest = database.latest_evaluation(generation.id, name)
    return GenerationResponse(
        id=generation.id,
        candidate_id=generation.candidate_id,
        parent_id=generation.parent_id,
        label=generation.label,
        status=generation.status,
        created_at=generation.created_at,
        benchmark=None if latest is None else latest.benchmark,
        success=None if latest is None else latest.success,
        elapsed_ms=None if latest is None else latest.elapsed_ms,
        selected=_is_selected(database, name, generation.id),
    )


def _candidate_response(database: Database, candidate: Candidate) -> CandidateResponse:
    proposal = database.get_proposal(candidate.proposal_id)
    return CandidateResponse(
        id=candidate.id,
        proposal_id=candidate.proposal_id,
        parent_revision=candidate.parent_revision,
        directory=candidate.directory,
        status=candidate.status,
        report=candidate.report,
        elapsed_ms=candidate.elapsed_ms,
        created_at=candidate.created_at,
        path=None if proposal is None else proposal.path,
        rationale=None if proposal is None else proposal.rationale,
    )


def _capabilities(modules: ModuleService) -> list[CapabilityResponse]:
    items = [
        CapabilityResponse(name=name, kind="built-in", status="active", description=description)
        for name, description in BUILTIN_CAPABILITIES
    ]
    for module in modules.db.list_modules():
        items.append(
            CapabilityResponse(
                name=module.id,
                kind="module",
                status=module.status,
                description=module.description,
                version=module.version,
            )
        )
    return items


def _sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload)}\n\n"
