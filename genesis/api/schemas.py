from pydantic import BaseModel, Field


class ModelStatusResponse(BaseModel):
    loaded: bool
    directory: str
    filename: str | None
    discovered: list[str]
    context_length: int
    n_gpu_layers: int
    detail: str | None


class SelectModelRequest(BaseModel):
    filename: str = Field(min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str | None = None
    temperature: float | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, gt=0)


class ChatResponse(BaseModel):
    conversation_id: str
    message: str
    model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    duration_ms: float


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    created_at: str


class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[MessageResponse] | None = None


class HealthResponse(BaseModel):
    status: str
    venv: str
    phase: str


class ObjectiveRequest(BaseModel):
    request: str = Field(min_length=1)


class StepResponse(BaseModel):
    title: str
    instruction: str
    result: str | None = None


class ObjectiveEventResponse(BaseModel):
    id: str
    kind: str
    message: str
    created_at: str


class ModuleResponse(BaseModel):
    id: str
    name: str
    version: str
    description: str
    status: str
    path: str
    error: str | None
    installed_at: str
    updated_at: str


class ModuleRequest(BaseModel):
    path: str = Field(min_length=1)


class InvokeRequest(BaseModel):
    task: str = Field(min_length=1)


class SkillResponse(BaseModel):
    id: str
    name: str
    body: str
    created_at: str
    updated_at: str


class SkillRequest(BaseModel):
    name: str = Field(min_length=1)
    body: str = Field(min_length=1)


class CapabilityResponse(BaseModel):
    name: str
    kind: str
    status: str
    description: str
    version: str | None = None


class ExperimentResponse(BaseModel):
    id: str
    objective_id: str
    name: str
    hypothesis: str
    probe: str
    expected: str
    outcome: str
    success: bool
    created_at: str


class ResourceResponse(BaseModel):
    id: str
    objective_id: str | None
    kind: str
    title: str
    url: str | None
    path: str | None
    sha256: str | None
    metadata: dict[str, object]
    retrieved_at: str


class ObjectiveResponse(BaseModel):
    id: str
    request: str
    status: str
    current_step: int
    steps: list[StepResponse]
    result: str | None
    error: str | None
    created_at: str
    updated_at: str
    events: list[ObjectiveEventResponse] | None = None
