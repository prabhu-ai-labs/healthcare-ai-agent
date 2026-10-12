"""FastAPI service exposing the chat agent and document extraction."""

import logging
from contextlib import asynccontextmanager

import anthropic
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .agent import AgentLoopLimitError, ConversationStore, SchedulingAgent
from .clinic import ClinicStore
from .config import Settings, get_settings
from .extraction import IntakeExtraction, UnsupportedDocumentError, extract_intake
from .llm import RefusalError, build_client

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = build_client()
    app.state.clinic = ClinicStore.from_synthetic_data()
    app.state.conversations = ConversationStore()
    yield


app = FastAPI(title="Healthcare AI Agent", version=__version__, lifespan=lifespan)


def get_client(request: Request) -> anthropic.Anthropic:
    return request.app.state.client


def get_agent(
    request: Request,
    client: anthropic.Anthropic = Depends(get_client),
    settings: Settings = Depends(get_settings),
) -> SchedulingAgent:
    return SchedulingAgent(client, request.app.state.clinic, settings)


@app.exception_handler(RefusalError)
async def _refusal(_: Request, exc: RefusalError) -> JSONResponse:
    return JSONResponse(
        status_code=422, content={"detail": "The request was declined.", "category": exc.category}
    )


@app.exception_handler(AgentLoopLimitError)
async def _loop_limit(_: Request, exc: AgentLoopLimitError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(anthropic.APIError)
async def _upstream(_: Request, exc: anthropic.APIError) -> JSONResponse:
    logger.error("Claude API error: %s", type(exc).__name__)
    retryable = isinstance(exc, anthropic.RateLimitError | anthropic.APIConnectionError)
    status = 503 if retryable else 502
    return JSONResponse(status_code=status, content={"detail": "Upstream model error."})


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    tool_calls: list[dict]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest, request: Request, agent: SchedulingAgent = Depends(get_agent)
) -> ChatResponse:
    conversation = request.app.state.conversations.get_or_create(body.conversation_id)
    result = agent.chat(conversation, body.message)
    return ChatResponse(
        conversation_id=result.conversation_id, reply=result.reply, tool_calls=result.tool_calls
    )


@app.post("/extract/intake", response_model=IntakeExtraction)
def extract(
    file: UploadFile = File(...),
    client: anthropic.Anthropic = Depends(get_client),
    settings: Settings = Depends(get_settings),
) -> IntakeExtraction:
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB).")
    if not content:
        raise HTTPException(status_code=400, detail="File is empty.")
    media_type = (file.content_type or "").split(";")[0].strip()
    try:
        return extract_intake(client, settings, content, media_type)
    except UnsupportedDocumentError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
