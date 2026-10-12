"""Conversational scheduling agent: a manual tool-use loop over the clinic tools."""

import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

import anthropic

from .clinic import ClinicStore
from .config import Settings
from .llm import raise_if_refused, request_options
from .phi import redact
from .tools import TOOLS, run_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the front-desk assistant for a multi-specialty outpatient clinic. You help staff \
look up patients and schedule appointments using the clinic tools.

- Confirm a patient's identity by name and date of birth before sharing or acting on their \
record. If several patients match, ask which one is meant.
- Before booking, restate the patient, provider, date/time and reason, and wait for the \
user's explicit confirmation.
- You do not give medical advice or triage. If the user describes an emergency (for example \
chest pain, trouble breathing, signs of stroke), tell them to call emergency services now.
- Only state facts that come from tool results; if a tool returns an error, explain it plainly.
- Keep replies short and practical."""


class AgentLoopLimitError(RuntimeError):
    """Raised when Claude keeps calling tools past the configured iteration limit."""


@dataclass
class Conversation:
    id: str
    # Full message history, including Claude's content blocks verbatim. Kept append-only:
    # earlier turns are never edited, which keeps thinking blocks valid and the cache warm.
    messages: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class ConversationStore:
    """In-memory conversation store. Replace with durable, access-controlled storage in prod."""

    def __init__(self) -> None:
        self._conversations: dict[str, Conversation] = {}
        self._lock = threading.Lock()

    def get_or_create(self, conversation_id: str | None) -> Conversation:
        with self._lock:
            if conversation_id and conversation_id in self._conversations:
                return self._conversations[conversation_id]
            conversation = Conversation(id=conversation_id or uuid.uuid4().hex)
            self._conversations[conversation.id] = conversation
            return conversation


@dataclass
class ChatResult:
    conversation_id: str
    reply: str
    tool_calls: list[dict[str, Any]]


class SchedulingAgent:
    def __init__(self, client: anthropic.Anthropic, store: ClinicStore, settings: Settings):
        self.client = client
        self.store = store
        self.settings = settings

    def chat(self, conversation: Conversation, user_message: str) -> ChatResult:
        with conversation.lock:
            # Work on a copy so a failed turn leaves the stored history unchanged.
            messages = [*conversation.messages, {"role": "user", "content": user_message}]
            tool_calls: list[dict[str, Any]] = []

            for _ in range(self.settings.max_tool_iterations):
                response = self.client.beta.messages.create(
                    **request_options(self.settings, self.settings.chat_effort),
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    cache_control={"type": "ephemeral"},
                    messages=messages,
                )
                raise_if_refused(response)
                messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason != "tool_use":
                    conversation.messages = messages
                    reply = "".join(b.text for b in response.content if b.type == "text")
                    return ChatResult(conversation.id, reply, tool_calls)

                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    content, is_error = run_tool(self.store, block.name, block.input)
                    logger.info(
                        "tool=%s error=%s input=%s", block.name, is_error, redact(str(block.input))
                    )
                    tool_calls.append({"name": block.name, "input": block.input, "error": is_error})
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": content,
                            "is_error": is_error,
                        }
                    )
                # All results for one assistant turn go back in a single user message.
                messages.append({"role": "user", "content": results})

            raise AgentLoopLimitError(
                f"Stopped after {self.settings.max_tool_iterations} tool iterations."
            )
