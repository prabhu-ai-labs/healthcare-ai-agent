"""Test fixtures. A scripted fake stands in for the Anthropic client, so no API key is needed."""

from types import SimpleNamespace
from typing import Any

import pytest

from healthcare_agent.clinic import ClinicStore
from healthcare_agent.config import Settings


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_use_block(tool_id: str, name: str, tool_input: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=tool_input)


def message(content: list[Any], stop_reason: str = "end_turn", **extra: Any) -> SimpleNamespace:
    return SimpleNamespace(content=content, stop_reason=stop_reason, stop_details=None, **extra)


class FakeMessages:
    def __init__(self) -> None:
        self.responses: list[Any] = []
        self.calls: list[dict[str, Any]] = []

    def _next(self, kwargs: dict[str, Any]) -> Any:
        # Snapshot the message list: the agent keeps appending to the same list object.
        self.calls.append({**kwargs, "messages": list(kwargs.get("messages", []))})
        return self.responses.pop(0)

    def create(self, **kwargs: Any) -> Any:
        return self._next(kwargs)

    def parse(self, **kwargs: Any) -> Any:
        return self._next(kwargs)


class FakeClient:
    def __init__(self) -> None:
        self.beta = SimpleNamespace(messages=FakeMessages())

    @property
    def messages(self) -> FakeMessages:
        return self.beta.messages


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def clinic() -> ClinicStore:
    return ClinicStore.from_synthetic_data()


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)
