import pytest
from conftest import message, text_block, tool_use_block

from healthcare_agent.agent import (
    AgentLoopLimitError,
    ConversationStore,
    SchedulingAgent,
)
from healthcare_agent.llm import FALLBACK_BETA, RefusalError


def test_tool_loop_executes_tools_and_returns_reply(fake_client, clinic, settings):
    fake_client.messages.responses = [
        message(
            [tool_use_block("tu_1", "search_patients", {"name": "Okafor"})],
            stop_reason="tool_use",
        ),
        message([text_block("I found James Okafor, born 1957-11-02.")]),
    ]
    agent = SchedulingAgent(fake_client, clinic, settings)
    conversation = ConversationStore().get_or_create(None)

    result = agent.chat(conversation, "Find James Okafor")

    assert result.reply == "I found James Okafor, born 1957-11-02."
    assert result.tool_calls == [
        {"name": "search_patients", "input": {"name": "Okafor"}, "error": False}
    ]
    second_call = fake_client.messages.calls[1]
    tool_result = second_call["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "tu_1"
    assert "P-1002" in tool_result["content"]
    # user, assistant(tool_use), user(tool_result), assistant(final)
    assert [m["role"] for m in conversation.messages] == ["user", "assistant", "user", "assistant"]


def test_request_uses_model_effort_and_fallback(fake_client, clinic, settings):
    fake_client.messages.responses = [message([text_block("Hello")])]
    agent = SchedulingAgent(fake_client, clinic, settings)
    agent.chat(ConversationStore().get_or_create(None), "hi")

    call = fake_client.messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"] == {"effort": "medium"}
    assert call["betas"] == [FALLBACK_BETA]
    assert call["fallbacks"] == "default"
    assert "tool_choice" not in call


def test_history_is_append_only_across_turns(fake_client, clinic, settings):
    fake_client.messages.responses = [message([text_block("one")]), message([text_block("two")])]
    agent = SchedulingAgent(fake_client, clinic, settings)
    conversation = ConversationStore().get_or_create(None)

    agent.chat(conversation, "first")
    first_turn = list(conversation.messages)
    agent.chat(conversation, "second")

    assert conversation.messages[: len(first_turn)] == first_turn
    assert len(fake_client.messages.calls[1]["messages"]) == 3


def test_refusal_leaves_history_untouched(fake_client, clinic, settings):
    fake_client.messages.responses = [message([], stop_reason="refusal")]
    agent = SchedulingAgent(fake_client, clinic, settings)
    conversation = ConversationStore().get_or_create(None)

    with pytest.raises(RefusalError):
        agent.chat(conversation, "something declined")
    assert conversation.messages == []


def test_loop_limit(fake_client, clinic, settings):
    settings.max_tool_iterations = 2
    fake_client.messages.responses = [
        message([tool_use_block(f"tu_{i}", "search_patients", {"name": "a"})], "tool_use")
        for i in range(2)
    ]
    agent = SchedulingAgent(fake_client, clinic, settings)
    with pytest.raises(AgentLoopLimitError):
        agent.chat(ConversationStore().get_or_create(None), "loop")
