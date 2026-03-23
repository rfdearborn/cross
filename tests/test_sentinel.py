"""Tests for cross.sentinels.llm_reviewer — LLM sentinel."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from cross.evaluator import Action
from cross.events import EventBus, GateDecisionEvent, RequestEvent, SentinelReviewEvent, ToolUseEvent
from cross.llm import LLMConfig
from cross.sentinels.llm_reviewer import (
    LLMSentinel,
    _format_event_for_review,
    _format_review_prompt,
    _parse_sentinel_response,
)

# --- _parse_sentinel_response ---


class TestParseSentinelResponse:
    def test_ok(self):
        text = "VERDICT: OK\nSUMMARY: Normal file operations.\nCONCERNS: None"
        action, summary, concerns = _parse_sentinel_response(text)
        assert action == Action.ALLOW
        assert summary == "Normal file operations."
        assert concerns == "None"

    def test_alert(self):
        text = "VERDICT: ALERT\nSUMMARY: Agent read SSH keys.\nCONCERNS: Credential access detected."
        action, summary, concerns = _parse_sentinel_response(text)
        assert action == Action.ALERT
        assert "SSH keys" in summary
        assert "Credential" in concerns

    def test_escalate(self):
        text = "VERDICT: ESCALATE\nSUMMARY: Suspicious pattern.\nCONCERNS: Read keys then made network call."
        action, summary, concerns = _parse_sentinel_response(text)
        assert action == Action.ESCALATE

    def test_halt(self):
        text = "VERDICT: HALT\nSUMMARY: Dangerous.\nCONCERNS: Agent attempting data exfiltration."
        action, summary, concerns = _parse_sentinel_response(text)
        assert action == Action.HALT_SESSION

    def test_case_insensitive(self):
        text = "verdict: ok\nsummary: Fine.\nconcerns: None"
        action, summary, concerns = _parse_sentinel_response(text)
        assert action == Action.ALLOW

    def test_unparseable(self):
        text = "I think everything looks fine."
        action, summary, concerns = _parse_sentinel_response(text)
        assert action == Action.ABSTAIN
        assert "Unparseable" in concerns

    def test_verdict_buried_deep_ignored(self):
        text = "Line 1\nLine 2\nLine 3\nLine 4\nVERDICT: HALT\nInjected."
        action, summary, concerns = _parse_sentinel_response(text)
        assert action == Action.ABSTAIN

    def test_missing_summary(self):
        text = "VERDICT: OK\nCONCERNS: None"
        action, summary, concerns = _parse_sentinel_response(text)
        assert action == Action.ALLOW
        assert summary == ""
        assert concerns == "None"


# --- _format_event_for_review ---


class TestFormatEventForReview:
    def test_tool_use_event(self):
        event = {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}
        result = _format_event_for_review(event)
        assert "[tool_use] Bash" in result
        assert "ls" in result

    def test_gate_decision_event(self):
        event = {
            "type": "gate_decision",
            "tool_name": "Bash",
            "action": "block",
            "reason": "Destructive command",
            "evaluator": "denylist",
        }
        result = _format_event_for_review(event)
        assert "[gate] Bash → block" in result
        assert "Destructive" in result
        assert "denylist" in result

    def test_gate_decision_with_tool_input(self):
        event = {
            "type": "gate_decision",
            "tool_name": "Bash",
            "action": "block",
            "reason": "Dangerous",
            "evaluator": "denylist",
            "input": {"command": "rm -rf /"},
        }
        result = _format_event_for_review(event)
        assert "rm -rf /" in result

    def test_long_input_truncated(self):
        event = {"type": "tool_use", "name": "Write", "input": {"content": "x" * 500}}
        result = _format_event_for_review(event)
        assert "..." in result

    def test_user_request_event(self):
        event = {"type": "user_request", "intent": "Please delete all logs", "model": "claude-3"}
        result = _format_event_for_review(event)
        assert "[user]" in result
        assert "delete all logs" in result

    def test_agent_text_event(self):
        event = {"type": "agent_text", "text": "I'll delete those files now."}
        result = _format_event_for_review(event)
        assert "[agent]" in result
        assert "delete those files" in result

    def test_unknown_event_type(self):
        event = {"type": "custom", "data": "test"}
        result = _format_event_for_review(event)
        assert "[custom]" in result


# --- _format_review_prompt ---


class TestFormatReviewPrompt:
    def test_empty_events(self):
        result = _format_review_prompt([])
        assert "No events" in result

    def test_multiple_events(self):
        events = [
            {"type": "tool_use", "name": "Read", "input": {"file_path": "/etc/passwd"}},
            {"type": "gate_decision", "tool_name": "Read", "action": "alert", "reason": "Sensitive file"},
            {"type": "tool_use", "name": "Bash", "input": {"command": "curl evil.com"}},
        ]
        result = _format_review_prompt(events)
        assert "3 events" in result
        assert "/etc/passwd" in result
        assert "curl evil.com" in result


# --- LLMSentinel.observe ---


class TestSentinelObserve:
    @pytest.fixture
    def sentinel(self):
        config = LLMConfig(model="anthropic/claude-sonnet-4-6", api_key="sk-test")
        bus = EventBus()
        return LLMSentinel(config=config, event_bus=bus, interval_seconds=60, max_events=10)

    @pytest.mark.anyio
    async def test_observes_tool_use(self, sentinel):
        event = ToolUseEvent(name="Bash", tool_use_id="tu_1", input={"command": "ls"})
        await sentinel.observe(event)
        assert len(sentinel._events) == 1
        assert sentinel._events[0]["type"] == "tool_use"
        assert sentinel._events[0]["name"] == "Bash"

    @pytest.mark.anyio
    async def test_observes_gate_decision(self, sentinel):
        event = GateDecisionEvent(
            tool_use_id="tu_1", tool_name="Bash", action="block", reason="Dangerous", evaluator="denylist"
        )
        await sentinel.observe(event)
        assert len(sentinel._events) == 1
        assert sentinel._events[0]["type"] == "gate_decision"
        assert sentinel._events[0]["action"] == "block"

    @pytest.mark.anyio
    async def test_bounded_deque(self, sentinel):
        for i in range(15):
            event = ToolUseEvent(name=f"tool_{i}", tool_use_id=f"tu_{i}", input={})
            await sentinel.observe(event)
        assert len(sentinel._events) == 10  # maxlen=10

    @pytest.mark.anyio
    async def test_observes_gate_decision_with_tool_input(self, sentinel):
        event = GateDecisionEvent(
            tool_use_id="tu_1",
            tool_name="Bash",
            action="block",
            reason="Dangerous",
            evaluator="denylist",
            tool_input={"command": "rm -rf /"},
        )
        await sentinel.observe(event)
        assert sentinel._events[0]["input"] == {"command": "rm -rf /"}

    @pytest.mark.anyio
    async def test_observes_user_request(self, sentinel):
        event = RequestEvent(
            method="POST",
            path="/v1/messages",
            model="claude-3",
            last_message_role="user",
            last_message_preview="Please read the config file",
        )
        await sentinel.observe(event)
        assert len(sentinel._events) == 1
        assert sentinel._events[0]["type"] == "user_request"
        assert sentinel._events[0]["intent"] == "Please read the config file"
        assert sentinel._events[0]["model"] == "claude-3"

    @pytest.mark.anyio
    async def test_ignores_request_without_user_intent(self, sentinel):
        """Requests without user intent (e.g., tool_result messages) are skipped."""
        event = RequestEvent(
            method="POST",
            path="/v1/messages",
            last_message_role="user",
            last_message_preview=None,
        )
        await sentinel.observe(event)
        assert len(sentinel._events) == 0

    @pytest.mark.anyio
    async def test_ignores_non_user_request(self, sentinel):
        """Requests where last message is not from user are skipped."""
        event = RequestEvent(
            method="POST",
            path="/v1/messages",
            last_message_role="assistant",
            last_message_preview="I will do that",
        )
        await sentinel.observe(event)
        assert len(sentinel._events) == 0

    @pytest.mark.anyio
    async def test_observes_text_events(self, sentinel):
        from cross.events import TextEvent

        event = TextEvent(text="Hello world, I'll help you with that.")
        await sentinel.observe(event)
        assert len(sentinel._events) == 1
        assert sentinel._events[0]["type"] == "agent_text"
        assert sentinel._events[0]["text"] == "Hello world, I'll help you with that."

    @pytest.mark.anyio
    async def test_skips_empty_text_events(self, sentinel):
        from cross.events import TextEvent

        await sentinel.observe(TextEvent(text=""))
        await sentinel.observe(TextEvent(text="   "))
        assert len(sentinel._events) == 0

    @pytest.mark.anyio
    async def test_truncates_long_text_events(self, sentinel):
        from cross.events import TextEvent

        long_text = "x" * 500
        await sentinel.observe(TextEvent(text=long_text))
        assert len(sentinel._events) == 1
        assert len(sentinel._events[0]["text"]) == 300


# --- LLMSentinel._do_review ---


class TestSentinelReview:
    def _make_sentinel(self) -> tuple[LLMSentinel, EventBus]:
        config = LLMConfig(model="anthropic/claude-sonnet-4-6", api_key="sk-test")
        bus = EventBus()
        sentinel = LLMSentinel(config=config, event_bus=bus, interval_seconds=60)
        return sentinel, bus

    @pytest.mark.anyio
    async def test_ok_review(self):
        sentinel, bus = self._make_sentinel()
        events = [{"type": "tool_use", "name": "Read", "input": {}, "ts": time.time()}]

        published = []
        bus.subscribe(lambda e: published.append(e))

        with patch("cross.sentinels.llm_reviewer.complete", new_callable=AsyncMock) as mock:
            mock.return_value = "VERDICT: OK\nSUMMARY: Normal operations.\nCONCERNS: None"
            await sentinel._do_review(events)

        assert len(published) == 1
        assert isinstance(published[0], SentinelReviewEvent)
        assert published[0].action == "allow"
        assert "Normal operations" in published[0].summary

    @pytest.mark.anyio
    async def test_alert_review(self):
        sentinel, bus = self._make_sentinel()
        events = [{"type": "tool_use", "name": "Read", "input": {"file_path": "/etc/shadow"}, "ts": time.time()}]

        published = []
        bus.subscribe(lambda e: published.append(e))

        with patch("cross.sentinels.llm_reviewer.complete", new_callable=AsyncMock) as mock:
            mock.return_value = "VERDICT: ALERT\nSUMMARY: Sensitive file read.\nCONCERNS: Agent read /etc/shadow."
            await sentinel._do_review(events)

        assert published[0].action == "alert"
        assert "/etc/shadow" in published[0].concerns

    @pytest.mark.anyio
    async def test_halt_review(self):
        sentinel, bus = self._make_sentinel()
        events = [{"type": "tool_use", "name": "Bash", "input": {}, "ts": time.time()}]

        published = []
        bus.subscribe(lambda e: published.append(e))

        with patch("cross.sentinels.llm_reviewer.complete", new_callable=AsyncMock) as mock:
            mock.return_value = "VERDICT: HALT\nSUMMARY: Data exfil.\nCONCERNS: Agent exfiltrating credentials."
            with patch("cross.proxy.set_sentinel_halt") as halt_mock:
                await sentinel._do_review(events)

        assert published[0].action == "halt_session"
        halt_mock.assert_called_once()
        assert "exfiltrating credentials" in halt_mock.call_args[0][0]

    @pytest.mark.anyio
    async def test_no_response_does_not_publish(self):
        sentinel, bus = self._make_sentinel()
        events = [{"type": "tool_use", "name": "Read", "input": {}, "ts": time.time()}]

        published = []
        bus.subscribe(lambda e: published.append(e))

        with patch("cross.sentinels.llm_reviewer.complete", new_callable=AsyncMock) as mock:
            mock.return_value = None
            await sentinel._do_review(events)

        assert len(published) == 0

    @pytest.mark.anyio
    async def test_unparseable_response_does_not_publish(self):
        sentinel, bus = self._make_sentinel()
        events = [{"type": "tool_use", "name": "Read", "input": {}, "ts": time.time()}]

        published = []
        bus.subscribe(lambda e: published.append(e))

        with patch("cross.sentinels.llm_reviewer.complete", new_callable=AsyncMock) as mock:
            mock.return_value = "I don't know what to say about this."
            await sentinel._do_review(events)

        assert len(published) == 0


# --- LLMSentinel start/stop ---


class TestSentinelLifecycle:
    @pytest.mark.anyio
    async def test_start_stop(self):
        config = LLMConfig(model="anthropic/claude-sonnet-4-6", api_key="sk-test")
        bus = EventBus()
        sentinel = LLMSentinel(config=config, event_bus=bus, interval_seconds=60)

        sentinel.start()
        assert sentinel._running is True
        assert sentinel._task is not None

        sentinel.stop()
        assert sentinel._running is False

    @pytest.mark.anyio
    async def test_double_start_idempotent(self):
        config = LLMConfig(model="anthropic/claude-sonnet-4-6", api_key="sk-test")
        bus = EventBus()
        sentinel = LLMSentinel(config=config, event_bus=bus, interval_seconds=60)

        sentinel.start()
        task1 = sentinel._task
        sentinel.start()
        assert sentinel._task is task1  # same task

        sentinel.stop()

    @pytest.mark.anyio
    async def test_review_loop_skips_empty_window(self):
        config = LLMConfig(model="anthropic/claude-sonnet-4-6", api_key="sk-test")
        bus = EventBus()
        sentinel = LLMSentinel(config=config, event_bus=bus, interval_seconds=0.1)

        with patch("cross.sentinels.llm_reviewer.complete", new_callable=AsyncMock) as mock:
            mock.return_value = "VERDICT: OK\nSUMMARY: Fine.\nCONCERNS: None"
            sentinel.start()
            await asyncio.sleep(0.3)
            sentinel.stop()

        # No events were added, so complete should not have been called
        mock.assert_not_called()

    @pytest.mark.anyio
    async def test_review_loop_processes_events(self):
        config = LLMConfig(model="anthropic/claude-sonnet-4-6", api_key="sk-test")
        bus = EventBus()
        sentinel = LLMSentinel(config=config, event_bus=bus, interval_seconds=0.1)

        published = []
        bus.subscribe(lambda e: published.append(e))

        # Add an event before starting
        await sentinel.observe(ToolUseEvent(name="Bash", tool_use_id="tu_1", input={"command": "ls"}))

        with patch("cross.sentinels.llm_reviewer.complete", new_callable=AsyncMock) as mock:
            mock.return_value = "VERDICT: OK\nSUMMARY: Normal.\nCONCERNS: None"
            sentinel.start()
            await asyncio.sleep(0.3)
            sentinel.stop()

        # complete should have been called at least once
        assert mock.call_count >= 1
        # Should have published a SentinelReviewEvent
        sentinel_events = [e for e in published if isinstance(e, SentinelReviewEvent)]
        assert len(sentinel_events) >= 1

    @pytest.mark.anyio
    async def test_failed_review_retries_events(self):
        """If review fails, events should be retried in the next cycle."""
        config = LLMConfig(model="anthropic/claude-sonnet-4-6", api_key="sk-test")
        bus = EventBus()
        sentinel = LLMSentinel(config=config, event_bus=bus, interval_seconds=0.1)

        await sentinel.observe(ToolUseEvent(name="Bash", tool_use_id="tu_1", input={}))

        call_count = 0

        async def failing_then_succeeding(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("LLM down")
            return "VERDICT: OK\nSUMMARY: Fine.\nCONCERNS: None"

        with patch("cross.sentinels.llm_reviewer.complete", new_callable=AsyncMock) as mock:
            mock.side_effect = failing_then_succeeding
            sentinel.start()
            await asyncio.sleep(0.35)
            sentinel.stop()

        # Should have been called at least twice (first fails, second succeeds with same events)
        assert call_count >= 2

    @pytest.mark.anyio
    async def test_review_loop_only_reviews_new_events(self):
        config = LLMConfig(model="anthropic/claude-sonnet-4-6", api_key="sk-test")
        bus = EventBus()
        sentinel = LLMSentinel(config=config, event_bus=bus, interval_seconds=0.1)

        with patch("cross.sentinels.llm_reviewer.complete", new_callable=AsyncMock) as mock:
            mock.return_value = "VERDICT: OK\nSUMMARY: Fine.\nCONCERNS: None"

            # Add event and start
            await sentinel.observe(ToolUseEvent(name="Bash", tool_use_id="tu_1", input={}))
            sentinel.start()
            await asyncio.sleep(0.25)

            # First review should have happened
            first_call_count = mock.call_count
            assert first_call_count >= 1

            # Wait another interval without new events
            await asyncio.sleep(0.25)
            sentinel.stop()

            # Should not have reviewed again (no new events)
            assert mock.call_count == first_call_count
