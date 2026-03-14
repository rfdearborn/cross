"""Tests for two-stage GateChain — denylist triage + LLM review."""

from __future__ import annotations

import asyncio
import unittest.mock

import pytest

from cross.chain import GateChain
from cross.config import settings
from cross.evaluator import Action, EvaluationResponse, Gate, GateRequest

# --- Test helpers ---


class StubGate(Gate):
    """Gate that returns a fixed response."""

    def __init__(self, action: Action, name: str = "stub", reason: str = "stub reason", **kwargs):
        super().__init__(name=name)
        self._action = action
        self._reason = reason
        for k, v in kwargs.items():
            setattr(self, k, v)

    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        return EvaluationResponse(action=self._action, reason=self._reason, evaluator=self.name)


class RecordingGate(Gate):
    """Gate that records calls and returns a fixed response."""

    def __init__(self, action: Action, name: str = "recording"):
        super().__init__(name=name)
        self._action = action
        self.calls: list[GateRequest] = []

    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        self.calls.append(request)
        return EvaluationResponse(action=self._action, evaluator=self.name)


class SlowGate(Gate):
    """Gate that takes a long time."""

    def __init__(self, delay_s: float = 5.0, name: str = "slow"):
        super().__init__(name=name)
        self.timeout_ms = 100  # short timeout
        self.delay_s = delay_s

    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        await asyncio.sleep(self.delay_s)
        return EvaluationResponse(action=Action.ALLOW, evaluator=self.name)


class ErrorGate(Gate):
    """Gate that raises an exception."""

    def __init__(self, name: str = "error"):
        super().__init__(name=name)

    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        raise RuntimeError("Gate exploded")


@pytest.fixture(autouse=True)
def _no_shadow(monkeypatch):
    """Ensure shadow mode is off so review gate results are used directly."""
    monkeypatch.setattr(settings, "llm_gate_shadow", False)


# --- Two-stage chain tests ---


class TestTwoStageChain:
    @pytest.mark.anyio
    async def test_no_review_when_under_threshold(self):
        """If stage 1 returns ALLOW, review gate is NOT invoked."""
        review = RecordingGate(Action.ALLOW, name="review")
        chain = GateChain(
            gates=[StubGate(Action.ALLOW, name="denylist")],
            review_gate=review,
            review_threshold=Action.BLOCK,
        )
        result = await chain.evaluate(GateRequest(tool_name="Bash"))
        assert result.action == Action.ALLOW
        assert len(review.calls) == 0  # review not called

    @pytest.mark.anyio
    async def test_review_invoked_when_at_threshold(self):
        """If stage 1 returns BLOCK (= threshold), review gate IS invoked."""
        review = RecordingGate(Action.ALLOW, name="review")
        chain = GateChain(
            gates=[StubGate(Action.BLOCK, name="denylist")],
            review_gate=review,
            review_threshold=Action.BLOCK,
        )
        result = await chain.evaluate(GateRequest(tool_name="Bash"))
        assert result.action == Action.ALLOW  # review overrode to ALLOW
        assert len(review.calls) == 1

    @pytest.mark.anyio
    async def test_review_invoked_when_above_threshold(self):
        """If stage 1 returns HALT_SESSION (> BLOCK threshold), review gate IS invoked."""
        review = RecordingGate(Action.BLOCK, name="review")
        chain = GateChain(
            gates=[StubGate(Action.HALT_SESSION, name="denylist")],
            review_gate=review,
            review_threshold=Action.BLOCK,
        )
        result = await chain.evaluate(GateRequest(tool_name="Bash"))
        assert result.action == Action.BLOCK
        assert len(review.calls) == 1

    @pytest.mark.anyio
    async def test_review_overrides_to_allow(self):
        """Review gate can wave through a false positive (BLOCK → ALLOW)."""
        chain = GateChain(
            gates=[StubGate(Action.BLOCK, name="denylist", reason="matched pattern")],
            review_gate=StubGate(Action.ALLOW, name="review", reason="false positive"),
            review_threshold=Action.BLOCK,
        )
        result = await chain.evaluate(GateRequest(tool_name="Bash"))
        assert result.action == Action.ALLOW
        assert result.evaluator == "review"

    @pytest.mark.anyio
    async def test_review_confirms_block(self):
        """Review gate can confirm the block."""
        chain = GateChain(
            gates=[StubGate(Action.BLOCK, name="denylist")],
            review_gate=StubGate(Action.BLOCK, name="review", reason="confirmed dangerous"),
            review_threshold=Action.BLOCK,
        )
        result = await chain.evaluate(GateRequest(tool_name="Bash"))
        assert result.action == Action.BLOCK
        assert result.evaluator == "review"

    @pytest.mark.anyio
    async def test_review_escalates(self):
        """Review gate can escalate to human."""
        chain = GateChain(
            gates=[StubGate(Action.BLOCK, name="denylist")],
            review_gate=StubGate(Action.ESCALATE, name="review", reason="needs human check"),
            review_threshold=Action.BLOCK,
        )
        result = await chain.evaluate(GateRequest(tool_name="Bash"))
        assert result.action == Action.ESCALATE

    @pytest.mark.anyio
    async def test_review_abstain_keeps_stage1(self):
        """If review abstains, stage 1 result stands."""
        chain = GateChain(
            gates=[StubGate(Action.BLOCK, name="denylist", reason="original reason")],
            review_gate=StubGate(Action.ABSTAIN, name="review"),
            review_threshold=Action.BLOCK,
        )
        result = await chain.evaluate(GateRequest(tool_name="Bash"))
        assert result.action == Action.BLOCK
        assert result.reason == "original reason"

    @pytest.mark.anyio
    async def test_review_receives_prior_result(self):
        """Review gate receives the stage 1 result via request.prior_result."""
        review = RecordingGate(Action.ALLOW, name="review")
        chain = GateChain(
            gates=[StubGate(Action.BLOCK, name="denylist", reason="matched rm -rf")],
            review_gate=review,
            review_threshold=Action.BLOCK,
        )
        await chain.evaluate(GateRequest(tool_name="Bash"))
        assert len(review.calls) == 1
        assert review.calls[0].prior_result is not None
        assert review.calls[0].prior_result.action == Action.BLOCK
        assert review.calls[0].prior_result.reason == "matched rm -rf"

    @pytest.mark.anyio
    async def test_request_not_mutated(self):
        """Chain must not mutate the caller's GateRequest object."""
        review = RecordingGate(Action.ALLOW, name="review")
        chain = GateChain(
            gates=[StubGate(Action.BLOCK, name="denylist")],
            review_gate=review,
            review_threshold=Action.BLOCK,
        )
        original_request = GateRequest(tool_name="Bash")
        await chain.evaluate(original_request)
        assert original_request.prior_result is None  # must not be mutated

    @pytest.mark.anyio
    async def test_no_review_gate_backwards_compat(self):
        """Without review gate, chain behaves like before."""
        chain = GateChain(gates=[StubGate(Action.BLOCK, name="denylist")])
        result = await chain.evaluate(GateRequest(tool_name="Bash"))
        assert result.action == Action.BLOCK

    @pytest.mark.anyio
    async def test_review_timeout_keeps_stage1(self):
        """If review gate times out, stage 1 result stands."""
        chain = GateChain(
            gates=[StubGate(Action.BLOCK, name="denylist", reason="original")],
            review_gate=SlowGate(delay_s=5.0, name="slow_review"),
            review_threshold=Action.BLOCK,
        )
        result = await chain.evaluate(GateRequest(tool_name="Bash"))
        assert result.action == Action.BLOCK
        assert result.reason == "original"

    @pytest.mark.anyio
    async def test_review_error_keeps_stage1(self):
        """If review gate errors, stage 1 result stands."""
        chain = GateChain(
            gates=[StubGate(Action.BLOCK, name="denylist", reason="original")],
            review_gate=ErrorGate(name="broken_review"),
            review_threshold=Action.BLOCK,
        )
        result = await chain.evaluate(GateRequest(tool_name="Bash"))
        assert result.action == Action.BLOCK
        assert result.reason == "original"

    @pytest.mark.anyio
    async def test_alert_threshold(self):
        """With threshold=ALERT, alerts also get reviewed."""
        review = RecordingGate(Action.ALLOW, name="review")
        chain = GateChain(
            gates=[StubGate(Action.ALERT, name="denylist")],
            review_gate=review,
            review_threshold=Action.ALERT,
        )
        result = await chain.evaluate(GateRequest(tool_name="Bash"))
        assert result.action == Action.ALLOW
        assert len(review.calls) == 1

    @pytest.mark.anyio
    async def test_alert_below_block_threshold(self):
        """With threshold=BLOCK, alerts do NOT get reviewed."""
        review = RecordingGate(Action.ALLOW, name="review")
        chain = GateChain(
            gates=[StubGate(Action.ALERT, name="denylist")],
            review_gate=review,
            review_threshold=Action.BLOCK,
        )
        result = await chain.evaluate(GateRequest(tool_name="Bash"))
        assert result.action == Action.ALERT
        assert len(review.calls) == 0


# --- Backward compatibility with existing chain tests ---


class TestChainBackwardCompat:
    """Verify existing chain behavior is preserved with the new two-stage architecture."""

    @pytest.mark.anyio
    async def test_empty_chain_allows(self):
        chain = GateChain()
        result = await chain.evaluate(GateRequest(tool_name="Bash"))
        assert result.action == Action.ALLOW

    @pytest.mark.anyio
    async def test_single_gate_allow(self):
        chain = GateChain(gates=[StubGate(Action.ALLOW)])
        result = await chain.evaluate(GateRequest(tool_name="Bash"))
        assert result.action == Action.ALLOW

    @pytest.mark.anyio
    async def test_max_action_wins(self):
        chain = GateChain(
            gates=[
                StubGate(Action.ALLOW, name="g1"),
                StubGate(Action.BLOCK, name="g2"),
                StubGate(Action.ALERT, name="g3"),
            ]
        )
        result = await chain.evaluate(GateRequest(tool_name="Bash"))
        assert result.action == Action.BLOCK

    @pytest.mark.anyio
    async def test_all_abstain_becomes_allow(self):
        chain = GateChain(gates=[StubGate(Action.ABSTAIN)])
        result = await chain.evaluate(GateRequest(tool_name="Bash"))
        assert result.action == Action.ALLOW

    @pytest.mark.anyio
    async def test_halt_session_remapped_to_block(self):
        chain = GateChain(gates=[StubGate(Action.HALT_SESSION)])
        result = await chain.evaluate(GateRequest(tool_name="Bash"))
        assert result.action == Action.BLOCK

    @pytest.mark.anyio
    async def test_gate_timeout(self):
        chain = GateChain(gates=[SlowGate()])
        result = await chain.evaluate(GateRequest(tool_name="Bash"))
        assert result.action == Action.ALLOW  # on_error default

    @pytest.mark.anyio
    async def test_gate_error(self):
        chain = GateChain(gates=[ErrorGate()])
        result = await chain.evaluate(GateRequest(tool_name="Bash"))
        assert result.action == Action.ALLOW  # on_error default

    @pytest.mark.anyio
    async def test_add_gate(self):
        chain = GateChain()
        chain.add(StubGate(Action.BLOCK))
        result = await chain.evaluate(GateRequest(tool_name="Bash"))
        assert result.action == Action.BLOCK


class TestShadowMode:
    """Shadow mode: LLM decides but all verdicts are escalated to human."""

    @pytest.mark.anyio
    async def test_shadow_escalates_llm_allow(self):
        """LLM says ALLOW → shadow mode escalates to human with LLM's reasoning."""
        chain = GateChain(
            gates=[StubGate(Action.BLOCK, name="denylist", reason="flagged")],
            review_gate=StubGate(Action.ALLOW, name="llm", reason="False positive, safe to run"),
        )
        with unittest.mock.patch.object(settings, "llm_gate_shadow", True):
            result = await chain.evaluate(GateRequest(tool_name="Bash"))

        assert result.action == Action.ESCALATE
        assert "Shadow" in result.reason
        assert "ALLOW" in result.reason
        assert "False positive" in result.reason
        assert result.metadata["shadow_verdict"] == "ALLOW"

    @pytest.mark.anyio
    async def test_shadow_escalates_llm_block(self):
        """LLM says BLOCK → shadow mode escalates to human with LLM's reasoning."""
        chain = GateChain(
            gates=[StubGate(Action.BLOCK, name="denylist", reason="flagged")],
            review_gate=StubGate(Action.BLOCK, name="llm", reason="Confirmed dangerous"),
        )
        with unittest.mock.patch.object(settings, "llm_gate_shadow", True):
            result = await chain.evaluate(GateRequest(tool_name="Bash"))

        assert result.action == Action.ESCALATE
        assert "BLOCK" in result.reason
        assert "Confirmed dangerous" in result.reason
        assert result.metadata["shadow_verdict"] == "BLOCK"

    @pytest.mark.anyio
    async def test_shadow_escalates_llm_escalate(self):
        """LLM says ESCALATE → shadow mode still wraps as shadow escalation."""
        chain = GateChain(
            gates=[StubGate(Action.BLOCK, name="denylist")],
            review_gate=StubGate(Action.ESCALATE, name="llm", reason="Needs human review"),
        )
        with unittest.mock.patch.object(settings, "llm_gate_shadow", True):
            result = await chain.evaluate(GateRequest(tool_name="Bash"))

        assert result.action == Action.ESCALATE
        assert "Shadow" in result.reason
        assert result.evaluator == "llm:shadow"

    @pytest.mark.anyio
    async def test_shadow_off_uses_llm_verdict_directly(self):
        """With shadow off, LLM verdict applies directly (normal behavior)."""
        chain = GateChain(
            gates=[StubGate(Action.BLOCK, name="denylist", reason="flagged")],
            review_gate=StubGate(Action.ALLOW, name="llm", reason="False positive"),
        )
        with unittest.mock.patch.object(settings, "llm_gate_shadow", False):
            result = await chain.evaluate(GateRequest(tool_name="Bash"))

        assert result.action == Action.ALLOW

    @pytest.mark.anyio
    async def test_shadow_does_not_affect_abstain(self):
        """If LLM abstains, shadow mode doesn't apply — stage 1 result stands."""
        chain = GateChain(
            gates=[StubGate(Action.BLOCK, name="denylist", reason="flagged")],
            review_gate=StubGate(Action.ABSTAIN, name="llm"),
        )
        with unittest.mock.patch.object(settings, "llm_gate_shadow", True):
            result = await chain.evaluate(GateRequest(tool_name="Bash"))

        assert result.action == Action.BLOCK  # stage 1 stands

    @pytest.mark.anyio
    async def test_shadow_does_not_affect_below_threshold(self):
        """Calls that don't trigger LLM review are unaffected by shadow mode."""
        chain = GateChain(
            gates=[StubGate(Action.ALLOW, name="denylist")],
            review_gate=StubGate(Action.ALLOW, name="llm"),
        )
        with unittest.mock.patch.object(settings, "llm_gate_shadow", True):
            result = await chain.evaluate(GateRequest(tool_name="Bash"))

        assert result.action == Action.ALLOW  # below threshold, no review
