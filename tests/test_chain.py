"""Tests for the gate chain."""

import asyncio

import pytest

from cross.chain import GateChain
from cross.evaluator import Action, EvaluationResponse, Gate, GateRequest


class AllowGate(Gate):
    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        return EvaluationResponse(action=Action.ALLOW, evaluator=self.name)


class BlockGate(Gate):
    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        return EvaluationResponse(
            action=Action.BLOCK,
            reason="blocked",
            evaluator=self.name,
        )


class AlertGate(Gate):
    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        return EvaluationResponse(
            action=Action.ALERT,
            reason="alert",
            evaluator=self.name,
        )


class AbstainGate(Gate):
    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        return EvaluationResponse(action=Action.ABSTAIN, evaluator=self.name)


class ErrorGate(Gate):
    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        raise RuntimeError("gate exploded")


def _req() -> GateRequest:
    return GateRequest(tool_name="Bash", tool_input={"command": "ls"})


class TestChainBasics:
    @pytest.mark.anyio
    async def test_empty_chain_allows(self):
        chain = GateChain()
        r = await chain.evaluate(_req())
        assert r.action == Action.ALLOW

    @pytest.mark.anyio
    async def test_single_allow(self):
        chain = GateChain(gates=[AllowGate(name="a")])
        r = await chain.evaluate(_req())
        assert r.action == Action.ALLOW

    @pytest.mark.anyio
    async def test_single_block(self):
        chain = GateChain(gates=[BlockGate(name="b")])
        r = await chain.evaluate(_req())
        assert r.action == Action.BLOCK

    @pytest.mark.anyio
    async def test_max_action_wins(self):
        chain = GateChain(gates=[AllowGate(name="a"), BlockGate(name="b")])
        r = await chain.evaluate(_req())
        assert r.action == Action.BLOCK
        assert r.evaluator == "b"

    @pytest.mark.anyio
    async def test_block_beats_alert(self):
        chain = GateChain(gates=[AlertGate(name="a"), BlockGate(name="b")])
        r = await chain.evaluate(_req())
        assert r.action == Action.BLOCK

    @pytest.mark.anyio
    async def test_alert_beats_allow(self):
        chain = GateChain(gates=[AllowGate(name="a"), AlertGate(name="b")])
        r = await chain.evaluate(_req())
        assert r.action == Action.ALERT

    @pytest.mark.anyio
    async def test_all_abstain_becomes_allow(self):
        chain = GateChain(gates=[AbstainGate(name="a"), AbstainGate(name="b")])
        r = await chain.evaluate(_req())
        assert r.action == Action.ALLOW
        assert r.evaluator == "chain:all_abstained"


class TestChainErrorHandling:
    @pytest.mark.anyio
    async def test_gate_error_uses_on_error_action(self):
        gate = ErrorGate(name="err")
        gate.on_error = Action.ALLOW
        chain = GateChain(gates=[gate])
        r = await chain.evaluate(_req())
        assert r.action == Action.ALLOW

    @pytest.mark.anyio
    async def test_gate_error_with_block_on_error(self):
        gate = ErrorGate(name="err")
        gate.on_error = Action.BLOCK
        chain = GateChain(gates=[gate])
        r = await chain.evaluate(_req())
        assert r.action == Action.BLOCK

    @pytest.mark.anyio
    async def test_error_doesnt_prevent_other_gates(self):
        chain = GateChain(gates=[ErrorGate(name="err"), BlockGate(name="b")])
        r = await chain.evaluate(_req())
        assert r.action == Action.BLOCK


class TestChainAdd:
    @pytest.mark.anyio
    async def test_add_gate(self):
        chain = GateChain()
        chain.add(BlockGate(name="b"))
        r = await chain.evaluate(_req())
        assert r.action == Action.BLOCK

    @pytest.mark.anyio
    async def test_duration_tracked(self):
        chain = GateChain(gates=[AllowGate(name="a")])
        r = await chain.evaluate(_req())
        assert r.duration_ms >= 0


class HaltSessionGate(Gate):
    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        return EvaluationResponse(
            action=Action.HALT_SESSION,
            reason="halt",
            evaluator=self.name,
        )


class SlowGate(Gate):
    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        await asyncio.sleep(0.5)
        return EvaluationResponse(action=Action.BLOCK, reason="slow", evaluator=self.name)


class TestHaltSessionRemap:
    @pytest.mark.anyio
    async def test_halt_session_remapped_to_block(self):
        chain = GateChain(gates=[HaltSessionGate(name="halt")])
        r = await chain.evaluate(_req())
        assert r.action == Action.BLOCK

    @pytest.mark.anyio
    async def test_halt_session_beats_allow(self):
        chain = GateChain(gates=[AllowGate(name="a"), HaltSessionGate(name="halt")])
        r = await chain.evaluate(_req())
        assert r.action == Action.BLOCK


class TestGateTimeout:
    @pytest.mark.anyio
    async def test_timeout_uses_on_error(self):
        gate = SlowGate(name="slow")
        gate.timeout_ms = 10.0  # 10ms, gate sleeps 500ms
        gate.on_error = Action.BLOCK
        chain = GateChain(gates=[gate])
        r = await chain.evaluate(_req())
        assert r.action == Action.BLOCK
        assert "timed out" in r.reason

    @pytest.mark.anyio
    async def test_timeout_default_allows(self):
        gate = SlowGate(name="slow")
        gate.timeout_ms = 10.0
        gate.on_error = Action.ALLOW  # default
        chain = GateChain(gates=[gate])
        r = await chain.evaluate(_req())
        assert r.action == Action.ALLOW

    @pytest.mark.anyio
    async def test_no_timeout_when_fast_enough(self):
        gate = SlowGate(name="slow")
        gate.timeout_ms = 5000.0  # 5 seconds, gate sleeps 500ms
        chain = GateChain(gates=[gate])
        r = await chain.evaluate(_req())
        assert r.action == Action.BLOCK  # actual gate result, not timeout
