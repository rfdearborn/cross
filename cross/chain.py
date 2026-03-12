"""Evaluator chain — runs gate evaluators and aggregates results.

Semantics: run all gates, max action wins (BLOCK > ESCALATE > ALERT > ALLOW > ABSTAIN).
"""

from __future__ import annotations

import asyncio
import logging
import time

from cross.evaluator import Action, EvaluationResponse, Gate, GateRequest

logger = logging.getLogger("cross.chain")


class GateChain:
    """Runs a list of gate evaluators against a tool call."""

    def __init__(self, gates: list[Gate] | None = None):
        self.gates: list[Gate] = gates or []

    def add(self, gate: Gate):
        self.gates.append(gate)

    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        """Run all gates, return the response with the max action."""
        if not self.gates:
            return EvaluationResponse(action=Action.ALLOW, evaluator="chain:empty")

        responses: list[EvaluationResponse] = []

        for gate in self.gates:
            start = time.monotonic()
            try:
                timeout_s = gate.timeout_ms / 1000.0
                resp = await asyncio.wait_for(gate.evaluate(request), timeout=timeout_s)
            except asyncio.TimeoutError:
                elapsed_ms = (time.monotonic() - start) * 1000
                logger.warning(
                    f"Gate '{gate.name}' timed out after {elapsed_ms:.1f}ms "
                    f"(limit: {gate.timeout_ms}ms), using on_error={gate.on_error.name}"
                )
                resp = EvaluationResponse(
                    action=gate.on_error,
                    reason=f"Gate timed out after {elapsed_ms:.1f}ms",
                    evaluator=gate.name,
                )
            except Exception as e:
                logger.exception(f"Gate '{gate.name}' raised: {e}")
                resp = EvaluationResponse(
                    action=gate.on_error,
                    reason=f"Gate error: {e}",
                    evaluator=gate.name,
                )
            elapsed_ms = (time.monotonic() - start) * 1000
            resp.duration_ms = elapsed_ms

            if not resp.evaluator:
                resp.evaluator = gate.name

            responses.append(resp)

        # Max action wins
        max_response = max(responses, key=lambda r: r.action.value)

        # If everything abstained, treat as allow
        if max_response.action == Action.ABSTAIN:
            return EvaluationResponse(action=Action.ALLOW, evaluator="chain:all_abstained")

        # Gates can't halt sessions — remap to BLOCK
        if max_response.action == Action.HALT_SESSION:
            max_response.action = Action.BLOCK

        return max_response
