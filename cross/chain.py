"""Evaluator chain — runs gate evaluators and aggregates results.

Two-stage evaluation:
  Stage 1: Run all gates (denylist, etc.), max action wins.
  Stage 2: If result >= REVIEW AND review gate exists, run LLM review.
           For REVIEW actions, the LLM decides the outcome (can override).
           For higher actions (ESCALATE/BLOCK/HALT_SESSION), LLM analysis
           is advisory — attached as metadata for human decision-makers.
           On ABSTAIN/error, stage 1 result stands.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time

from cross.config import settings
from cross.evaluator import Action, EvaluationResponse, Gate, GateRequest

logger = logging.getLogger("cross.chain")


class GateChain:
    """Runs gate evaluators against a tool call, with optional LLM review stage."""

    def __init__(
        self,
        gates: list[Gate] | None = None,
        review_gate: Gate | None = None,
    ):
        self.gates: list[Gate] = gates or []
        self.review_gate: Gate | None = review_gate

    def add(self, gate: Gate):
        self.gates.append(gate)

    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        """Run all gates, optionally escalate to LLM review."""
        # Stage 1: run all gates
        stage1_result = await self._run_gates(request)

        # Stage 2: LLM review for anything at REVIEW or above
        if self.review_gate and stage1_result.action.value >= Action.REVIEW.value:
            review_result = await self._run_review(request, stage1_result)
            if review_result is not None:
                return review_result
            # ABSTAIN/error: if stage 1 was REVIEW (i.e. "needs LLM judgment"),
            # promote to ESCALATE — the LLM couldn't decide, so a human should.
            # For higher actions (ESCALATE/BLOCK/HALT_SESSION) the stage-1 result
            # already stands at the right severity.
            if stage1_result.action == Action.REVIEW:
                logger.warning("LLM review unavailable for REVIEW-level result, promoting to ESCALATE")
                return EvaluationResponse(
                    action=Action.ESCALATE,
                    reason=stage1_result.reason,
                    rule_id=stage1_result.rule_id,
                    evaluator=stage1_result.evaluator,
                    confidence=stage1_result.confidence,
                    duration_ms=stage1_result.duration_ms,
                    metadata={"promoted_from": "REVIEW", "reason": "LLM review unavailable"},
                )

        return stage1_result

    async def _run_gates(self, request: GateRequest) -> EvaluationResponse:
        """Stage 1: run all gates, max action wins."""
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

        return max_response

    async def _run_review(self, request: GateRequest, stage1_result: EvaluationResponse) -> EvaluationResponse | None:
        """Stage 2: run LLM review gate. Returns None if review abstains/errors."""
        assert self.review_gate is not None

        # Copy request to avoid mutating the caller's object
        request = copy.copy(request)
        request.prior_result = stage1_result

        start = time.monotonic()
        try:
            timeout_s = self.review_gate.timeout_ms / 1000.0
            resp = await asyncio.wait_for(self.review_gate.evaluate(request), timeout=timeout_s)
        except asyncio.TimeoutError:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.warning(f"Review gate timed out after {elapsed_ms:.1f}ms, keeping stage-1 result")
            return None
        except Exception as e:
            logger.warning(f"Review gate error: {e}, keeping stage-1 result")
            return None

        elapsed_ms = (time.monotonic() - start) * 1000
        resp.duration_ms = elapsed_ms

        if resp.action == Action.ABSTAIN:
            logger.info("Review gate abstained, keeping stage-1 result")
            return None

        # Shadow mode: LLM decides, but escalate to human with LLM's reasoning
        if settings.llm_gate_shadow:
            shadow_reason = f"[Shadow] LLM gate would {resp.action.name}: {resp.reason}"
            logger.info(f"Shadow mode — escalating to human. {shadow_reason[:150]}")
            return EvaluationResponse(
                action=Action.ESCALATE,
                reason=shadow_reason,
                evaluator=f"{resp.evaluator}:shadow",
                confidence=resp.confidence,
                duration_ms=resp.duration_ms,
                metadata={"shadow_verdict": resp.action.name, "shadow_reason": resp.reason},
            )

        # For REVIEW: LLM has decision power — its verdict overrides stage 1
        if stage1_result.action == Action.REVIEW:
            # Guard: review gate must not return REVIEW itself (would leak through)
            if resp.action == Action.REVIEW:
                logger.warning("Review gate returned REVIEW — promoting to ESCALATE")
                resp = EvaluationResponse(
                    action=Action.ESCALATE,
                    reason=resp.reason,
                    evaluator=resp.evaluator,
                    confidence=resp.confidence,
                    duration_ms=resp.duration_ms,
                    metadata={"promoted_from": "REVIEW", "reason": "review gate returned REVIEW"},
                )
            logger.info(
                f"Review gate overrides stage-1 ({stage1_result.action.name} → {resp.action.name}): {resp.reason[:100]}"
            )
            return resp

        # For higher actions (ESCALATE/BLOCK/HALT_SESSION): LLM analysis is advisory.
        # Original action stands, but LLM reasoning is attached as metadata for humans.
        logger.info(
            f"Review gate advisory for {stage1_result.action.name} (LLM says {resp.action.name}): {resp.reason[:100]}"
        )
        return EvaluationResponse(
            action=stage1_result.action,
            reason=stage1_result.reason,
            rule_id=stage1_result.rule_id,
            evaluator=stage1_result.evaluator,
            confidence=stage1_result.confidence,
            duration_ms=stage1_result.duration_ms + resp.duration_ms,
            metadata={
                "llm_review_action": resp.action.name,
                "llm_review_reason": resp.reason,
                "llm_review_evaluator": resp.evaluator,
                "llm_review_confidence": resp.confidence,
            },
        )
