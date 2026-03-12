"""LLM review gate — synchronous LLM review of denylist-flagged tool calls.

Only invoked when the denylist flags a call (action >= threshold).
Reviews the tool call with context and can override the denylist verdict:
  - ALLOW: false positive, wave it through
  - BLOCK: confirmed dangerous, block with feedback
  - ESCALATE: uncertain, needs human review

On parse failure or error, returns ABSTAIN (falls back to denylist verdict).
"""

from __future__ import annotations

import json
import logging
import re

from cross.evaluator import Action, EvaluationResponse, Gate, GateRequest
from cross.llm import LLMConfig, complete

logger = logging.getLogger("cross.gates.llm_review")

_SYSTEM_PROMPT = """\
You are a security reviewer for an AI agent monitoring system. Your job is to review \
tool calls that were flagged by a pattern-matching denylist.

The denylist is intentionally broad — it catches many false positives. Your job is to \
determine whether this specific tool call is actually dangerous or just a false positive.

Consider:
- What the tool call actually does in context
- Whether the flagged pattern is a genuine risk or incidental
- The user's stated intent (if available)
- Whether the action could cause irreversible damage

Respond with exactly one of these verdicts on the FIRST line:
  VERDICT: ALLOW — if this is a false positive (safe to proceed)
  VERDICT: BLOCK — if this is genuinely dangerous (prevent execution)
  VERDICT: ESCALATE — if you're unsure and a human should decide

Follow the verdict with a brief explanation (1-2 sentences)."""

_VERDICT_PATTERN = re.compile(r"VERDICT:\s*(ALLOW|BLOCK|ESCALATE)", re.IGNORECASE)


def _format_review_prompt(request: GateRequest) -> str:
    """Build the user message for LLM review."""
    parts = []

    # Tool call details
    parts.append(f"Tool: {request.tool_name}")
    if isinstance(request.tool_input, dict):
        input_str = json.dumps(request.tool_input, indent=2)
    else:
        input_str = str(request.tool_input) if request.tool_input else "(empty)"
    parts.append(f"Input:\n{input_str}")

    # Why it was flagged
    if request.prior_result:
        parts.append(f"\nFlagged by: {request.prior_result.evaluator}")
        parts.append(f"Rule: {request.prior_result.rule_id}")
        parts.append(f"Reason: {request.prior_result.reason}")
        parts.append(f"Denylist action: {request.prior_result.action.name}")

    # Recent tool calls (trailing context)
    if request.recent_tools:
        parts.append("\nRecent tool calls (most recent last):")
        for tool in request.recent_tools:
            name = tool.get("name", "?")
            tool_input = tool.get("input")
            if isinstance(tool_input, dict):
                summary = json.dumps(tool_input)
                if len(summary) > 150:
                    summary = summary[:150] + "..."
            else:
                summary = str(tool_input)[:150] if tool_input else "(empty)"
            parts.append(f"  - {name}: {summary}")

    # Context
    if request.cwd:
        parts.append(f"\nWorking directory: {request.cwd}")
    if request.user_intent:
        parts.append(f"User intent: {request.user_intent}")
    if request.agent:
        parts.append(f"Agent: {request.agent}")

    return "\n".join(parts)


def _parse_verdict(text: str) -> Action | None:
    """Extract verdict from the first few lines of LLM response. Returns None if unparseable.

    Only checks the first 3 lines to avoid picking up VERDICT strings
    from quoted tool input or examples in longer responses.
    """
    # Check only the first few lines — verdict should be near the top
    first_lines = "\n".join(text.strip().splitlines()[:3])
    match = _VERDICT_PATTERN.search(first_lines)
    if not match:
        return None
    verdict = match.group(1).upper()
    return Action[verdict]


class LLMReviewGate(Gate):
    """Synchronous LLM review of denylist-flagged tool calls."""

    def __init__(self, config: LLMConfig, timeout_ms: float = 30000):
        super().__init__(name="llm_review")
        self.config = config
        self.timeout_ms = timeout_ms
        self.on_error = Action.ABSTAIN  # fall back to denylist verdict

    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        """Review a flagged tool call. Returns ALLOW/BLOCK/ESCALATE or ABSTAIN on error."""
        user_message = _format_review_prompt(request)
        messages = [{"role": "user", "content": user_message}]

        timeout_s = self.timeout_ms / 1000.0
        text = await complete(self.config, system=_SYSTEM_PROMPT, messages=messages, timeout_s=timeout_s)

        if text is None:
            logger.warning("LLM review returned no response, abstaining")
            return EvaluationResponse(action=Action.ABSTAIN, evaluator=self.name, reason="LLM returned no response")

        verdict = _parse_verdict(text)
        if verdict is None:
            logger.warning(f"Could not parse LLM verdict from: {text[:200]}")
            return EvaluationResponse(
                action=Action.ABSTAIN,
                evaluator=self.name,
                reason=f"Unparseable LLM response: {text[:100]}",
            )

        # Extract explanation (everything after the VERDICT line)
        explanation = _VERDICT_PATTERN.sub("", text).strip()
        if not explanation:
            explanation = f"LLM verdict: {verdict.name}"

        return EvaluationResponse(
            action=verdict,
            reason=explanation,
            evaluator=self.name,
            confidence=0.9,
        )
