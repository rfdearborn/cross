"""Conversation store — inline follow-up conversations with gate/sentinel reviewers.

Maintains per-evaluation conversation state so users can ask the LLM reviewer
follow-up questions about a specific gate decision or sentinel review, grounded
in the original evaluation context.

Conversation IDs: "gate:{tool_use_id}" or "sentinel:{review_id}".
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from cross.custom_instructions import format_instructions_block
from cross.llm import LLMConfig, complete

logger = logging.getLogger("cross.conversations")

MAX_MESSAGES = 20  # per conversation
MAX_CONVERSATIONS = 100  # total active


@dataclass
class ConversationMessage:
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


# --- System prompts ---

_GATE_SYSTEM_PROMPT = """\
You are a security reviewer for an AI agent monitoring system called Cross. \
You previously evaluated a tool call and made a decision. The user is now asking \
you follow-up questions about that evaluation.

Here is the context of what happened:

Tool: {tool_name}
Input:
{tool_input}
{script_section}\
Your decision: {action}
Your reasoning: {reason}
{rule_section}\
{recent_tools_section}\
{intent_section}\
Answer the user's questions about this evaluation. Be specific and reference \
the actual tool call details. If you made an error in your assessment, \
acknowledge it. Keep responses concise (2-3 sentences unless more detail \
is requested)."""

_SENTINEL_SYSTEM_PROMPT = """\
You are a security sentinel for an AI agent monitoring system called Cross. \
You previously reviewed a window of agent activity and made an assessment. \
The user is now asking you follow-up questions about that review.

Your review:
  Verdict: {action}
  Summary: {summary}
  Concerns: {concerns}
  Events reviewed: {event_count}

Activity window:
{event_window}

Answer the user's questions about this review. Reference specific events \
when relevant. Keep responses concise unless more detail is requested."""


def _build_gate_system_prompt(ctx: dict[str, Any], custom_instructions: str) -> str:
    """Build system prompt for a gate conversation."""
    tool_input = ctx.get("tool_input")
    if isinstance(tool_input, dict):
        input_str = json.dumps(tool_input, indent=2)
    else:
        input_str = str(tool_input) if tool_input else "(empty)"

    # Script contents section
    script_section = ""
    script_contents = ctx.get("script_contents")
    if script_contents:
        parts = ["\nScript file contents:"]
        for path, source in script_contents.items():
            if len(source) > 10000:
                source = source[:10000] + "\n... [truncated]"
            parts.append(f"--- {path} ---\n{source}\n--- end {path} ---")
        script_section = "\n".join(parts) + "\n\n"

    # Rule section
    rule_section = ""
    rule_id = ctx.get("rule_id")
    if rule_id:
        rule_section = f"Matched rule: {rule_id}\n"

    # Recent tools section
    recent_tools_section = ""
    recent_tools = ctx.get("recent_tools")
    if recent_tools:
        lines = ["\nRecent tool calls (most recent last):"]
        for tool in recent_tools[-10:]:
            name = tool.get("name", "?")
            t_input = tool.get("input")
            if isinstance(t_input, dict):
                summary = json.dumps(t_input)
                if len(summary) > 150:
                    summary = summary[:150] + "..."
            else:
                summary = str(t_input)[:150] if t_input else "(empty)"
            lines.append(f"  - {name}: {summary}")
        recent_tools_section = "\n".join(lines) + "\n"

    # Intent section
    intent_section = ""
    user_intent = ctx.get("user_intent")
    if user_intent:
        intent_section = f"\nUser intent: {user_intent}\n"

    prompt = _GATE_SYSTEM_PROMPT.format(
        tool_name=ctx.get("tool_name", "unknown"),
        tool_input=input_str,
        script_section=script_section,
        action=ctx.get("action", "unknown"),
        reason=ctx.get("reason", ""),
        rule_section=rule_section,
        recent_tools_section=recent_tools_section,
        intent_section=intent_section,
    )

    if custom_instructions:
        prompt += format_instructions_block(custom_instructions)
    return prompt


def _build_sentinel_system_prompt(ctx: dict[str, Any], custom_instructions: str) -> str:
    """Build system prompt for a sentinel conversation."""
    event_window = ctx.get("event_window_text", "(no event window available)")
    # Cap event window to avoid blowing context
    if len(event_window) > 8000:
        event_window = event_window[:8000] + "\n... [truncated]"

    prompt = _SENTINEL_SYSTEM_PROMPT.format(
        action=ctx.get("action", "unknown"),
        summary=ctx.get("summary", ""),
        concerns=ctx.get("concerns", ""),
        event_count=ctx.get("event_count", 0),
        event_window=event_window,
    )

    if custom_instructions:
        prompt += format_instructions_block(custom_instructions)
    return prompt


class ConversationStore:
    """In-memory conversation state with bounded history and LRU eviction."""

    def __init__(
        self,
        gate_llm_config: LLMConfig | None = None,
        sentinel_llm_config: LLMConfig | None = None,
        get_custom_instructions: Callable[[], str] | None = None,
    ):
        self._gate_config = gate_llm_config
        self._sentinel_config = sentinel_llm_config
        self._get_custom_instructions = get_custom_instructions
        # conversation_id -> context dict
        self._contexts: dict[str, dict[str, Any]] = {}
        # conversation_id -> message list
        self._messages: dict[str, list[ConversationMessage]] = {}
        # Per-conversation locks for serializing LLM calls
        self._locks: dict[str, asyncio.Lock] = {}
        # LRU tracking (most recently used at end)
        self._access_order: list[str] = []

    def register_gate_context(
        self,
        tool_use_id: str,
        tool_name: str,
        tool_input: dict[str, Any] | None,
        action: str,
        reason: str,
        rule_id: str = "",
        evaluator: str = "",
        script_contents: dict[str, str] | None = None,
        recent_tools: list[dict[str, Any]] | None = None,
        user_intent: str = "",
    ) -> str:
        """Register context for a gate conversation. Returns the conversation_id."""
        conv_id = f"gate:{tool_use_id}"
        self._contexts[conv_id] = {
            "type": "gate",
            "tool_use_id": tool_use_id,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "action": action,
            "reason": reason,
            "rule_id": rule_id,
            "evaluator": evaluator,
            "script_contents": script_contents,
            "recent_tools": recent_tools or [],
            "user_intent": user_intent,
        }
        self._touch(conv_id)
        return conv_id

    def register_sentinel_context(
        self,
        review_id: str,
        action: str,
        summary: str,
        concerns: str,
        event_count: int,
        event_window_text: str = "",
    ) -> str:
        """Register context for a sentinel conversation. Returns the conversation_id."""
        conv_id = f"sentinel:{review_id}"
        self._contexts[conv_id] = {
            "type": "sentinel",
            "review_id": review_id,
            "action": action,
            "summary": summary,
            "concerns": concerns,
            "event_count": event_count,
            "event_window_text": event_window_text,
        }
        self._touch(conv_id)
        return conv_id

    def get_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        """Return message history for a conversation."""
        msgs = self._messages.get(conversation_id, [])
        return [{"role": m.role, "content": m.content, "timestamp": m.timestamp} for m in msgs]

    def has_context(self, conversation_id: str) -> bool:
        """Check if a conversation context is registered."""
        return conversation_id in self._contexts

    async def send_message(self, conversation_id: str, user_message: str) -> str | None:
        """Send a user message and get LLM response. Returns reply text or None."""
        ctx = self._contexts.get(conversation_id)
        if not ctx:
            logger.warning(f"No context for conversation {conversation_id}")
            return None

        conv_type = ctx.get("type", "")
        if conv_type == "gate":
            config = self._gate_config
        elif conv_type == "sentinel":
            config = self._sentinel_config
        else:
            logger.warning(f"Unknown conversation type: {conv_type}")
            return None

        if not config or not config.model:
            logger.warning(f"No LLM config for {conv_type} conversations")
            return None

        # Serialize per conversation
        lock = self._locks.setdefault(conversation_id, asyncio.Lock())
        async with lock:
            # Add user message
            msgs = self._messages.setdefault(conversation_id, [])
            msgs.append(ConversationMessage(role="user", content=user_message))

            # Trim to max
            if len(msgs) > MAX_MESSAGES:
                msgs[:] = msgs[-MAX_MESSAGES:]

            # Build system prompt
            custom = self._get_custom_instructions() if self._get_custom_instructions else ""
            if conv_type == "gate":
                system = _build_gate_system_prompt(ctx, custom)
            else:
                system = _build_sentinel_system_prompt(ctx, custom)

            # Build messages for LLM
            llm_messages = [{"role": m.role, "content": m.content} for m in msgs]

            reply = await complete(config, system=system, messages=llm_messages, timeout_s=60.0)

            if reply:
                msgs.append(ConversationMessage(role="assistant", content=reply))
                if len(msgs) > MAX_MESSAGES:
                    msgs[:] = msgs[-MAX_MESSAGES:]
                self._touch(conversation_id)
            else:
                # Remove the user message if LLM failed
                msgs.pop()
                logger.warning(f"LLM returned no response for conversation {conversation_id}")

            return reply

    def _touch(self, conversation_id: str) -> None:
        """Update LRU order and evict if over limit."""
        if conversation_id in self._access_order:
            self._access_order.remove(conversation_id)
        self._access_order.append(conversation_id)

        # Evict oldest conversations if over limit
        while len(self._access_order) > MAX_CONVERSATIONS:
            oldest = self._access_order.pop(0)
            self._contexts.pop(oldest, None)
            self._messages.pop(oldest, None)
            self._locks.pop(oldest, None)
