"""JSONL structured logger plugin — writes events to data/cross.log."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from cross.config import settings
from cross.events import (
    CrossEvent,
    ErrorEvent,
    MessageDeltaEvent,
    MessageStartEvent,
    RequestEvent,
    TextEvent,
    ToolUseEvent,
)

logger = logging.getLogger("cross.plugins.logger")


class LoggerPlugin:
    def __init__(self):
        log_path = Path(settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "a")
        logger.info(f"Logging events to {log_path}")

    def _write(self, record: dict):
        record["ts"] = datetime.now(timezone.utc).isoformat()
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    async def handle(self, event: CrossEvent):
        match event:
            case RequestEvent():
                self._write(
                    {
                        "type": "request",
                        "method": event.method,
                        "path": event.path,
                        "model": event.model,
                        "messages_count": event.messages_count,
                        "stream": event.stream,
                        "tools": event.tool_names[:15],
                        "tools_count": len(event.tool_names),
                        "last_message_role": event.last_message_role,
                        "last_message_preview": event.last_message_preview,
                    }
                )
                # Also log to console for visibility
                tools_str = f" tools={len(event.tool_names)}" if event.tool_names else ""
                logger.info(
                    f"REQUEST {event.method} {event.path} "
                    f"model={event.model} msgs={event.messages_count} "
                    f"stream={event.stream}{tools_str}"
                )

            case MessageStartEvent():
                self._write(
                    {
                        "type": "message_start",
                        "message_id": event.message_id,
                        "model": event.model,
                    }
                )
                logger.info(f"  message_start id={event.message_id} model={event.model}")

            case ToolUseEvent():
                self._write(
                    {
                        "type": "tool_use",
                        "name": event.name,
                        "tool_use_id": event.tool_use_id,
                        "input": event.input,
                    }
                )
                # Log tool input preview
                input_str = json.dumps(event.input)
                if len(input_str) > 200:
                    input_str = input_str[:200] + "..."
                logger.info(f"  tool_use: {event.name} input={input_str}")

            case TextEvent():
                preview = event.text[:200] + "..." if len(event.text) > 200 else event.text
                self._write(
                    {
                        "type": "text",
                        "text": event.text,
                    }
                )
                logger.info(f"  text: {preview}")

            case MessageDeltaEvent():
                self._write(
                    {
                        "type": "message_delta",
                        "stop_reason": event.stop_reason,
                        "output_tokens": event.output_tokens,
                    }
                )
                logger.info(f"  message_delta: stop_reason={event.stop_reason} output_tokens={event.output_tokens}")

            case ErrorEvent():
                self._write(
                    {
                        "type": "error",
                        "status_code": event.status_code,
                        "body": event.body,
                    }
                )
                logger.warning(f"  ERROR {event.status_code}: {event.body[:200]}")
