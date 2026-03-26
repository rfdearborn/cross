# cross

**Configurable Runtime Oversight and Safety Standard**

cross makes AI agents more trustworthy *and* more capable with lightweight gating and monitoring.

> **Not sure you need this?** Ask your agent to explain why you do.

## Why cross

Your agents are overeager. They're so locked in on your task that they'll `rm -rf` an errant directory or push your credentials public without a second thought.

cross is a minimal-friction harness of deterministic and LLM checking layers to guard against errors and misalignment. It pairs agents with spotters which screen actions and monitor sessions with separate context and fresh eyes. It expands the capability-safety frontier and relieves the pit that's been growing in your stomach since you stopped reviewing tool calls.

**New:** cross supports gate and sentinel screening via Claude Code, and so adds no cost at the margin for Claude subscribers!

## Quick Start

```bash
pip install cross-ai               # or: pip install cross-ai[slack] for Slack integration, cross-ai[email] for email
cross setup                        # interactive setup wizard (starts daemon automatically on macOS)
```

Or install from source:

```bash
git clone https://github.com/rfdearborn/cross && cd cross
pip install -e ".[dev]"
```

Then point your agent's API traffic through cross. For Claude Code, add a shell wrapper:

```bash
# In your .zshrc / .bashrc
claude() { cross wrap -- claude "$@"; }
```

`cross wrap` does two things: wraps the agent in a PTY for I/O control, and routes API traffic through the local proxy (`ANTHROPIC_BASE_URL=http://localhost:2767`) for structured monitoring and gating.

## How It Works

cross evaluates every tool call through a 3-layer pipeline:

```
Tool call arrives
       |
       v
 +-----------+     no match     +----------+
 |  Denylist  | --------------> |  ALLOW   |
 |   Gate     |                 +----------+
 +-----------+
       | match
       v
 +-----------+     false pos    +----------+
 |  LLM Gate  | --------------> |  ALLOW   |
 |  (review)  |                 +----------+
 +-----------+
       | confirmed
       v
  BLOCK / ESCALATE
  (feedback to agent / human review)

       --- meanwhile, in the background ---

 +-----------+
 | Sentinel  |  periodic review of all activity
 +-----------+
```

**Layer 1: Denylist gate** -- Fast deterministic pattern matching against YAML rules. Sub-millisecond, zero cost. Default rules cover destructive commands, dangerous git operations, credential exfiltration, reverse shells, system path writes, process termination, privilege escalation, mutating HTTP requests, container destruction, and more. Intentionally broad -- it's a triage filter, not a final judge.

**Layer 2: LLM gate** -- Only invoked when the denylist flags a call. Reviews the tool call with full context (user intent, recent tool history, why it was flagged) and renders a verdict: ALLOW (false positive), BLOCK (confirmed dangerous), or ESCALATE (needs human review). This is why the denylist can be aggressive without generating noise -- the LLM catches false positives.

**Layer 3: Sentinel** -- Async periodic reviewer that watches the full event stream over time. Detects patterns that per-call evaluation misses: tool calls that don't match user intent, suspicious sequences (read credentials then network call), escalating privilege patterns, and agents working around restrictions. Reports to the dashboard and Slack.

Blocked tool calls are suppressed from the API response stream. The proxy automatically retries with the block reason injected, so the agent self-corrects without user intervention. For critical threats (credential exfiltration, reverse shells), the session is halted entirely until a human intervenes.

## Supported Agents

- **Claude Code** -- validated, full PTY + proxy + tool-level gating
- **Codex** (OpenAI) -- validated, full PTY + proxy + tool-level gating. `cross wrap -- codex` routes API traffic through the proxy for streaming interception and pre-execution blocking of tool calls
- **OpenClaw** -- validated, PTY + tool-level gating via `beforeToolCall` hook. `cross wrap -- openclaw` automatically injects a Node.js hook that gates every tool call through the cross daemon
- **Any CLI agent** -- `cross wrap -- <agent-command>` provides PTY wrapping and API proxy for any CLI agent
- **Any agent using Anthropic APIs** -- set `ANTHROPIC_BASE_URL=http://localhost:2767`

## Daemon Management

```bash
cross start                    # start the daemon (backgrounds automatically)
cross stop                     # stop the running daemon
cross restart                  # stop + start (picks up code/config changes)
cross start --foreground       # run in foreground (for development/debugging)
```

`cross daemon` is an alias for `cross start`.

## Dashboard

cross ships with a built-in web dashboard at `http://localhost:2767`. No dependencies, no setup -- it's always active when the daemon is running.

The dashboard shows:
- **Agents** -- monitoring coverage with status chips: green for monitored agents, grey for detected but unmonitored
- **Permission prompts** -- Claude Code permission prompts detected in the terminal, surfaced with Approve/Allow All/Deny actions. Also relayed to Slack and email
- **Pending approvals** -- escalated tool calls waiting for human review, with Approve/Deny buttons
- **Live event feed** -- real-time stream of user messages, agent responses, tool calls (with details), LLM gate decisions (with reasoning), and sentinel reviews. Persists across daemon restarts. Filterable by event type, agent, and text search.

You can also manage pending escalations from the CLI:

```bash
cross pending                          # list pending escalations
cross pending approve <tool_use_id>    # approve
cross pending deny <tool_use_id>       # deny
```

## Notifications

cross delivers notifications through two layers:

- **Native desktop notifications** (macOS) -- via `terminal-notifier`. Clicking a notification opens the dashboard. Enabled during `cross setup` or by setting `CROSS_NATIVE_NOTIFICATIONS_ENABLED=true`. When the dashboard tab is open, browser notifications take priority to avoid opening duplicate tabs.
- **Browser notifications** -- fired from the dashboard tab when it's open. Click to focus the tab. Enable via the button in the dashboard header.
- **Slack** (optional) -- gate decisions, sentinel reviews, and interactive approval buttons. Configure with `CROSS_SLACK_BOT_TOKEN` and `CROSS_SLACK_APP_TOKEN`. Install the `slack` extra: `pip install cross-ai[slack]`.
- **Email** (optional) -- mirrors agent sessions to email threads with approval workflows. Supports SMTP (outbound) and IMAP (inbound replies). Configure via `cross setup` or `CROSS_EMAIL_*` environment variables.

## Configuration

### LLM Providers

cross uses LLMs for the gate reviewer and sentinel. The default routes through Claude Code (`anthropic/claude-code/*`), using your Pro/Max subscription at no additional API cost. You can also use any other supported provider:

| Provider | Model format | API key env var | Notes |
|----------|-------------|-----------------|-------|
| Claude Code | `anthropic/claude-code/claude-sonnet-4-6` | (none needed) | Default for gate. Uses your Claude subscription via `claude -p` |
| Claude Code | `anthropic/claude-code/claude-opus-4-6` | (none needed) | Default for sentinel |
| Codex | `openai/codex/gpt-5.4` | (none needed) | Uses your Codex/ChatGPT Pro subscription via `codex exec` |
| Google Gemini | `google/gemini-3-flash-preview` | `GOOGLE_API_KEY` | Free tier available |
| Anthropic API | `anthropic/claude-haiku-4-5` | `ANTHROPIC_API_KEY` | Direct API (pay-per-token) |
| OpenAI | `openai/gpt-5.4-mini` | `OPENAI_API_KEY` | Direct API |
| Ollama | `ollama/llama3` | (none needed) | Local models |

Configure via environment variables (all prefixed `CROSS_`):

```bash
# LLM gate (default: Claude Code subscription, Sonnet)
CROSS_LLM_GATE_MODEL=anthropic/claude-code/claude-sonnet-4-6

# LLM sentinel (default: Claude Code subscription, Opus)
CROSS_LLM_SENTINEL_MODEL=anthropic/claude-code/claude-opus-4-6
CROSS_LLM_SENTINEL_INTERVAL_SECONDS=60

# Or use an API provider directly
CROSS_LLM_GATE_MODEL=google/gemini-3-flash-preview
CROSS_LLM_GATE_API_KEY=...          # or set GOOGLE_API_KEY
```

Or use `cross setup` for guided interactive configuration.

### Custom Instructions

You can provide custom instructions that are automatically included in gate and sentinel LLM prompts. Store them in `~/.cross/instructions.md` -- they hot-reload on every access, no daemon restart needed.

Edit instructions from the dashboard (Ctrl/Cmd+S to save) or directly in the file. Use this to tailor gate/sentinel behavior to your project, e.g. "allow database migrations" or "flag any network calls in test files".

### Denylist Rules

Default rules ship with cross and escalate to LLM review. They cover destructive commands, dangerous git operations (force push, reset --hard, push to main), credential exfiltration, reverse shells, system path writes, process termination, privilege escalation (sudo), mutating HTTP requests, Docker destruction, package management, and shell config edits. Customize with YAML files in `~/.cross/rules.d/`:

```yaml
# ~/.cross/rules.d/my-rules.yaml
rules:
  - name: no-docker-push
    tools: [Bash]
    field: command
    action: escalate
    description: Prevent pushing Docker images
    patterns:
      - 'docker\s+push\b'

# Disable a default rule by name
disable:
  - destructive-rm
```

Rules support `patterns` (regex, case-insensitive) and `contains` (substring matching), and can target specific tools and input fields. Actions: `escalate` (LLM review), `block` (immediate block), `alert` (log only), `halt_session` (freeze session).

### Updating

```bash
cross update                   # latest from PyPI
cross update --path            # install from local source (current directory)
cross update --path /some/dir  # install from a specific local path
cross update --head            # install from main branch on GitHub
```

Auto-update is enabled by default -- the daemon checks PyPI every 24 hours and installs newer versions automatically. Disable with `CROSS_AUTO_UPDATE_ENABLED=false`.

### All Settings

Settings can be set via environment variables (`CROSS_` prefix) or `.env` files. cross loads `~/.cross/local.env` (personal overrides, survives `cross setup`), then `~/.cross/.env` (generated by setup), then `.env` in the working directory.

See [`cross/config.py`](cross/config.py) for all available settings and their defaults.

## Architecture

cross uses two complementary interception layers:

**PTY wrapper** (`cross wrap`) -- Wraps any CLI agent in a pseudo-terminal for full I/O control. Enables bidirectional messaging relay (Slack/dashboard to agent), terminal-to-phone handoff, and session management. Agent-agnostic.

**Network proxy** -- Intercepts API traffic via `ANTHROPIC_BASE_URL` redirect. Parses streaming SSE responses, buffers tool_use blocks for gate evaluation, and suppresses blocked calls from the response stream. Provides structured monitoring with zero agent modification.

Both layers are coordinated by the daemon (`cross start`), which runs the proxy, gate chain, sentinel, dashboard, and optional Slack and notification plugins as a single process.

## Development

```bash
git clone https://github.com/rfdearborn/cross
cd cross
pip install -e ".[dev,slack]"
python -m pytest tests/
```

Ruff for linting (`ruff check`).

## License

MIT
