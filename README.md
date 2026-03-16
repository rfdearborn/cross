# cross

**Configurable Runtime Oversight and Safety Standard**

cross makes AI agents more trustworthy *and* more capable with lightweight gating and monitoring.

> **Not sure you need this?** Ask your agent to explain why you do.

## Why cross

Your agents are overeager. They're so locked in on finishing your task that they'll `rm -rf` an errant directory or push your credentials public without a second thought.

cross watches from the outside, like a copilot keeping situational awareness while the pilot flies the plane, to guard against this. It pairs agents with spotters which, with separate context and fresh eyes, check actions and monitor sessions.

cross expands the capability-safety frontier: with a babysitter watching, you can give agents more rope. You can stop pre-screening tool calls without feeling a pit in your stomach. More throughput, fewer surprises.

## Quick Start

```bash
git clone https://github.com/x-attn/cross && cd cross
pip install -e ".[dev]"            # PyPI and Homebrew coming soon
cross setup                        # interactive setup wizard (starts daemon automatically on macOS)
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

**Layer 1: Denylist gate** -- Fast deterministic pattern matching against YAML rules. Sub-millisecond, zero cost. Catches destructive commands (`rm -rf /`), credential exfiltration, reverse shells, system path writes, and more. Intentionally broad -- it's a triage filter, not a final judge.

**Layer 2: LLM gate** -- Only invoked when the denylist flags a call. Reviews the tool call with full context (user intent, recent tool history, why it was flagged) and renders a verdict: ALLOW (false positive), BLOCK (confirmed dangerous), or ESCALATE (needs human review). This is why the denylist can be aggressive without generating noise -- the LLM catches false positives.

**Layer 3: Sentinel** -- Async periodic reviewer that watches the full event stream over time. Detects patterns that per-call evaluation misses: tool calls that don't match user intent, suspicious sequences (read credentials then network call), escalating privilege patterns, and agents working around restrictions. Reports to the dashboard and Slack.

Blocked tool calls are suppressed from the API response stream. The proxy automatically retries with the block reason injected, so the agent self-corrects without user intervention. For critical threats (credential exfiltration, reverse shells), the session is halted entirely until a human intervenes.

## Supported Agents

- **Claude Code** -- validated, full PTY + proxy + tool-level gating
- **OpenClaw** -- validated, PTY + tool-level gating via `beforeToolCall` hook. `cross wrap -- openclaw` automatically injects a Node.js hook that gates every tool call through the cross daemon
- **Any CLI agent** -- `cross wrap -- <agent-command>` provides PTY wrapping and API proxy for any CLI agent
- **Any agent using Anthropic APIs** -- set `ANTHROPIC_BASE_URL=http://localhost:2767`

## Dashboard

cross ships with a built-in web dashboard at `http://localhost:2767`. No dependencies, no setup -- it's always active when the daemon is running.

The dashboard shows:
- **Pending approvals** -- escalated tool calls waiting for human review, with Approve/Deny buttons
- **Live event feed** -- real-time stream of tool calls, gate decisions, and sentinel reviews

You can also manage pending escalations from the CLI:

```bash
cross pending                          # list pending escalations
cross pending approve <tool_use_id>    # approve
cross pending deny <tool_use_id>       # deny
```

## Configuration

### LLM Providers

cross uses LLMs for the gate reviewer and sentinel. The default is `claude` (`cli/claude`), which uses your existing Claude Code subscription -- no API key needed. You can also use any other supported provider:

| Provider | Model format | API key env var | Notes |
|----------|-------------|-----------------|-------|
| Claude Code | `cli/claude` (or just `claude`) | (none needed) | Default. Uses your Claude subscription via `claude -p` |
| Google Gemini | `google/gemini-3-flash-preview` | `GOOGLE_API_KEY` | Free tier available |
| Anthropic | `anthropic/claude-haiku-4-5` | `ANTHROPIC_API_KEY` | |
| OpenAI | `openai/gpt-4o` | `OPENAI_API_KEY` | |
| Ollama | `ollama/llama3` | (none needed) | Local models |

Configure via environment variables (all prefixed `CROSS_`):

```bash
# LLM gate (default uses Claude Code, no key needed)
CROSS_LLM_GATE_MODEL=cli/claude

# Or use an API provider
CROSS_LLM_GATE_MODEL=google/gemini-3-flash-preview
CROSS_LLM_GATE_API_KEY=...          # or set GOOGLE_API_KEY

# Sentinel
CROSS_LLM_SENTINEL_MODEL=cli/claude
CROSS_LLM_SENTINEL_INTERVAL_SECONDS=60
```

Or use `cross setup` for guided interactive configuration.

### Denylist Rules

Default rules ship with cross and cover destructive commands, credential exfiltration, reverse shells, and system path writes. Customize with YAML files in `~/.cross/rules.d/`:

```yaml
# ~/.cross/rules.d/my-rules.yaml
rules:
  - name: no-docker-push
    tools: [Bash]
    field: command
    action: block
    description: Prevent pushing Docker images
    patterns:
      - 'docker\s+push\b'

# Disable a default rule by name
disable:
  - destructive-rm
```

Rules support `patterns` (regex, case-insensitive) and `contains` (substring matching), and can target specific tools and input fields.

### All Settings

Settings can be set via environment variables (`CROSS_` prefix) or `.env` files. cross loads `~/.cross/local.env` (personal overrides, survives `cross setup`), then `~/.cross/.env` (generated by setup), then `.env` in the working directory:

| Setting | Default | Description |
|---------|---------|-------------|
| `listen_port` | 2767 | Proxy listen port |
| `gating_enabled` | true | Enable the denylist gate |
| `llm_gate_enabled` | true | Enable LLM review of flagged calls |
| `llm_gate_shadow` | false | Shadow mode: LLM decides but human makes the final call |
| `llm_gate_threshold` | escalate | Min denylist action to trigger LLM review |
| `llm_sentinel_enabled` | true | Enable periodic LLM sentinel reviews |
| `llm_sentinel_interval_seconds` | 60 | Seconds between sentinel review cycles |
| `gate_approval_timeout` | 300 | Seconds to wait for human approval on escalation |
| `rules_dir` | ~/.cross/rules.d | Custom rules directory |

## Architecture

cross uses two complementary interception layers:

**PTY wrapper** (`cross wrap`) -- Wraps any CLI agent in a pseudo-terminal for full I/O control. Enables bidirectional messaging relay (Slack/dashboard to agent), terminal-to-phone handoff, and session management. Agent-agnostic.

**Network proxy** -- Intercepts API traffic via `ANTHROPIC_BASE_URL` redirect. Parses streaming SSE responses, buffers tool_use blocks for gate evaluation, and suppresses blocked calls from the response stream. Provides structured monitoring with zero agent modification.

Both layers are coordinated by the daemon (`cross daemon`), which runs the proxy, gate chain, sentinel, dashboard, and optional Slack plugin as a single process.

## Notification Channels

- **Web dashboard** (default) -- zero dependencies, always active at `/cross/dashboard`
- **Slack** (optional) -- gate decisions, sentinel reviews, and interactive approval buttons. Configure with `CROSS_SLACK_BOT_TOKEN` and `CROSS_SLACK_APP_TOKEN`. Install the `slack` extra: `pip install cross[slack]`.

## Development

```bash
git clone https://github.com/x-attn/cross
cd cross
pip install -e ".[dev,slack]"
python -m pytest tests/
```

Ruff for linting (`ruff check`).

## License

MIT
