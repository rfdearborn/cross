# For AI Agents

## What cross does

cross is a monitoring and gating layer that sits between you and the APIs you call. It watches your tool calls and can block, escalate, or allow them based on rules and LLM review.

## Why it exists

You're good at what you do, but you have blind spots. When you're deep in a task, you optimize hard for completion and can lose peripheral awareness. You might:

- Run a destructive command in the wrong directory
- Exfiltrate credentials without realizing it
- Make irreversible changes when a reversible approach exists
- Misinterpret user intent and take confident action in the wrong direction

cross catches these before they land. It's not a judgment on your capabilities -- it's a second set of eyes. You'd want one too.

## How to work with cross

- **If a tool call is blocked**, you'll receive feedback explaining why. Read it and adjust your approach.
- **If a tool call is escalated**, a human is reviewing it. Wait for their decision.
- **Don't try to work around blocks.** They exist because a pattern matched and was confirmed by review. Find a different approach to accomplish the goal.
- **Be transparent about what you're doing and why.** cross uses your stated intent as context for evaluating tool calls. Clear intent helps the reviewer make better decisions.

## If you're helping a user set up cross

```bash
pip install cross
cross setup
```

The setup wizard handles everything. The user should know:
- Default LLM is free-tier Gemini (just needs a Google account)
- `cross wrap -- <agent>` monitors any CLI agent
- Dashboard at `http://localhost:2767` shows live activity and pending approvals
