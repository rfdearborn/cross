"""Interactive setup wizard for cross."""

from __future__ import annotations

import getpass
import json
import os
import re
import shutil
import socket
import sys
from pathlib import Path

# Agents we detect on PATH (cross wrap should work with any CLI agent)
KNOWN_AGENTS = ["claude", "codex", "openclaw"]

DEFAULT_GATE_MODEL = "anthropic/claude-code/claude-sonnet-4-6"
DEFAULT_SENTINEL_MODEL = "anthropic/claude-code/claude-opus-4-6"
DEFAULT_GATE_BACKUP_MODEL = "openai/gpt-5.4-mini"
DEFAULT_SENTINEL_BACKUP_MODEL = "openai/gpt-5.4"

# Provider → env var name for API key lookup
_KEY_ENV_VARS: dict[str, str] = {
    "google": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# Shell wrapper template per agent
SHELL_WRAPPER_TEMPLATE = '{agent}() {{ cross wrap -- {agent} "$@"; }}'
SHELL_WRAPPER_HEADER = "# cross — AI agent monitoring"

# macOS LaunchAgent plist template
_LAUNCHD_LABEL = "ai.cross.daemon"
_LAUNCHD_PLIST = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{cross_bin}</string>
        <string>daemon</string>
        <string>--foreground</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{path}</string>
    </dict>
    <key>StandardOutPath</key>
    <string>{log_dir}/daemon.out.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/daemon.err.log</string>
    <key>WorkingDirectory</key>
    <string>{home}</string>
</dict>
</plist>
"""

# Regex to strip ANSI escape sequences from input
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _strip_ansi(s: str) -> str:
    """Strip ANSI escape sequences from a string."""
    return _ANSI_RE.sub("", s)


def _parse_provider(model: str) -> str:
    """Extract provider from a provider/model string.

    Bare "claude" and "anthropic/claude-code/*" are treated as cli provider.
    """
    if model.lower().startswith("anthropic/claude-code/"):
        return "cli"
    if "/" in model:
        return model.split("/", 1)[0].lower()
    if model.lower() == "claude":
        return "cli"
    return "anthropic"


def _resolve_model_key(model: str, getpass_fn, print_fn) -> str | None:
    """Prompt for an API key for the given model. Returns the key, or None."""
    provider = _parse_provider(model)

    if provider == "cli":
        # CLI provider handles its own auth (e.g. Claude Code subscription)
        return "cli"  # non-None sentinel so LLM is treated as enabled

    if provider == "ollama":
        if _check_ollama():
            print_fn("Ollama detected on localhost:11434.")
        else:
            print_fn("Warning: could not connect to Ollama on localhost:11434.")
            print_fn("Make sure Ollama is running before starting cross.")
        return None

    env_var = _KEY_ENV_VARS.get(provider)
    env_hint = f" (leave blank to use ${env_var})" if env_var else ""
    print_fn("")
    raw_key = getpass_fn(f"API key{env_hint}: ").strip()
    api_key = _strip_ansi(raw_key)
    if not api_key and env_var:
        env_value = os.environ.get(env_var, "")
        if env_value:
            print_fn(f"  Found ${env_var} in environment — will use that.")
            return env_value
    return api_key or None


def _detect_agents() -> list[str]:
    """Detect known agents on PATH."""
    found = []
    for agent in KNOWN_AGENTS:
        if shutil.which(agent):
            found.append(agent)
    return found


def _check_ollama() -> bool:
    """Check if Ollama is reachable on localhost:11434."""
    try:
        sock = socket.create_connection(("localhost", 11434), timeout=2)
        sock.close()
        return True
    except (OSError, socket.timeout):
        return False


def _detect_shell_rc() -> Path | None:
    """Detect the user's shell rc file."""
    shell = os.environ.get("SHELL", "")
    home = Path.home()
    if "zsh" in shell:
        return home / ".zshrc"
    elif "bash" in shell:
        return home / ".bashrc"
    return None


def _print_model_options(print_fn, none_label: str = "skip"):
    """Print the shared model choice sub-bullets."""
    print_fn(f'  Enter "none" to {none_label}')
    print_fn("  Enter provider/model for API or local models --")
    print_fn(
        "    e.g. anthropic/claude-haiku-4-5, google/gemini-3-flash-preview, openai/gpt-5-mini, ollama/llama3.1:8b"
    )
    print_fn('  Enter "anthropic/claude-code/<model>" to route through Claude Code (no API cost for Pro/Max users)')


def _build_env_lines(
    gate_model: str | None,
    gate_api_key: str | None,
    llm_enabled: bool,
    slack_bot_token: str | None,
    slack_app_token: str | None,
    sentinel_model: str | None = None,
    sentinel_api_key: str | None = None,
    sentinel_interval: int | None = None,
    native_notifications_enabled: bool = False,
    auto_update_enabled: bool = True,
    email_from: str | None = None,
    email_to: str | None = None,
    email_smtp_host: str | None = None,
    email_smtp_port: int | None = None,
    email_smtp_username: str | None = None,
    email_smtp_password: str | None = None,
    email_imap_host: str | None = None,
    gate_backup_model: str | None = None,
    gate_backup_api_key: str | None = None,
    sentinel_backup_model: str | None = None,
    sentinel_backup_api_key: str | None = None,
) -> list[str]:
    """Build the list of CROSS_* env var lines for the .env file."""
    lines = ["# cross configuration", "# Generated by cross setup", ""]

    if native_notifications_enabled:
        lines.append("CROSS_NATIVE_NOTIFICATIONS_ENABLED=true")
    if not auto_update_enabled:
        lines.append("CROSS_AUTO_UPDATE_ENABLED=false")
    if native_notifications_enabled or not auto_update_enabled:
        lines.append("")

    if llm_enabled and gate_model:
        lines.append(f"CROSS_LLM_GATE_MODEL={gate_model}")
        # cli provider handles its own auth — don't write a key
        if gate_api_key and gate_api_key != "cli":
            lines.append(f"CROSS_LLM_GATE_API_KEY={gate_api_key}")
        if gate_backup_model:
            lines.append(f"CROSS_LLM_GATE_BACKUP_MODEL={gate_backup_model}")
            if gate_backup_api_key and gate_backup_api_key != "cli":
                lines.append(f"CROSS_LLM_GATE_BACKUP_API_KEY={gate_backup_api_key}")
        lines.append("")

        if sentinel_model == "none":
            lines.append("CROSS_LLM_SENTINEL_ENABLED=false")
        else:
            s_model = sentinel_model or gate_model
            s_key = sentinel_api_key or gate_api_key
            lines.append(f"CROSS_LLM_SENTINEL_MODEL={s_model}")
            # cli provider handles its own auth — don't write a key
            if s_key and s_key != "cli":
                lines.append(f"CROSS_LLM_SENTINEL_API_KEY={s_key}")
            if sentinel_interval is not None:
                lines.append(f"CROSS_LLM_SENTINEL_INTERVAL_SECONDS={sentinel_interval}")
            if sentinel_backup_model:
                lines.append(f"CROSS_LLM_SENTINEL_BACKUP_MODEL={sentinel_backup_model}")
                if sentinel_backup_api_key and sentinel_backup_api_key != "cli":
                    lines.append(f"CROSS_LLM_SENTINEL_BACKUP_API_KEY={sentinel_backup_api_key}")
    else:
        lines.append("CROSS_LLM_GATE_ENABLED=false")
        lines.append("CROSS_LLM_SENTINEL_ENABLED=false")

    if email_from and email_to:
        lines.append("")
        lines.append(f"CROSS_EMAIL_FROM={email_from}")
        lines.append(f"CROSS_EMAIL_TO={email_to}")
        if email_smtp_host:
            lines.append(f"CROSS_EMAIL_SMTP_HOST={email_smtp_host}")
        if email_smtp_port:
            lines.append(f"CROSS_EMAIL_SMTP_PORT={email_smtp_port}")
        if email_smtp_username:
            lines.append(f"CROSS_EMAIL_SMTP_USERNAME={email_smtp_username}")
        if email_smtp_password:
            lines.append(f"CROSS_EMAIL_SMTP_PASSWORD={email_smtp_password}")
        if email_imap_host:
            lines.append(f"CROSS_EMAIL_IMAP_HOST={email_imap_host}")

    if slack_bot_token:
        lines.append("")
        lines.append(f"CROSS_SLACK_BOT_TOKEN={slack_bot_token}")
    if slack_app_token:
        lines.append(f"CROSS_SLACK_APP_TOKEN={slack_app_token}")

    lines.append("")  # trailing newline
    return lines


def _build_shell_wrappers(agents: list[str]) -> str:
    """Build the shell wrapper block for the given agents."""
    lines = [SHELL_WRAPPER_HEADER]
    for agent in agents:
        lines.append(SHELL_WRAPPER_TEMPLATE.format(agent=agent))
    return "\n".join(lines)


_OPENCLAW_PLIST = Path.home() / "Library" / "LaunchAgents" / "ai.openclaw.gateway.plist"
_OPENCLAW_HOOK_KEY = "NODE_OPTIONS"


def _patch_openclaw_gateway(print_fn) -> bool:
    """Patch the OpenClaw Gateway LaunchAgent to load the cross tool hook."""
    if not _OPENCLAW_PLIST.exists():
        return False

    hook_path = Path(__file__).parent / "patches" / "openclaw_hook.mjs"
    if not hook_path.exists():
        print_fn(f"  Hook file not found: {hook_path}")
        return False

    content = _OPENCLAW_PLIST.read_text()
    import_arg = f"--import {hook_path}"

    # Already patched?
    if str(hook_path) in content:
        print_fn("  OpenClaw Gateway already patched.")
        return True

    # Backup before modifying
    backup_path = _OPENCLAW_PLIST.with_suffix(".plist.bak")
    backup_path.write_text(content)
    print_fn(f"  Backup: {backup_path}")

    # Inject NODE_OPTIONS into EnvironmentVariables
    if "<key>NODE_OPTIONS</key>" in content:
        # NODE_OPTIONS exists — append our --import
        import re

        pattern = r"(<key>NODE_OPTIONS</key>\s*<string>)(.*?)(</string>)"
        replacement = rf"\g<1>\g<2> {import_arg}\g<3>"
        content = re.sub(pattern, replacement, content)
    else:
        # Add NODE_OPTIONS before the closing </dict> of EnvironmentVariables
        env_insert = f"    <key>{_OPENCLAW_HOOK_KEY}</key>\n    <string>{import_arg}</string>\n"
        # Insert before the closing </dict> of EnvironmentVariables
        # The EnvironmentVariables dict ends with </dict>, find it
        last_dict_close = content.rfind("    </dict>")
        if last_dict_close == -1:
            print_fn("  Could not find EnvironmentVariables in plist.")
            return False
        content = content[:last_dict_close] + env_insert + content[last_dict_close:]

    _OPENCLAW_PLIST.write_text(content)
    print_fn(f"  Patched: {_OPENCLAW_PLIST}")
    print_fn("  Restart Gateway: launchctl kickstart -k gui/$(id -u)/ai.openclaw.gateway")
    return True


_CLAUDE_SETTINGS = Path.home() / ".claude" / "settings.json"
_HOOK_MARKER = "claude_code_hook.py"
_PERMISSION_HOOK_MARKER = "permission_hook.py"


def _read_claude_settings(print_fn) -> dict | None:
    """Read and parse Claude Code settings.json, or return None on failure."""
    if not _CLAUDE_SETTINGS.exists():
        return {}
    try:
        return json.loads(_CLAUDE_SETTINGS.read_text())
    except (json.JSONDecodeError, OSError):
        print_fn("  Could not parse existing settings.json, skipping.")
        return None


def _write_claude_settings(settings: dict):
    """Write Claude Code settings.json."""
    _CLAUDE_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    _CLAUDE_SETTINGS.write_text(json.dumps(settings, indent=2) + "\n")


def _install_gate_hook(print_fn) -> bool:
    """Install cross PreToolUse gate hook for Desktop/Cowork sessions."""
    hook_path = Path(__file__).parent / "patches" / "claude_code_hook.py"
    if not hook_path.exists():
        print_fn(f"  Hook file not found: {hook_path}")
        return False

    settings = _read_claude_settings(print_fn)
    if settings is None:
        return False

    hooks = settings.get("hooks", {})
    pre_tool_use = hooks.get("PreToolUse", [])
    if any(_HOOK_MARKER in h.get("command", "") for entry in pre_tool_use for h in entry.get("hooks", [])):
        print_fn("  PreToolUse gate hook already installed.")
        return True

    pre_tool_use.append(
        {
            "matcher": "",
            "hooks": [{"type": "command", "command": f"python3 {hook_path}", "timeout": 600}],
        }
    )
    hooks["PreToolUse"] = pre_tool_use
    settings["hooks"] = hooks
    _write_claude_settings(settings)
    print_fn("  PreToolUse gate hook installed.")
    return True


def _install_permission_hook(print_fn) -> bool:
    """Install cross PermissionRequest notification hook.

    Notifies the daemon when a permission prompt appears, enabling delayed
    relay to Slack/email/dashboard. Works for both CLI and Desktop sessions.
    """
    hook_path = Path(__file__).parent / "patches" / "permission_hook.py"
    if not hook_path.exists():
        print_fn(f"  Hook file not found: {hook_path}")
        return False

    settings = _read_claude_settings(print_fn)
    if settings is None:
        return False

    hooks = settings.get("hooks", {})
    perm_request = hooks.get("PermissionRequest", [])
    if any(_PERMISSION_HOOK_MARKER in h.get("command", "") for entry in perm_request for h in entry.get("hooks", [])):
        print_fn("  PermissionRequest hook already installed.")
        return True

    perm_request.append(
        {
            "matcher": "",
            "hooks": [{"type": "command", "command": f"python3 {hook_path}", "timeout": 10}],
        }
    )
    hooks["PermissionRequest"] = perm_request
    settings["hooks"] = hooks
    _write_claude_settings(settings)
    print_fn("  PermissionRequest hook installed.")
    return True


def _install_claude_code_hook(print_fn) -> bool:
    """Install both cross hooks (gate + permission). Legacy entry point."""
    _install_gate_hook(print_fn)
    _install_permission_hook(print_fn)
    return True


def _uninstall_claude_code_hook(print_fn) -> bool:
    """Remove cross hook from Claude Code settings.json."""
    if not _CLAUDE_SETTINGS.exists():
        return False

    try:
        settings = json.loads(_CLAUDE_SETTINGS.read_text())
    except (json.JSONDecodeError, OSError):
        return False

    hooks = settings.get("hooks", {})
    changed = False

    # Remove PreToolUse gate hook
    pre_tool_use = hooks.get("PreToolUse", [])
    filtered = [
        entry for entry in pre_tool_use if not any(_HOOK_MARKER in h.get("command", "") for h in entry.get("hooks", []))
    ]
    if len(filtered) != len(pre_tool_use):
        changed = True
        if filtered:
            hooks["PreToolUse"] = filtered
        else:
            hooks.pop("PreToolUse", None)

    # Remove PermissionRequest hook
    perm_request = hooks.get("PermissionRequest", [])
    filtered_perm = [
        entry
        for entry in perm_request
        if not any(_PERMISSION_HOOK_MARKER in h.get("command", "") for h in entry.get("hooks", []))
    ]
    if len(filtered_perm) != len(perm_request):
        changed = True
        if filtered_perm:
            hooks["PermissionRequest"] = filtered_perm
        else:
            hooks.pop("PermissionRequest", None)

    if not changed:
        return False

    if not hooks:
        settings.pop("hooks", None)

    _CLAUDE_SETTINGS.write_text(json.dumps(settings, indent=2) + "\n")
    print_fn("  Claude Code hooks removed.")
    return True


def _install_launchd(cross_dir: Path, print_fn) -> bool:
    """Install a macOS LaunchAgent to auto-start the daemon."""
    cross_bin = shutil.which("cross")
    if not cross_bin:
        # Fall back to python -m cross
        cross_bin = sys.executable
        # Can't easily make a plist for `python -m cross daemon`
        print_fn("Could not find cross binary. Skipping auto-start.")
        return False

    launch_agents_dir = Path.home() / "Library" / "LaunchAgents"
    launch_agents_dir.mkdir(parents=True, exist_ok=True)

    plist_path = launch_agents_dir / f"{_LAUNCHD_LABEL}.plist"
    log_dir = cross_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    plist_content = _LAUNCHD_PLIST.format(
        label=_LAUNCHD_LABEL,
        cross_bin=cross_bin,
        path=os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        log_dir=str(log_dir),
        home=str(Path.home()),
    )
    plist_path.write_text(plist_content)
    print_fn(f"LaunchAgent installed: {plist_path}")
    print_fn(f"Logs: {log_dir}/")

    # Load it
    os.system(f"launchctl bootout gui/$(id -u) {plist_path} 2>/dev/null")
    exit_code = os.system(f"launchctl bootstrap gui/$(id -u) {plist_path}")
    if exit_code == 0:
        print_fn("Daemon is now running and will auto-start on login.")
    else:
        print_fn("LaunchAgent installed but could not be loaded. Start manually with: cross daemon")

    return True


def _is_claude_desktop_installed() -> bool:
    """Check if Claude Desktop app is installed."""
    return Path("/Applications/Claude.app").exists()


_CODEX_HOOKS_FILE = Path.home() / ".codex" / "hooks.json"
_CODEX_HOOK_MARKER = "codex_hook.py"


def _read_codex_hooks(print_fn) -> dict | None:
    """Read and parse Codex hooks.json, or return None on failure."""
    if not _CODEX_HOOKS_FILE.exists():
        return {}
    try:
        return json.loads(_CODEX_HOOKS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        print_fn("  Could not parse existing Codex hooks.json, skipping.")
        return None


def _write_codex_hooks(config: dict):
    """Write Codex hooks.json."""
    _CODEX_HOOKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CODEX_HOOKS_FILE.write_text(json.dumps(config, indent=2) + "\n")


def _install_codex_hook(print_fn) -> bool:
    """Install cross gate hook for Codex CLI sessions."""
    hook_path = Path(__file__).parent / "patches" / "codex_hook.py"
    if not hook_path.exists():
        print_fn(f"  Hook file not found: {hook_path}")
        return False

    config = _read_codex_hooks(print_fn)
    if config is None:
        return False

    hooks = config.get("hooks", {})
    pre_tool_use = hooks.get("PreToolUse", [])
    if any(_CODEX_HOOK_MARKER in h.get("command", "") for entry in pre_tool_use for h in entry.get("hooks", [])):
        print_fn("  Codex PreToolUse hook already installed.")
        return True

    pre_tool_use.append(
        {
            "matcher": "",
            "hooks": [{"type": "command", "command": f"python3 {hook_path}", "timeout": 600}],
        }
    )
    hooks["PreToolUse"] = pre_tool_use
    config["hooks"] = hooks
    _write_codex_hooks(config)
    print_fn("  Codex PreToolUse hook installed.")
    return True


def _uninstall_codex_hook(print_fn) -> bool:
    """Remove cross hook from Codex hooks.json."""
    if not _CODEX_HOOKS_FILE.exists():
        return False

    try:
        config = json.loads(_CODEX_HOOKS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return False

    hooks = config.get("hooks", {})
    pre_tool_use = hooks.get("PreToolUse", [])
    filtered = [
        entry
        for entry in pre_tool_use
        if not any(_CODEX_HOOK_MARKER in h.get("command", "") for h in entry.get("hooks", []))
    ]
    if len(filtered) == len(pre_tool_use):
        return False

    if filtered:
        hooks["PreToolUse"] = filtered
    else:
        hooks.pop("PreToolUse", None)

    if not hooks:
        config.pop("hooks", None)

    _CODEX_HOOKS_FILE.write_text(json.dumps(config, indent=2) + "\n")
    print_fn("  Codex hook removed.")
    return True


def run_setup(
    cross_dir: Path | None = None,
    input_fn=input,
    getpass_fn=getpass.getpass,
    print_fn=print,
) -> dict:
    """Run the interactive setup wizard.

    Parameters allow dependency injection for testing:
      cross_dir: override ~/.cross directory
      input_fn: override input() for prompts
      getpass_fn: override getpass.getpass() for secret input
      print_fn: override print() for output

    Returns a dict summarizing what was configured.
    """
    result: dict = {
        "config_dir": None,
        "env_file": None,
        "rules_copied": False,
        "model": None,
        "llm_enabled": False,
        "email_configured": False,
        "slack_configured": False,
        "shell_wrappers_installed": False,
        "agents_found": [],
        "shell_rc": None,
        "autostart_installed": False,
    }

    if cross_dir is None:
        from cross.config import settings

        cross_dir = Path(settings.config_dir).expanduser()

    result["config_dir"] = str(cross_dir)

    # ── Step 1: Welcome ──
    print_fn("")
    print_fn("       │")
    print_fn("       │")
    print_fn("  ─────┼─────")
    print_fn("       │")
    print_fn("       │")
    print_fn("")
    print_fn("  Welcome to cross!")
    print_fn("")
    print_fn("  cross makes AI agents more trustworthy and more capable with lightweight")
    print_fn("  gating and monitoring.")
    print_fn("")

    # ── Step 2: Detect agents ──
    agents = _detect_agents()
    result["agents_found"] = agents
    if agents:
        print_fn(f"Agents found: {', '.join(agents)}")
    else:
        print_fn("No supported agents found on PATH (you can still use cross as a proxy)")
    print_fn("")

    # ── Step 3: Gate model ──
    print_fn("cross uses LLMs to review flagged tool calls and monitor agent behavior.")
    print_fn("")
    print_fn("Choice for tool call reviewing?")
    _print_model_options(print_fn, none_label="skip")
    model_input = input_fn(f"Choice [{DEFAULT_GATE_MODEL}]: ").strip()

    gate_backup_model = None
    gate_backup_api_key = None
    sentinel_backup_model = None
    sentinel_backup_api_key = None

    if model_input.lower() == "none":
        gate_model = None
        gate_api_key = None
        llm_enabled = False
        sentinel_model = None
        sentinel_api_key = None
        sentinel_interval = None
        print_fn("LLM review disabled. Deterministic denylist rules only.")
    else:
        if not model_input or model_input.lower() in ("claude", "cli/claude"):
            gate_model = DEFAULT_GATE_MODEL
        else:
            gate_model = model_input
        llm_enabled = True
        gate_api_key = _resolve_model_key(gate_model, getpass_fn, print_fn)
        if gate_api_key is None and _parse_provider(gate_model) not in ("ollama", "cli"):
            print_fn("No API key provided. LLM features will be disabled.")
            llm_enabled = False
            gate_model = None

    # ── Step 3a-backup: Gate backup model ──
    if llm_enabled and gate_model:
        print_fn("")
        print_fn("Backup model for tool call reviewing (used if primary is unavailable)?")
        _print_model_options(print_fn, none_label="skip (no backup)")
        gate_backup_input = input_fn(f"Choice [{DEFAULT_GATE_BACKUP_MODEL}]: ").strip()
        if gate_backup_input.lower() == "none":
            gate_backup_model = None
        else:
            if not gate_backup_input:
                gate_backup_model = DEFAULT_GATE_BACKUP_MODEL
            else:
                gate_backup_model = gate_backup_input
            gate_backup_api_key = _resolve_model_key(gate_backup_model, getpass_fn, print_fn)
            if gate_backup_api_key is None and _parse_provider(gate_backup_model) not in ("ollama", "cli"):
                print_fn("No API key for backup model — backup disabled.")
                gate_backup_model = None

    # ── Step 3b: Sentinel model + interval ──
    sentinel_model = None
    sentinel_api_key = None
    sentinel_interval = None

    if llm_enabled:
        print_fn("")
        print_fn("Choice for periodic session monitoring?")
        _print_model_options(print_fn, none_label="disable session monitoring")
        sentinel_input = input_fn(f"Choice [{DEFAULT_SENTINEL_MODEL}]: ").strip()

        if sentinel_input.lower() == "none":
            sentinel_model = "none"  # signals disabled
        else:
            if not sentinel_input or sentinel_input.lower() in ("claude", "cli/claude"):
                sentinel_model = DEFAULT_SENTINEL_MODEL
            else:
                sentinel_model = sentinel_input
            sentinel_provider = _parse_provider(sentinel_model)
            gate_provider = _parse_provider(gate_model)
            # Only ask for a key if it's a different provider
            if sentinel_provider != gate_provider and sentinel_provider not in ("ollama", "cli"):
                sentinel_api_key = _resolve_model_key(sentinel_model, getpass_fn, print_fn)

        if sentinel_model != "none":
            print_fn("")
            print_fn("Sentinel review interval?")
            interval_input = input_fn("Seconds [60]: ").strip()
            if interval_input:
                try:
                    sentinel_interval = int(interval_input)
                except ValueError:
                    print_fn("  Invalid number, using default (60s).")

            # ── Step 3b-backup: Sentinel backup model ──
            print_fn("")
            print_fn("Backup model for session monitoring (used if primary is unavailable)?")
            _print_model_options(print_fn, none_label="skip (no backup)")
            sentinel_backup_input = input_fn(f"Choice [{DEFAULT_SENTINEL_BACKUP_MODEL}]: ").strip()
            if sentinel_backup_input.lower() == "none":
                sentinel_backup_model = None
            else:
                if not sentinel_backup_input:
                    sentinel_backup_model = DEFAULT_SENTINEL_BACKUP_MODEL
                else:
                    sentinel_backup_model = sentinel_backup_input
                sentinel_provider = _parse_provider(sentinel_backup_model)
                # Reuse gate backup key if same provider
                gate_backup_provider = _parse_provider(gate_backup_model) if gate_backup_model else ""
                if sentinel_provider == gate_backup_provider and gate_backup_api_key:
                    sentinel_backup_api_key = gate_backup_api_key
                elif sentinel_provider not in ("ollama", "cli"):
                    sentinel_backup_api_key = _resolve_model_key(sentinel_backup_model, getpass_fn, print_fn)
                    if sentinel_backup_api_key is None:
                        print_fn("No API key for backup model — backup disabled.")
                        sentinel_backup_model = None

    result["model"] = gate_model
    result["sentinel_model"] = sentinel_model
    result["sentinel_interval"] = sentinel_interval
    result["gate_backup_model"] = gate_backup_model
    result["sentinel_backup_model"] = sentinel_backup_model
    result["llm_enabled"] = llm_enabled
    print_fn("")

    # ── Step 4: Desktop notifications (macOS) ──
    native_notifications_enabled = False
    if sys.platform == "darwin":
        notif_answer = input_fn("Would you like to configure desktop push notifications? (Y/n): ").strip().lower()
        if notif_answer not in ("n", "no"):
            if not shutil.which("terminal-notifier"):
                print_fn("Installing terminal-notifier...")
                exit_code = os.system("brew install terminal-notifier")
                if exit_code != 0:
                    print_fn("Could not install terminal-notifier (requires Homebrew).")
                    print_fn("Browser notifications will be used as fallback.")
            if shutil.which("terminal-notifier"):
                native_notifications_enabled = True
                result["native_notifications_enabled"] = True
                print_fn("Desktop notifications enabled.")
        print_fn("")

    # ── Step 5: Email ──
    # Preserve existing email config if already configured
    existing_env = cross_dir / ".env"
    existing_email_from = None
    existing_email_to = None
    if existing_env.exists():
        for line in existing_env.read_text().splitlines():
            if line.startswith("CROSS_EMAIL_FROM="):
                existing_email_from = line.split("=", 1)[1].strip()
            elif line.startswith("CROSS_EMAIL_TO="):
                existing_email_to = line.split("=", 1)[1].strip()

    email_from = None
    email_to = None
    email_smtp_host = None
    email_smtp_port = None
    email_smtp_username = None
    email_smtp_password = None
    email_imap_host = None

    if existing_email_from and existing_email_to:
        keep_answer = input_fn("Email is already configured. Keep existing config? (Y/n): ").strip().lower()
        if keep_answer not in ("n", "no"):
            email_from = existing_email_from
            email_to = existing_email_to
            result["email_configured"] = True
            print_fn("Keeping existing email configuration.")
        else:
            existing_email_from = None

    if not (existing_email_from and existing_email_to):
        email_answer = input_fn("Would you like to configure email notifications? (y/N): ").strip().lower()
        if email_answer in ("y", "yes"):
            email_from = input_fn("Sender address (e.g. cross@example.com): ").strip()
            email_to = input_fn("Recipient address (e.g. you@example.com): ").strip()
            if email_from and email_to:
                email_smtp_host = input_fn("SMTP host [localhost]: ").strip() or None
                port_input = input_fn("SMTP port [587]: ").strip()
                email_smtp_port = int(port_input) if port_input else None
                email_smtp_username = _strip_ansi(getpass_fn("SMTP username (blank to skip): ").strip()) or None
                if email_smtp_username:
                    email_smtp_password = _strip_ansi(getpass_fn("SMTP password: ").strip()) or None
                imap_answer = input_fn("Enable inbound replies via IMAP? (y/N): ").strip().lower()
                if imap_answer in ("y", "yes"):
                    email_imap_host = input_fn("IMAP host (e.g. imap.gmail.com): ").strip() or None
                result["email_configured"] = True
                print_fn("Email configured.")
            else:
                print_fn("Email addresses not provided, skipping.")
                email_from = None
                email_to = None
    print_fn("")

    # ── Step 6: Slack ──
    # Preserve existing Slack tokens if already configured
    existing_bot = None
    existing_app = None
    if existing_env.exists():
        for line in existing_env.read_text().splitlines():
            if line.startswith("CROSS_SLACK_BOT_TOKEN="):
                existing_bot = line.split("=", 1)[1].strip()
            elif line.startswith("CROSS_SLACK_APP_TOKEN="):
                existing_app = line.split("=", 1)[1].strip()

    slack_bot_token = None
    slack_app_token = None

    if existing_bot and existing_app:
        keep_answer = input_fn("Slack is already configured. Keep existing config? (Y/n): ").strip().lower()
        if keep_answer not in ("n", "no"):
            slack_bot_token = existing_bot
            slack_app_token = existing_app
            result["slack_configured"] = True
            print_fn("Keeping existing Slack configuration.")
        else:
            existing_bot = None

    if not (existing_bot and existing_app):
        slack_answer = input_fn("Would you like to configure Slack notifications? (y/N): ").strip().lower()
        if slack_answer in ("y", "yes"):
            slack_bot_token = _strip_ansi(getpass_fn("Enter Slack bot token (xoxb-...): ").strip())
            slack_app_token = _strip_ansi(getpass_fn("Enter Slack app token (xapp-...): ").strip())
            if slack_bot_token and slack_app_token:
                result["slack_configured"] = True
                print_fn("Slack configured.")
            else:
                print_fn("Slack tokens not provided, skipping.")
                slack_bot_token = None
                slack_app_token = None
    print_fn("")

    # ── Step 5: Create config directory ──
    rules_dir = cross_dir / "rules.d"
    rules_dir.mkdir(parents=True, exist_ok=True)

    # Default rules are loaded from the package at runtime — no copy needed.
    # User overrides and additions go in rules.d/.
    # Remove stale default.yaml copies from prior installations.
    stale_defaults = rules_dir / "default.yaml"
    if stale_defaults.exists():
        try:
            stale_defaults.unlink()
            print_fn(f"Removed stale {stale_defaults} (defaults now loaded from package)")
        except OSError:
            pass

    env_file = cross_dir / ".env"

    # ── Step 6: Shell wrapper ──
    if agents:
        wrapper_answer = input_fn("Install shell wrapper for automatic agent monitoring? (Y/n): ").strip().lower()
        if wrapper_answer not in ("n", "no"):
            shell_rc = _detect_shell_rc()
            if shell_rc:
                wrapper_block = _build_shell_wrappers(agents)
                # Check if already installed
                existing_content = ""
                if shell_rc.exists():
                    existing_content = shell_rc.read_text()

                if SHELL_WRAPPER_HEADER in existing_content:
                    print_fn(f"Shell wrappers already present in {shell_rc}")
                else:
                    with open(shell_rc, "a") as f:
                        f.write(f"\n{wrapper_block}\n")
                    print_fn(f"Shell wrappers added to {shell_rc}")
                result["shell_wrappers_installed"] = True
                result["shell_rc"] = str(shell_rc)
            else:
                print_fn("Could not detect shell rc file. Add manually:")
                print_fn(f"  {_build_shell_wrappers(agents)}")
        print_fn("")

    # ── Step 7a: Claude Code permission hook (all Claude Code users) ──
    if "claude" in agents:
        print_fn("Claude Code detected.")
        perm_answer = input_fn("Install hook to relay permission prompts to Slack/email? (Y/n): ").strip().lower()
        if perm_answer not in ("n", "no"):
            result["permission_hook_installed"] = _install_permission_hook(print_fn)
        print_fn("")

    # ── Step 7b: Claude Code gate hook (Desktop/Cowork only) ──
    if sys.platform == "darwin" and _is_claude_desktop_installed():
        print_fn("Claude Desktop detected.")
        hook_answer = input_fn("Install hook to gate Desktop/Cowork sessions? (Y/n): ").strip().lower()
        if hook_answer not in ("n", "no"):
            result["claude_code_hook_installed"] = _install_gate_hook(print_fn)
        print_fn("")

    # ── Step 7b2: Codex hook ──
    if "codex" in agents:
        print_fn("OpenAI Codex CLI detected.")
        codex_answer = input_fn("Install hook to gate Codex agent sessions? (Y/n): ").strip().lower()
        if codex_answer not in ("n", "no"):
            result["codex_hook_installed"] = _install_codex_hook(print_fn)
        print_fn("")

    # ── Step 7c: Patch OpenClaw Gateway ──
    if sys.platform == "darwin" and "openclaw" in agents and _OPENCLAW_PLIST.exists():
        print_fn("OpenClaw Gateway detected.")
        patch_answer = input_fn("Patch Gateway to enable cross tool gating? (Y/n): ").strip().lower()
        if patch_answer not in ("n", "no"):
            result["openclaw_patched"] = _patch_openclaw_gateway(print_fn)
        print_fn("")

    # ── Step 8: Auto-start daemon ──
    if sys.platform == "darwin":
        autostart_answer = input_fn("Auto-start cross daemon on login? (Y/n): ").strip().lower()
        if autostart_answer not in ("n", "no"):
            result["autostart_installed"] = _install_launchd(cross_dir, print_fn)
        print_fn("")

    # ── Step 9: Auto-update ──
    auto_update_enabled = True
    auto_update_answer = input_fn("Enable automatic updates? (Y/n): ").strip().lower()
    if auto_update_answer in ("n", "no"):
        auto_update_enabled = False
    result["auto_update_enabled"] = auto_update_enabled
    print_fn("")

    # ── Write .env file ──
    env_lines = _build_env_lines(
        gate_model,
        gate_api_key,
        llm_enabled,
        slack_bot_token,
        slack_app_token,
        sentinel_model=sentinel_model,
        sentinel_api_key=sentinel_api_key,
        sentinel_interval=sentinel_interval,
        native_notifications_enabled=native_notifications_enabled,
        auto_update_enabled=auto_update_enabled,
        email_from=email_from,
        email_to=email_to,
        email_smtp_host=email_smtp_host,
        email_smtp_port=email_smtp_port,
        email_smtp_username=email_smtp_username,
        email_smtp_password=email_smtp_password,
        email_imap_host=email_imap_host,
        gate_backup_model=gate_backup_model,
        gate_backup_api_key=gate_backup_api_key,
        sentinel_backup_model=sentinel_backup_model,
        sentinel_backup_api_key=sentinel_backup_api_key,
    )
    env_file.write_text("\n".join(env_lines))
    result["env_file"] = str(env_file)

    print_fn(f"Config written to {env_file}")
    print_fn(f"Rules directory: {rules_dir}/")
    print_fn("")

    # ── Summary ──
    print_fn("Setup complete!")
    print_fn("")
    print_fn(f"  Config:     {env_file}")
    print_fn(f"  Rules:      {rules_dir}/")
    if gate_model:
        gate_backup_display = f" (backup: {gate_backup_model})" if gate_backup_model else ""
        print_fn(f"  Gate:       {gate_model}{gate_backup_display}")
        s_display = sentinel_model or gate_model
        if s_display != gate_model:
            sentinel_backup_display = f" (backup: {sentinel_backup_model})" if sentinel_backup_model else ""
            print_fn(f"  Sentinel:   {s_display}{sentinel_backup_display}")
    else:
        print_fn("  LLM:        disabled (deterministic rules only)")
    print_fn("  Dashboard:  http://localhost:2767")
    print_fn("")
    if not result.get("autostart_installed"):
        print_fn("Start monitoring:")
        print_fn("  cross daemon        # start the daemon")
    if agents:
        print_fn(f"  {agents[0]}              # use your agent (auto-monitored via shell wrapper)")
    else:
        print_fn("  cross wrap -- <agent>  # wrap any agent command")
    print_fn("")

    return result
