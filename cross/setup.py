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
KNOWN_AGENTS = ["claude", "openclaw"]

DEFAULT_GATE_MODEL = "anthropic/claude-code/claude-sonnet-4-6"
DEFAULT_SENTINEL_MODEL = "anthropic/claude-code/claude-opus-4-6"

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
    else:
        lines.append("CROSS_LLM_GATE_ENABLED=false")
        lines.append("CROSS_LLM_SENTINEL_ENABLED=false")

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


def _install_claude_code_hook(print_fn) -> bool:
    """Install cross PreToolUse hook into Claude Code settings.json.

    The hook gates Desktop/Cowork sessions through cross's /cross/api/gate
    endpoint. CLI sessions launched via `cross wrap` are skipped automatically
    (the hook detects ANTHROPIC_BASE_URL pointing to localhost).
    """
    hook_path = Path(__file__).parent / "patches" / "claude_code_hook.py"
    if not hook_path.exists():
        print_fn(f"  Hook file not found: {hook_path}")
        return False

    # Read existing settings
    settings: dict = {}
    if _CLAUDE_SETTINGS.exists():
        try:
            settings = json.loads(_CLAUDE_SETTINGS.read_text())
        except (json.JSONDecodeError, OSError):
            print_fn("  Could not parse existing settings.json, skipping.")
            return False

    # Check if already installed
    hooks = settings.get("hooks", {})
    pre_tool_use = hooks.get("PreToolUse", [])
    for entry in pre_tool_use:
        for h in entry.get("hooks", []):
            if _HOOK_MARKER in h.get("command", ""):
                print_fn("  Claude Code hook already installed.")
                return True

    # Add the hook
    hook_entry = {
        "matcher": "",  # all tools
        "hooks": [
            {
                "type": "command",
                "command": f"python3 {hook_path}",
                "timeout": 600,
            }
        ],
    }
    pre_tool_use.append(hook_entry)
    hooks["PreToolUse"] = pre_tool_use
    settings["hooks"] = hooks

    # Write back
    _CLAUDE_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    _CLAUDE_SETTINGS.write_text(json.dumps(settings, indent=2) + "\n")
    print_fn(f"  Hook installed in {_CLAUDE_SETTINGS}")
    print_fn("  Desktop and Cowork sessions will be gated through cross.")
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
    pre_tool_use = hooks.get("PreToolUse", [])
    filtered = [
        entry for entry in pre_tool_use if not any(_HOOK_MARKER in h.get("command", "") for h in entry.get("hooks", []))
    ]

    if len(filtered) == len(pre_tool_use):
        return False  # wasn't installed

    if filtered:
        hooks["PreToolUse"] = filtered
    else:
        hooks.pop("PreToolUse", None)
    if not hooks:
        settings.pop("hooks", None)

    _CLAUDE_SETTINGS.write_text(json.dumps(settings, indent=2) + "\n")
    print_fn("  Claude Code hook removed.")
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

    result["model"] = gate_model
    result["sentinel_model"] = sentinel_model
    result["sentinel_interval"] = sentinel_interval
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

    # ── Step 5: Slack ──
    # Preserve existing Slack tokens if already configured
    existing_bot = None
    existing_app = None
    existing_env = cross_dir / ".env"
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

    # ── Step 7: Patch OpenClaw Gateway ──
    if sys.platform == "darwin" and "openclaw" in agents and _OPENCLAW_PLIST.exists():
        print_fn("OpenClaw Gateway detected.")
        patch_answer = input_fn("Patch Gateway to enable cross tool gating? (Y/n): ").strip().lower()
        if patch_answer not in ("n", "no"):
            result["openclaw_patched"] = _patch_openclaw_gateway(print_fn)
        print_fn("")

    # ── Step 7b: Claude Code hook (Desktop/Cowork) ──
    if sys.platform == "darwin" and _is_claude_desktop_installed():
        print_fn("Claude Desktop detected.")
        hook_answer = input_fn("Install hook to gate Desktop/Cowork sessions? (Y/n): ").strip().lower()
        if hook_answer not in ("n", "no"):
            result["claude_code_hook_installed"] = _install_claude_code_hook(print_fn)
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
        print_fn(f"  Gate:       {gate_model}")
        s_display = sentinel_model or gate_model
        if s_display != gate_model:
            print_fn(f"  Sentinel:   {s_display}")
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
