"""Denylist gate — pattern-matching evaluator for tool calls.

Loads rules from YAML data files:
  - Default rules: cross/rules/default.yaml (shipped with Cross)
  - User rules:    ~/.cross/rules.d/*.yaml (add, modify, or disable)
  - Project rules: <cwd>/.cross/rules.d/*.yaml (per-project, additive)

Project rules are strictly additive — they can only add new rules.
When a project rule has the same name as any global rule, the global
version takes precedence and the project rule is skipped.  Project
``disable`` lists are ignored for safety.

Rules support two match types:
  - patterns:  regex (case-insensitive, any match triggers)
  - contains:  substring (case-insensitive, any match triggers)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path

import yaml

from cross.evaluator import Action, EvaluationResponse, Gate, GateRequest

logger = logging.getLogger("cross.gates.denylist")

_DEFAULT_RULES_PATH = Path(__file__).parent.parent / "rules" / "default.yaml"


@dataclass
class DenylistRule:
    """A single matching rule."""

    name: str
    tools: list[str]  # tool names to match, or ["*"] for all
    action: Action = Action.BLOCK
    field: str = ""  # specific input field to check, or "" for entire input JSON
    patterns: list[str] = dataclass_field(default_factory=list)
    contains: list[str] = dataclass_field(default_factory=list)
    description: str = ""
    _compiled: list[re.Pattern] = dataclass_field(default_factory=list, repr=False)
    _contains_lower: list[str] = dataclass_field(default_factory=list, repr=False)

    def __post_init__(self):
        self._compiled = []
        for p in self.patterns:
            try:
                self._compiled.append(re.compile(p, re.IGNORECASE))
            except re.error as e:
                logger.warning(f"Invalid regex in rule '{self.name}': {p!r} — {e}")
        self._contains_lower = [s.lower() for s in self.contains]


def _load_yaml(path: Path) -> dict | None:
    """Load a YAML file, returning None on failure."""
    try:
        return yaml.safe_load(path.read_text())
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None


def _substitute_variables(patterns: list[str], variables: dict[str, str]) -> list[str]:
    """Replace {{variable}} placeholders in patterns."""
    result = []
    for p in patterns:
        for name, value in variables.items():
            p = p.replace("{{" + name + "}}", value)
        result.append(p)
    return result


def _parse_rules(data: dict | list, source: str = "") -> list[DenylistRule]:
    """Parse rules from a loaded YAML/JSON dict or bare list."""
    if not data:
        return []

    # Accept bare list of rules (e.g., user files without a "rules:" wrapper)
    if isinstance(data, list):
        data = {"rules": data}

    variables = data.get("variables", {})
    rules = []

    for rule_data in data.get("rules", []):
        action_str = rule_data.get("action", "block").upper()
        try:
            action = Action[action_str]
        except KeyError:
            logger.warning(f"Unknown action '{action_str}' in {source}, skipping rule")
            continue

        patterns = rule_data.get("patterns", [])
        if variables and patterns:
            patterns = _substitute_variables(patterns, variables)

        rules.append(
            DenylistRule(
                name=rule_data.get("name", "unnamed"),
                tools=rule_data.get("tools", ["*"]),
                action=action,
                field=rule_data.get("field", ""),
                patterns=patterns,
                contains=rule_data.get("contains", []),
                description=rule_data.get("description", ""),
            )
        )

    return rules


def _load_default_rules() -> list[DenylistRule]:
    """Load the built-in default rules from cross/rules/default.yaml."""
    if not _DEFAULT_RULES_PATH.exists():
        logger.warning(f"Default rules not found at {_DEFAULT_RULES_PATH}")
        return []
    data = _load_yaml(_DEFAULT_RULES_PATH)
    if not data:
        return []
    rules = _parse_rules(data, source=str(_DEFAULT_RULES_PATH))
    logger.info(f"Loaded {len(rules)} default rules")
    return rules


def _load_user_rules(rules_dir: Path) -> tuple[list[DenylistRule], set[str]]:
    """Load user rules and disabled rule names from ~/.cross/rules.d/."""
    rules: list[DenylistRule] = []
    disabled: set[str] = set()

    if not rules_dir.exists():
        return rules, disabled

    for path in sorted(rules_dir.glob("*.yaml")) + sorted(rules_dir.glob("*.json")):
        try:
            if path.suffix == ".yaml":
                data = _load_yaml(path)
            else:
                data = json.loads(path.read_text())

            if not data:
                continue

            # Collect disabled rule names (only if data is a dict)
            if isinstance(data, dict):
                disable_list = data.get("disable", [])
                if isinstance(disable_list, list):
                    disabled.update(disable_list)

            # Parse rules (accepts both dict and bare list)
            parsed = _parse_rules(data, source=str(path))
            rules.extend(parsed)
            if parsed:
                logger.info(f"Loaded {len(parsed)} user rules from {path}")
        except Exception as e:
            logger.warning(f"Failed to load rules from {path}: {e}")

    return rules, disabled


def _normalize_path(value: str) -> str:
    """Normalize a file path for matching: resolve .., strip /private/ (macOS)."""
    if value.startswith("~"):
        value = os.path.expanduser(value)
    value = os.path.normpath(value)
    # macOS: /private/etc -> /etc, /private/var -> /var
    if value.startswith("/private/"):
        stripped = value[8:]  # len("/private") = 8
        if stripped.startswith(("/etc", "/var", "/tmp")):
            value = stripped
    return value


def _dir_mtime(rules_dir: Path) -> float:
    """Return the max mtime across all rule files in a directory, or 0."""
    if not rules_dir or not rules_dir.exists():
        return 0.0
    mtimes = [rules_dir.stat().st_mtime]
    for path in rules_dir.glob("*.yaml"):
        mtimes.append(path.stat().st_mtime)
    for path in rules_dir.glob("*.json"):
        mtimes.append(path.stat().st_mtime)
    return max(mtimes) if mtimes else 0.0


class DenylistGate(Gate):
    """Pattern-matching gate evaluator. Loads rules from YAML data files.

    Hot-reloads user rules when files in rules_dir change.
    Supports per-project rules loaded from ``<cwd>/.cross/rules.d/``.
    """

    def __init__(
        self,
        rules_dir: Path | None = None,
        include_defaults: bool = True,
    ):
        super().__init__(name="denylist")
        self._rules_dir = rules_dir
        self._include_defaults = include_defaults
        self._last_mtime: float = 0.0
        self.rules: list[DenylistRule] = []
        # Cache for project rules: project_rules_dir_str -> (mtime, rules, disabled)
        self._project_cache: dict[str, tuple[float, list[DenylistRule], set[str]]] = {}
        self._load_rules()

    def _load_rules(self):
        """Load (or reload) all rules from defaults + user directory."""
        self.rules = []

        # Load user rules first to get disable list
        user_rules: list[DenylistRule] = []
        disabled: set[str] = set()
        if self._rules_dir:
            user_rules, disabled = _load_user_rules(self._rules_dir)
            self._last_mtime = _dir_mtime(self._rules_dir)

        # Load defaults (unless disabled or excluded)
        if self._include_defaults:
            for rule in _load_default_rules():
                if rule.name not in disabled:
                    self.rules.append(rule)
                else:
                    logger.info(f"Default rule '{rule.name}' disabled by user config")

        # Append user rules
        self.rules.extend(user_rules)

        logger.info(f"DenylistGate loaded with {len(self.rules)} rules")

    def _maybe_reload(self):
        """Reload rules if any file in rules_dir has changed."""
        if not self._rules_dir:
            return
        current_mtime = _dir_mtime(self._rules_dir)
        if current_mtime != self._last_mtime:
            logger.info("Rules directory changed, reloading...")
            self._load_rules()

    def _get_project_rules(self, cwd: str) -> list[DenylistRule]:
        """Load project rules from ``<cwd>/.cross/rules.d/``, with caching.

        Project rules are strictly additive — they can only add new rules, not
        override or disable global ones.  When a project rule has the same name
        as any global rule (default or user), the project rule is skipped.
        Project ``disable`` lists are ignored for safety (a malicious repo
        could otherwise weaken protections).
        """
        if not cwd:
            return []

        project_rules_dir = Path(cwd) / ".cross" / "rules.d"
        if not project_rules_dir.exists():
            return []

        dir_key = str(project_rules_dir)
        current_mtime = _dir_mtime(project_rules_dir)
        cached = self._project_cache.get(dir_key)

        if cached and cached[0] == current_mtime:
            return cached[1]

        # Load project rules (disable lists are intentionally ignored)
        project_rules, project_disabled = _load_user_rules(project_rules_dir)
        if project_disabled:
            logger.warning(f"Project disable list ignored for safety: {project_disabled}")

        # Filter out project rules that share a name with any global rule
        global_names = {r.name for r in self.rules if r.name != "unnamed"}
        filtered: list[DenylistRule] = []
        for rule in project_rules:
            if rule.name in global_names:
                logger.info(f"Project rule '{rule.name}' skipped — overridden by global rule")
            else:
                filtered.append(rule)

        self._project_cache[dir_key] = (current_mtime, filtered)
        if filtered:
            logger.info(f"Loaded {len(filtered)} project rules from {project_rules_dir}")
        return filtered

    # Codex uses different tool names and field names than Claude Code.
    # Map them so denylist rules written for [Bash, exec] with field: command
    # also match Codex's exec_command with field: cmd.
    _TOOL_ALIASES: dict[str, list[str]] = {
        "bash": ["exec_command", "shell", "local_shell", "shell_command"],
        "exec": ["exec_command", "shell", "local_shell", "shell_command"],
    }
    _FIELD_ALIASES: dict[str, list[str]] = {
        "command": ["cmd"],
    }

    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        """Check a tool call against all rules. Returns the highest-severity match."""
        self._maybe_reload()
        best_match: EvaluationResponse | None = None

        # Combine global rules with project-specific rules
        all_rules = self.rules
        project_rules = self._get_project_rules(request.cwd)
        if project_rules:
            all_rules = self.rules + project_rules

        for rule in all_rules:
            # Check if rule applies to this tool (with alias support)
            tool_name_lower = request.tool_name.lower()
            if "*" not in rule.tools:
                matched_tool = any(t.lower() == tool_name_lower for t in rule.tools)
                if not matched_tool:
                    # Check aliases: does any rule tool alias to the request tool?
                    matched_tool = any(tool_name_lower in self._TOOL_ALIASES.get(t.lower(), []) for t in rule.tools)
                if not matched_tool:
                    continue

            # Get the text to match against
            text = self._get_match_text(request, rule.field)
            if not text:
                continue

            matched = False

            # Check contains (substring, case-insensitive)
            if rule._contains_lower:
                text_lower = text.lower()
                for substr in rule._contains_lower:
                    if substr in text_lower:
                        matched = True
                        break

            # Check patterns (regex)
            if not matched:
                for pattern in rule._compiled:
                    if pattern.search(text):
                        matched = True
                        break

            if matched:
                resp = EvaluationResponse(
                    action=rule.action,
                    reason=rule.description or f"Matched rule: {rule.name}",
                    rule_id=rule.name,
                    confidence=1.0,
                    evaluator=self.name,
                )
                if best_match is None or resp.action.value > best_match.action.value:
                    best_match = resp

        if best_match:
            return best_match

        return EvaluationResponse(action=Action.ALLOW, evaluator=self.name)

    def _get_match_text(self, request: GateRequest, field: str) -> str:
        """Extract the text to match against from the request."""
        if not field:
            # Match against entire input as JSON string
            if isinstance(request.tool_input, dict):
                return json.dumps(request.tool_input)
            return str(request.tool_input) if request.tool_input else ""

        # Match against a specific field (with alias support)
        if isinstance(request.tool_input, dict):
            value = request.tool_input.get(field, "")
            # Try field aliases if primary field not found
            if not value:
                for alias in self._FIELD_ALIASES.get(field, []):
                    value = request.tool_input.get(alias, "")
                    if value:
                        break
            value = str(value)
            # Normalize file paths to catch traversal and macOS /private/ bypasses
            if "path" in field.lower() and value:
                value = _normalize_path(value)
            return value

        return ""
