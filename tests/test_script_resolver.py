"""Tests for cross.script_resolver — script content resolution."""

import os

import pytest

from cross.script_resolver import extract_script_paths, resolve_script_contents


class TestExtractScriptPaths:
    def test_python_script(self):
        paths = extract_script_paths("python script.py")
        assert paths == ["script.py"]

    def test_python3_script(self):
        paths = extract_script_paths("python3 myscript.py")
        assert paths == ["myscript.py"]

    def test_python_with_version(self):
        paths = extract_script_paths("python3.11 script.py")
        assert paths == ["script.py"]

    def test_node_script(self):
        paths = extract_script_paths("node app.js")
        assert paths == ["app.js"]

    def test_node_ts_script(self):
        paths = extract_script_paths("node server.ts")
        assert paths == ["server.ts"]

    def test_ruby_script(self):
        paths = extract_script_paths("ruby deploy.rb")
        assert paths == ["deploy.rb"]

    def test_perl_script(self):
        paths = extract_script_paths("perl process.pl")
        assert paths == ["process.pl"]

    def test_bash_script(self):
        paths = extract_script_paths("bash setup.sh")
        assert paths == ["setup.sh"]

    def test_sh_script(self):
        paths = extract_script_paths("sh install.sh")
        assert paths == ["install.sh"]

    def test_php_script(self):
        paths = extract_script_paths("php index.php")
        assert paths == ["index.php"]

    def test_script_with_flags(self):
        paths = extract_script_paths("python -u script.py")
        assert paths == ["script.py"]

    def test_script_with_path(self):
        paths = extract_script_paths("python ./scripts/run.py")
        assert paths == ["./scripts/run.py"]

    def test_script_with_absolute_path(self):
        paths = extract_script_paths("python /home/user/script.py")
        assert paths == ["/home/user/script.py"]

    def test_chained_commands(self):
        paths = extract_script_paths("cd /tmp && python script.py")
        assert "script.py" in paths

    def test_sudo_prefix(self):
        paths = extract_script_paths("sudo python3 script.py")
        assert paths == ["script.py"]

    def test_no_script(self):
        paths = extract_script_paths("ls -la")
        assert paths == []

    def test_git_command(self):
        paths = extract_script_paths("git status")
        assert paths == []

    def test_python_m_flag_not_script(self):
        """python -m module should not extract 'module' as a script path."""
        paths = extract_script_paths("python -m pytest tests/")
        assert paths == []

    def test_python_c_flag_not_script(self):
        """python -c 'code' should not extract 'code' as a script path."""
        paths = extract_script_paths("python -c 'print(1)'")
        assert paths == []

    def test_pipe_command(self):
        paths = extract_script_paths("python script.py | grep error")
        assert paths == ["script.py"]


class TestResolveScriptContents:
    def test_resolves_existing_script(self, tmp_path):
        script = tmp_path / "test_script.py"
        script.write_text("print('hello')\n")

        result = resolve_script_contents(
            f"python {script}",
            cwd=str(tmp_path),
        )
        assert str(script) in result or str(script.resolve()) in result
        contents = list(result.values())[0]
        assert "print('hello')" in contents

    def test_resolves_relative_path(self, tmp_path):
        script = tmp_path / "run.py"
        script.write_text("import os\nos.listdir('/')\n")

        result = resolve_script_contents("python run.py", cwd=str(tmp_path))
        assert len(result) == 1
        contents = list(result.values())[0]
        assert "os.listdir" in contents

    def test_nonexistent_script_returns_empty(self, tmp_path):
        result = resolve_script_contents(
            "python nonexistent.py",
            cwd=str(tmp_path),
        )
        assert result == {}

    def test_non_script_command_returns_empty(self):
        result = resolve_script_contents("git status")
        assert result == {}

    def test_large_file_truncated(self, tmp_path):
        script = tmp_path / "big.py"
        script.write_text("x = 1\n" * 100000)  # Very large file

        result = resolve_script_contents(f"python {script}", cwd=str(tmp_path))
        assert len(result) == 1
        contents = list(result.values())[0]
        assert "too large" in contents

    def test_multiple_interpreters_in_chain(self, tmp_path):
        script1 = tmp_path / "first.py"
        script1.write_text("print('first')\n")
        script2 = tmp_path / "second.py"
        script2.write_text("print('second')\n")

        result = resolve_script_contents(
            f"python {script1} && python {script2}",
            cwd=str(tmp_path),
        )
        assert len(result) == 2
