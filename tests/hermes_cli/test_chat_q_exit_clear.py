"""Regression tests for #53009: chat -q final response erased by exit-summary clear."""

from types import SimpleNamespace
import subprocess

import pytest

import cli as cli_mod
from hermes_cli.cli_session_mixin import CLISessionMixin


# ── A3.1 Test-First: verify _clear_terminal_on_exit gating ──────────────────

def test_print_exit_summary_clears_screen_by_default(monkeypatch):
    """Default behavior: _print_exit_summary() calls _clear_terminal_on_exit()."""
    calls = []

    class FakeCLI:
        conversation_history = []
        session_start = None

        def _clear_terminal_on_exit(self):
            calls.append("clear")

    monkeypatch.setattr(cli_mod, "datetime", SimpleNamespace(
        now=lambda: SimpleNamespace(
            __sub__=lambda self, other: SimpleNamespace(
                total_seconds=lambda: 0
            )
        )
    ))

    fake = FakeCLI()
    cli_mod.HermesCLI._print_exit_summary(fake)  # default clear_screen=True

    assert "clear" in calls, "_clear_terminal_on_exit should be called by default"


def test_print_exit_summary_skips_clear_when_clear_screen_false(monkeypatch):
    """With clear_screen=False, _print_exit_summary() does NOT clear."""
    calls = []

    class FakeCLI:
        conversation_history = []
        session_start = None

        def _clear_terminal_on_exit(self):
            calls.append("clear")

    monkeypatch.setattr(cli_mod, "datetime", SimpleNamespace(
        now=lambda: SimpleNamespace(
            __sub__=lambda self, other: SimpleNamespace(
                total_seconds=lambda: 0
            )
        )
    ))

    fake = FakeCLI()
    cli_mod.HermesCLI._print_exit_summary(fake, clear_screen=False)

    assert "clear" not in calls, (
        "_clear_terminal_on_exit should NOT be called when clear_screen=False"
    )


# ── Production-path test: single-query -q path skips the clear ──────────────

def test_single_query_main_skips_clear_on_exit_summary(monkeypatch):
    """The single-query (-q) path calls _print_exit_summary without clearing."""
    calls = []
    clear_calls = []

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.console = SimpleNamespace(print=lambda *_a, **_kw: calls.append("query-label"))
            self.session_id = "sq-test"
            self.agent = SimpleNamespace(
                session_id="sq-test",
                platform="cli",
            )

        def _claim_active_session(self, surface, *, stderr=False):
            calls.append(("claim", surface, stderr))
            return True

        def _show_security_advisories(self):
            calls.append("advisories")

        def chat(self, query, images=None):
            calls.append(("chat", query, images))
            self._last_turn_result = {"final_response": "done", "completed": True}
            return "done"

        def _print_exit_summary(self, clear_screen=True):
            calls.append(("summary", clear_screen))
            if clear_screen:
                clear_calls.append("CLEARED")  # should NOT happen

    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        cli_mod,
        "_finalize_single_query",
        lambda fake_cli: calls.append(("finalize", fake_cli.session_id)),
    )

    with pytest.raises(SystemExit) as exc_info:  # the one-shot path exits with the turn's outcome
        cli_mod.main(query="hello", quiet=False, toolsets="terminal")

    assert exc_info.value.code == 0
    assert calls == [
        ("claim", "cli", False),
        "query-label",
        "advisories",
        ("chat", "hello", None),
        ("summary", False),  # <-- clear_screen=False for single-query
        ("finalize", "sq-test"),
    ]
    assert len(clear_calls) == 0, (
        "_clear_terminal_on_exit must NOT be called in single-query mode"
    )


# ── Verify interactive mode still clears ────────────────────────────────────

def test_print_exit_summary_still_clears_in_interactive_path(monkeypatch):
    """Interactive mode should still clear the screen (preserving #38928)."""
    from datetime import datetime as real_datetime

    calls = []

    class FakeCLI:
        conversation_history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        session_start = real_datetime(2026, 1, 1, 12, 0, 0)
        session_id = "test-session"
        _session_db = None
        agent = None

        def _clear_terminal_on_exit(self):
            calls.append("clear")

    monkeypatch.setattr(cli_mod, "datetime", SimpleNamespace(
        now=lambda: real_datetime(2026, 1, 1, 12, 1, 0)  # 1 min elapsed
    ))

    fake = FakeCLI()
    cli_mod.HermesCLI._print_exit_summary(fake)  # default clear_screen=True

    assert "clear" in calls, (
        "Interactive mode should still clear the screen (regression test for #38928)"
    )


def test_clear_terminal_fallback_uses_subprocess_without_shell(monkeypatch):
    class BrokenStream:
        def isatty(self):
            return True

        def write(self, _text):
            raise OSError("ansi write failed")

        def flush(self):
            raise AssertionError("flush should not run after write fails")

    calls = []
    monkeypatch.setattr("sys.stdout", BrokenStream())
    monkeypatch.setattr("hermes_cli.cli_session_mixin.shutil.which", lambda name: f"/bin/{name}")
    monkeypatch.setattr("hermes_cli.cli_session_mixin.windows_hide_flags", lambda: 123)

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("hermes_cli.cli_session_mixin.subprocess.run", fake_run)

    CLISessionMixin()._clear_terminal_on_exit()

    assert calls == [
        (
            ["/bin/clear"],
            {
                "check": False,
                "stdin": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "creationflags": 123,
            },
        )
    ]


def test_clear_terminal_fallback_skips_when_clear_binary_missing(monkeypatch):
    class BrokenStream:
        def isatty(self):
            return True

        def write(self, _text):
            raise OSError("ansi write failed")

        def flush(self):
            raise AssertionError("flush should not run after write fails")

    monkeypatch.setattr("sys.stdout", BrokenStream())
    monkeypatch.setattr("hermes_cli.cli_session_mixin.shutil.which", lambda _name: None)
    monkeypatch.setattr(
        "hermes_cli.cli_session_mixin.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("subprocess should not run")),
    )

    CLISessionMixin()._clear_terminal_on_exit()


def test_clear_terminal_windows_fallback_uses_cmd_builtin_without_shell(monkeypatch):
    class BrokenStream:
        def isatty(self):
            return True

        def write(self, _text):
            raise OSError("ansi write failed")

        def flush(self):
            raise AssertionError("flush should not run after write fails")

    calls = []
    monkeypatch.setattr("sys.stdout", BrokenStream())
    monkeypatch.setattr("hermes_cli.cli_session_mixin.os.name", "nt")
    monkeypatch.setenv("COMSPEC", r"C:\Windows\System32\cmd.exe")
    monkeypatch.setattr("hermes_cli.cli_session_mixin.windows_hide_flags", lambda: 123)

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("hermes_cli.cli_session_mixin.subprocess.run", fake_run)

    CLISessionMixin()._clear_terminal_on_exit()

    assert calls == [
        (
            [r"C:\Windows\System32\cmd.exe", "/d", "/c", "cls"],
            {
                "check": False,
                "stdin": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "creationflags": 123,
            },
        )
    ]
