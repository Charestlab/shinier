import builtins

import pytest

from shinier.SHINIER import prompt

pytestmark = pytest.mark.unit_tests


def test_prompt_bool_retries_after_invalid_input(monkeypatch, capsys, recwarn):
    """Invalid bool input should repeat the same prompt and preserve the default."""
    answers = iter(["¥", "n"])
    monkeypatch.setattr(builtins, "input", lambda: next(answers))

    value = prompt("Safe luminance matching?", default="n", kind="bool")

    captured = capsys.readouterr()
    assert value is False
    assert "Invalid input." in captured.out
    assert captured.out.count("Safe luminance matching?") == 2
    assert len(recwarn) == 0


def test_prompt_bool_default_returns_boolean(monkeypatch):
    """Pressing Enter for a bool prompt should return True/False, not a choice index."""
    monkeypatch.setattr(builtins, "input", lambda: "")

    value = prompt("Safe luminance matching?", default="n", kind="bool")

    assert value is False


def test_prompt_bool_yes_default_returns_boolean(monkeypatch):
    """Bool defaults should preserve boolean semantics for yes defaults too."""
    monkeypatch.setattr(builtins, "input", lambda: "")

    value = prompt("Histogram optimization?", default="y", kind="bool")

    assert value is True


def test_prompt_choice_retries_after_invalid_input(monkeypatch, capsys):
    """Invalid choice input should repeat the prompt with the same default."""
    answers = iter(["bad", "2"])
    monkeypatch.setattr(builtins, "input", lambda: next(answers))

    value = prompt("Processing mode", default=1, kind="choice", choices=["A", "B"])

    captured = capsys.readouterr()
    assert value == 2
    assert "Invalid input." in captured.out
    assert captured.out.count("Processing mode") == 2


def test_prompt_choice_default_returns_index(monkeypatch):
    """Choice defaults should keep their existing index-based return value."""
    monkeypatch.setattr(builtins, "input", lambda: "")

    value = prompt("Processing mode", default=2, kind="choice", choices=["A", "B"])

    assert value == 2
