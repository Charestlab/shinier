import builtins

import pytest

from shinier.SHINIER import prompt

pytestmark = pytest.mark.unit_tests


def _mock_input(monkeypatch, answers):
    answers = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda: next(answers))


def _assert_invalid_retries(monkeypatch, capsys, label, answers, expected, **kwargs):
    _mock_input(monkeypatch, answers)
    value = prompt(label, **kwargs)
    captured = capsys.readouterr()

    assert value == expected
    assert "Invalid input." in captured.out
    assert captured.out.count(label) == 2


@pytest.mark.parametrize(
    "default, expected",
    [("n", False), ("y", True)],
)
def test_prompt_bool_default_returns_boolean(monkeypatch, default, expected):
    """Pressing Enter for a bool prompt should return True/False."""
    _mock_input(monkeypatch, [""])

    assert prompt("Safe luminance matching?", default=default, kind="bool") is expected


def test_prompt_bool_retries_after_invalid_input(monkeypatch, capsys, recwarn):
    """Invalid bool input should re-prompt without warning."""
    _assert_invalid_retries(
        monkeypatch, capsys, "Safe luminance matching?", ["¥", "n"],
        False, default="n", kind="bool"
    )
    assert len(recwarn) == 0


def test_prompt_choice_default_returns_index(monkeypatch):
    """Choice defaults should keep their existing index-based return value."""
    _mock_input(monkeypatch, [""])

    assert prompt("Processing mode", default=2, kind="choice", choices=["A", "B"]) == 2


@pytest.mark.parametrize(
    "answers, expected",
    [(["bad", "2"], 2), (["99", "1"], 1)],
)
def test_prompt_choice_invalid_input_retries(monkeypatch, capsys, answers, expected):
    """Invalid choice input should re-prompt, not quit."""
    _assert_invalid_retries(
        monkeypatch, capsys, "Processing mode", answers, expected,
        default=1, kind="choice", choices=["A", "B"]
    )


def test_prompt_int_enter_returns_default(monkeypatch):
    """Enter on an int prompt should return the default integer."""
    _mock_input(monkeypatch, [""])

    value = prompt("Kernel size", default=3, kind="int")

    assert value == 3
    assert isinstance(value, int)


@pytest.mark.parametrize(
    "answers, expected, kwargs",
    [
        (["abc", "7"], 7, {}),
        (["200", "5"], 5, {"min_v": 1, "max_v": 10}),
    ],
)
def test_prompt_int_invalid_input_retries(monkeypatch, capsys, answers, expected, kwargs):
    """Invalid int input should re-prompt, not quit."""
    _assert_invalid_retries(
        monkeypatch, capsys, "Kernel size", answers, expected,
        default=3, kind="int", **kwargs
    )


def test_prompt_float_enter_returns_default(monkeypatch):
    """Enter on a float prompt should return the default float."""
    _mock_input(monkeypatch, [""])

    value = prompt("Learning rate", default=0.5, kind="float")

    assert value == 0.5
    assert isinstance(value, float)


@pytest.mark.parametrize(
    "label, answers, expected, kwargs",
    [
        ("Learning rate", ["notanumber", "1.2"], 1.2, {}),
        ("Clip threshold", ["-1.0", "0.8"], 0.8, {"min_v": 0.0, "max_v": 1.0}),
    ],
)
def test_prompt_float_invalid_input_retries(monkeypatch, capsys, label, answers, expected, kwargs):
    """Invalid float input should re-prompt, not quit."""
    _assert_invalid_retries(
        monkeypatch, capsys, label, answers, pytest.approx(expected),
        default=0.5, kind="float", **kwargs
    )


def test_prompt_str_enter_returns_default(monkeypatch):
    """Enter on a str prompt should return the default string."""
    _mock_input(monkeypatch, [""])

    assert prompt("Output name", default="result") == "result"


def test_prompt_tuple_enter_returns_default(monkeypatch):
    """Enter on a tuple prompt should return the default tuple."""
    _mock_input(monkeypatch, [""])

    assert prompt("Range", default=(0.0, 1.0), kind="tuple") == (0.0, 1.0)


def test_prompt_tuple_invalid_retries(monkeypatch, capsys):
    """Malformed tuple input should re-prompt, not quit."""
    _assert_invalid_retries(
        monkeypatch, capsys, "Range", ["notvalid,,", "0.0, 1.0"],
        (0.0, 1.0), default=(0.0, 1.0), kind="tuple"
    )


def test_prompt_validator_failure_retries(monkeypatch, capsys):
    """A failing validator should re-prompt instead of returning the bad value."""
    def validator(v):
        if v == "bad_path":
            return False, "Path does not exist"
        return True, ""

    _mock_input(monkeypatch, ["bad_path", "good_path"])
    value = prompt("Input path", default="default_path", validator=validator)
    captured = capsys.readouterr()

    assert value == "good_path"
    assert "Path does not exist" in captured.out
    assert captured.out.count("Input path") == 2
