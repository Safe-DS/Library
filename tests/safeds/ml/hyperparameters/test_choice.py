from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from safeds.exceptions import EmptyChoiceError
from safeds.ml.hyperparameters import Choice

if TYPE_CHECKING:
    from typing import Any


class TestInit:
    def test_should_iterate_values(self) -> None:
        with pytest.raises(EmptyChoiceError):
            Choice()


class TestContains:
    @pytest.mark.parametrize(
        ("choice", "value", "expected"),
        [
            (Choice(1, 2, 3), 1, True),
            (Choice(1, 2, 3), 2, True),
            (Choice(1, 2, 3), 3, True),
            (Choice(1, 2, 3), 4, False),
            (Choice(1, 2, 3), "3", False),
        ],
        ids=[
            "value in choice (start)",
            "value in choice (middle)",
            "value in choice (end)",
            "value not in choice",
            "value not in choice (wrong type)",
        ],
    )
    def test_should_check_whether_choice_contains_value(self, choice: Choice, value: Any, expected: bool) -> None:
        assert (value in choice) == expected


class TestIter:
    @pytest.mark.parametrize(
        ("choice", "expected"),
        [
            (Choice(1, 2, 3), [1, 2, 3]),
        ],
        ids=[
            "non-empty",
        ],
    )
    def test_should_iterate_values(self, choice: Choice, expected: list[Any]) -> None:
        assert list(choice) == expected


class TestLen:
    @pytest.mark.parametrize(
        ("choice", "expected"),
        [
            (Choice(1, 2, 3), 3),
        ],
        ids=[
            "non-empty",
        ],
    )
    def test_should_return_number_of_values(self, choice: Choice, expected: int) -> None:
        assert len(choice) == expected


class TestEq:
    @pytest.mark.parametrize(
        ("choice1", "choice2", "equal"),
        [
            (
                Choice(1),
                Choice(1),
                True,
            ),
            (
                Choice(1),
                Choice(2),
                False,
            ),
            (
                Choice(1, 2, 3),
                Choice(1, 2, 3),
                True,
            ),
        ],
        ids=["equal", "not_equal", "equal with multiple values"],
    )
    def test_should_compare_choices(self, choice1: Choice[int], choice2: Choice[int], equal: bool) -> None:
        assert (choice1.__eq__(choice2)) == equal
