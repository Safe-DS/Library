from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from safeds.ml.hyperparameters import Choice

if TYPE_CHECKING:
    from typing import Any


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
            (Choice(), []),
            (Choice(1, 2, 3), [1, 2, 3]),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_should_iterate_values(self, choice: Choice, expected: list[Any]) -> None:
        assert list(choice) == expected


class TestLen:
    @pytest.mark.parametrize(
        ("choice", "expected"),
        [
            (Choice(), 0),
            (Choice(1, 2, 3), 3),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_should_return_number_of_values(self, choice: Choice, expected: int) -> None:
        assert len(choice) == expected
