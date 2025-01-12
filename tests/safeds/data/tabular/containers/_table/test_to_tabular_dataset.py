import pytest

from safeds.data.tabular.containers import Table
from safeds.exceptions import ColumnNotFoundError


@pytest.mark.parametrize(
    ("table", "target_name", "extra_names", "expected_extra_names", "expected_feature_names"),
    [
        (
            Table({"a": [1, 2], "b": [3, 4], "c": [5, 6], "d": [7, 8]}),
            "a",
            None,
            [],
            ["b", "c", "d"],
        ),
        (
            Table({"a": [1, 2], "b": [3, 4], "c": [5, 6], "d": [7, 8]}),
            "a",
            [],
            [],
            ["b", "c", "d"],
        ),
        (
            Table({"a": [1, 2], "b": [3, 4], "c": [5, 6], "d": [7, 8]}),
            "a",
            "b",
            ["b"],
            ["c", "d"],
        ),
        (
            Table({"a": [1, 2], "b": [3, 4], "c": [5, 6], "d": [7, 8]}),
            "a",
            ["b", "c"],
            ["b", "c"],
            ["d"],
        ),
    ],
    ids=[
        "no extras, implicit",
        "no extras, explicit",
        "one extra",
        "multiple extras",
    ],
)
class TestHappyPath:
    def test_should_have_correct_targets(
        self,
        table: Table,
        target_name: str,
        extra_names: str | list[str] | None,
        expected_extra_names: list[str],  # noqa: ARG002
        expected_feature_names: list[str],  # noqa: ARG002
    ) -> None:
        actual = table.to_tabular_dataset(target_name, extra_names=extra_names)
        assert actual.target.name == target_name

    def test_should_have_correct_extras(
        self,
        table: Table,
        target_name: str,
        extra_names: str | list[str] | None,
        expected_extra_names: list[str],
        expected_feature_names: list[str],  # noqa: ARG002
    ) -> None:
        actual = table.to_tabular_dataset(target_name, extra_names=extra_names)
        assert actual.extras.column_names == expected_extra_names

    def test_should_have_correct_features(
        self,
        table: Table,
        target_name: str,
        extra_names: str | list[str] | None,
        expected_extra_names: list[str],  # noqa: ARG002
        expected_feature_names: list[str],
    ) -> None:
        actual = table.to_tabular_dataset(target_name, extra_names=extra_names)
        assert actual.features.column_names == expected_feature_names

    def test_should_have_correct_data(
        self,
        table: Table,
        target_name: str,
        extra_names: str | list[str] | None,
        expected_extra_names: list[str],  # noqa: ARG002
        expected_feature_names: list[str],  # noqa: ARG002
    ) -> None:
        actual = table.to_tabular_dataset(target_name, extra_names=extra_names)
        assert actual.to_table() == table


@pytest.mark.parametrize(
    ("table", "target_name", "extra_names"),
    [
        (
            Table({"a": [], "b": []}),
            "unknown",
            None,
        ),
        (
            Table({"a": [], "b": []}),
            "a",
            "unknown",
        ),
    ],
    ids=[
        "unknown target",
        "unknown extra",
    ],
)
def test_should_raise_if_column_not_found(
    table: Table,
    target_name: str,
    extra_names: str | list[str] | None,
) -> None:
    with pytest.raises(ColumnNotFoundError):
        table.to_tabular_dataset(target_name, extra_names=extra_names)


def test_should_raise_if_target_is_extra() -> None:
    table = Table({"a": [], "b": []})
    with pytest.raises(ValueError, match=r"Column 'a' cannot be both target and extra column\."):
        table.to_tabular_dataset("a", extra_names="a")


@pytest.mark.parametrize(
    ("table", "target_name", "extra_names"),
    [
        (
            Table({"a": []}),
            "a",
            None,
        ),
        (
            Table({"a": [], "b": []}),
            "a",
            "b",
        ),
    ],
    ids=[
        "without extras",
        "with extras",
    ],
)
def test_should_raise_if_no_feature_columns_remain(
    table: Table,
    target_name: str,
    extra_names: str | list[str] | None,
) -> None:
    with pytest.raises(ValueError, match=r"At least one feature column must remain\."):
        table.to_tabular_dataset(target_name, extra_names=extra_names)
