import pytest
from safeds.data.tabular.containers import TaggedTable


@pytest.mark.parametrize(
    "tagged_table",
    [
        TaggedTable({"a": [], "b": []}, target_name="b", feature_names=["a"]),
        TaggedTable({"a": ["a", 3, 0.1], "b": [True, False, None]}, target_name="b", feature_names=["a"]),
        TaggedTable(
            {"a": ["a", 3, 0.1], "b": [True, False, None], "c": ["a", "b", "c"]},
            target_name="b",
            feature_names=["a"],
        ),
        TaggedTable({"a": [], "b": [], "c": []}, target_name="b", feature_names=["a"]),
    ],
    ids=["empty-rows", "normal", "column_as_non_feature", "column_as_non_feature_with_empty_rows"],
)
def test_should_copy_tagged_table(tagged_table: TaggedTable) -> None:
    copied = tagged_table._copy()
    assert copied == tagged_table
    assert copied is not tagged_table
