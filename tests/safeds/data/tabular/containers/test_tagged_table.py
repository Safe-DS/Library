import pytest
from safeds.data.tabular.containers import TaggedTable


class TestCopy:
    @pytest.mark.parametrize(
        "tagged_table",
        [
            TaggedTable({"a": [], "b": []}, target_name="b", feature_names=["a"]),
            TaggedTable({"a": ["a", 3, 0.1], "b": [True, False, None]}, target_name="b", feature_names=["a"]),
        ],
        ids=["empty-rows", "normal"],
    )
    def test_should_copy_tagged_table(self, tagged_table: TaggedTable) -> None:
        copied = tagged_table._copy()
        assert copied == tagged_table
        assert copied is not tagged_table
