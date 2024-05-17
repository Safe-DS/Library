# Tests

At the very least, we aim for 100% line coverage, so automated tests should be added for any new function. Additionally,
you should cover as many edge cases as possible, like empty data structures, invalid input, and so on.

## File structure

Tests belong in the [`tests`][tests-folder] folder. The file structure in the tests folder should mirror the file
structure of the [`src`][src-folder] folder.

## Naming

Names of test functions shall start with `test_should_` followed by a description of the expected behaviour.

!!! success "**DO** (library code):"

    ```py
    def test_should_raise_if_less_than_or_equal_to_0() -> None:
        ...
    ```

!!! failure "**DON'T** (library code):"

    ```py
    def test_value_error() -> None:
        ...
    ```

## Parametrization

Tests should be parametrized using `@pytest.mark.parametrize`, even if there is only a single test case. This makes it
easier to add new test cases in the future. Test cases should be given descriptive IDs.

!!! success "**DO** (library code):"

    ```py
    @pytest.mark.parametrize(
        "tree_count",
        [0, -1],
        ids=["zero", "negative"],
    )
    def test_should_raise_if_less_than_or_equal_to_0(tree_count: int) -> None:
        with pytest.raises(ValueError, match="The parameter 'tree_count' has to be greater than 0."):
            RandomForestRegressor(tree_count=tree_count)
    ```

!!! failure "**DON'T** (library code):"

    ```py
    def test_should_raise_if_equal_to_0(tree_count: int) -> None:
        with pytest.raises(ValueError, match="The parameter 'tree_count' has to be greater than 0."):
            RandomForestRegressor(tree_count=0)

    def test_should_raise_if_less_than_0(tree_count: int) -> None:
        with pytest.raises(ValueError, match="The parameter 'tree_count' has to be greater than 0."):
            RandomForestRegressor(tree_count=-1)
    ```

[src-folder]: https://github.com/Safe-DS/Library/tree/main/src

[tests-folder]: https://github.com/Safe-DS/Library/tree/main/tests
