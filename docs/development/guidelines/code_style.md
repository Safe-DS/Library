# Code style

## Consistency

If there is more than one way to solve a particular task, check how it has been solved elsewhere in the codebase and
stick to that solution.

## Sort exported classes in `__init__.py`

Classes defined in a module that other classes shall be able to import must be defined in a list named `__all__` in the
module's `__init__.py` file. This list should be sorted in the same order that the declarations are imported in the
file. This reduces the likelihood of merge conflicts when adding new classes to it.

!!! success "**DO** (library code):"

    ```py
    __all__ = [
        "Column",
        "Row",
        "Table",
    ]
    ```

!!! failure "**DON'T** (library code):"

    ```py
    __all__ = [
        "Table",
        "Column",
        "Row",
    ]
    ```
