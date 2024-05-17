# Docstrings

The docstrings **should** use the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) format. The
descriptions **should not** start with "this" and **should** use imperative mood. Docstrings **should not** contain type
hints, since they are already specified in the code. Refer to the subsections below for more details on how to document
specific API elements.

!!! success "**DO**:"

    ```py
    def add_ints(a: int, b: int) -> int:
        """Add two integers."""
        return a + b
    ```

!!! failure "**DON'T**:"

    ```py
    def add_ints(a: int, b: int) -> int:
        """This function adds two integers."""
        return a + b
    ```

!!! failure "**DON'T**:"

    ```py
    def add_ints(a: int, b: int) -> int:
        """Adds two integers."""
        return a + b
    ```

## Modules

All modules should have

* A one-line description ([short summary][short-summary-section]).
* A longer description if needed ([extended summary][extended-summary-section]).

Example:

```py
"""Containers for tabular data."""
```

## Classes

All public classes should have

* A one-line description ([short summary][short-summary-section]).
* A longer description if needed ([extended summary][extended-summary-section]).
* A description of the parameters of their `__init__` method ([`Parameters` section][parameters-section]). Specify a
  name and a description, with a colon to separate them. Omit types and default values.
* Examples that show how to use them correctly ([`Examples` section][examples-section]).

Example:

```py
"""
A row is a collection of named values.

Parameters
----------
data:
    The data. If None, an empty row is created.

Examples
--------
>>> from safeds.data.tabular.containers import Row
>>> row = Row({"a": 1, "b": 2})
"""
```

## Functions

All public functions should have

* A one-line description ([short summary][short-summary-section]).
* A longer description if needed ([extended summary][extended-summary-section]).
* A description of their parameters ([`Parameters` section][parameters-section]). Specify a name and a description, with
  a colon to separate them. Omit types and default values.
* A description of their results ([`Returns` section][returns-section]). Specify a name and a description, with a colon
  to separate them. Omit types.
* A description of any exceptions that may be raised and under which conditions that may
  happen ([`Raises` section][raises-section]).
* A description of any warnings that may be issued and under which conditions that may
  happen ([`Warns` section][warns-section]).
* Examples that show how to use them correctly ([`Examples` section][examples-section]).

Example:

```py
"""
Return the value of a specified column.

Parameters
----------
column_name:
    The column name.

Returns
-------
value:
    The column value.

Raises
------
UnknownColumnNameError
    If the row does not contain the specified column.

Examples
--------
>>> from safeds.data.tabular.containers import Row
>>> row = Row({"a": 1, "b": 2})
>>> row.get_value("a")
1
"""
```

[short-summary-section]: https://numpydoc.readthedocs.io/en/latest/format.html#short-summary

[extended-summary-section]: https://numpydoc.readthedocs.io/en/latest/format.html#extended-summary

[parameters-section]: https://numpydoc.readthedocs.io/en/latest/format.html#parameters

[returns-section]: https://numpydoc.readthedocs.io/en/latest/format.html#returns

[raises-section]: https://numpydoc.readthedocs.io/en/latest/format.html#raises

[warns-section]: https://numpydoc.readthedocs.io/en/latest/format.html#warns

[examples-section]: https://numpydoc.readthedocs.io/en/latest/format.html#examples
