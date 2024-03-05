# Project Guidelines

This document describes general guidelines for the Safe-DS Python Library. In the **DO**/**DON'T** examples below we either show _client code_ to describe the code users should/shouldn't have to write, or _library code_ to describe the code we, as library developers, need to write to achieve readable client code. We'll continuously update this document as we find new categories of usability issues.

## Design

### Prefer a usable API to simple implementation

It's more important to provide a user-friendly API to many people than to save some of our time when implementing the functionality.

### Prefer named functions to overloaded operators

The names can better convey the intention of the programmer and enable better auto-completion.

!!! success "**DO** (client code):"

    ```py
    table.keep_columns("name", "age")
    ```

!!! failure "**DON'T** (client code):"

    ```py
    table[["name", "age"]]
    ```

### Prefer methods to global functions

This aids discoverability and again enables better auto-completion. It also supports polymorphism.

!!! success "**DO** (client code):"

    ```py
    model.fit(training_data)
    ```

!!! failure "**DON'T** (client code):"

    ```py
    fit(model, training_data)
    ```

### Prefer separate functions to functions with a flag parameter

Some flag parameters drastically alter the semantics of a function. This can lead to confusion, and, if the parameter is optional, to errors if the default value is kept unknowingly. In such cases having two separate functions is preferable.

!!! success "**DO** (client code):"

    ```py
    table.drop_columns("name")
    ```

!!! failure "**DON'T** (client code):"

    ```py
    table.drop("name", axis="columns")
    ```

### Return copies of objects

Modifying objects in-place
can lead to surprising behaviour
and hard-to-find bugs.
Methods shall never change
the object they're called on
or any of their parameters.

!!! success "**DO** (library code):"

    ```py
        result = self._data.copy()
        result.columns = self._schema.column_names
        result[new_column.name] = new_column._data
        return Table._from_pandas_dataframe(result)
    ```

!!! failure "**DON'T** (library code):"

    ```py
        self._data.add(new_column, axis='column')
    ```

The corresponding docstring should explicitly state
that a method returns a copy:

!!! success "**DO** (library code):"

    ```py
    """
    Return a new table with the given column added as the last column.
    The original table is not modified.
    ...
    """
    ```

### Avoid uncommon abbreviations

Write full words rather than abbreviations. The increased verbosity is offset by better readability, better functioning auto-completion, and a reduced need to consult the documentation when writing code. Common abbreviations like CSV or HTML are fine though, since they rarely require explanation.

!!! success "**DO** (client code):"

    ```py
    figure.set_color_scheme(ColorScheme.AUTUMN)
    ```

!!! failure "**DON'T** (client code):"

    ```py
    figure.scs(CS.AUT)
    ```

### Place more important parameters first

Parameters that are more important to the user should be placed first. This also applies to [keyword-only parameters](#consider-marking-optional-parameters-as-keyword-only), since they still have a fixed order in the documentation. In particular, parameters of model constructors should have the following order:

1. Model hyperparameters (e.g., the number of trees in a random forest)
2. Algorithm hyperparameters (e.g., the learning rate of a gradient boosting algorithm)
3. Regularization hyperparameters (e.g., the maximum depth of a decision tree)
4. Other parameters (e.g., the random seed)

!!! success "**DO** (library code):"

    ```py
    class GradientBoosting:
        def __init__(
            self,
            *,
            number_of_trees: int = 100,
            learning_rate: float = 0.1,
            random_seed: int = 0,
        ) -> None:
            ...
    ```

!!! failure "**DON'T** (library code):"

    ```py
    class GradientBoosting:
        def __init__(
            self,
            *,
            random_seed: int = 0,
            number_of_trees: int = 100,
            learning_rate: float = 0.1,
        ) -> None:
            ...
    ```

### Consider marking optional parameters as keyword-only

_Keyword-only parameters_ are parameters that can only be passed by name. It prevents users from accidentally passing a value to the wrong parameter. This can happen easily if several parameters have the same type. Moreover, marking a parameter as keyword-only allows us to change the order of parameters without breaking client code. Because of this, strongly consider marking optional parameters as keyword-only. In particular, optional hyperparameters of models should be keyword-only.

!!! success "**DO** (library code):"

    ```py
    class RandomForest:
        def __init__(self, *, number_of_trees: int = 100) -> None:
            ...
    ```

!!! failure "**DON'T** (library code):"

    ```py
    class RandomForest:
        def __init__(self, number_of_trees: int = 100) -> None:
            ...
    ```

### Specify types of parameters and results

Use [type hints](https://docs.python.org/3/library/typing.html) to describe the types of parameters and results of functions. This enables static type checking of client code.

!!! success "**DO** (library code):"

    ```py
    def add_ints(a: int, b: int) -> int:
        return a + b
    ```

!!! failure "**DON'T** (library code):"

    ```py
    def add_ints(a, b):
        return a + b
    ```

### Use narrow data types

Use data types that can accurately model the legal values of a declaration. This improves static detection of wrong client code.

!!! success "**DO** (client code):"

    ```py
    SupportVectorMachine(kernel=Kernel.LINEAR) # (enum)
    ```

!!! failure "**DON'T** (client code):"

    ```py
    SupportVectorMachine(kernel="linear") # (string)
    ```

### Check preconditions of functions and fail early

Not all preconditions of functions can be described with type hints but must instead be checked at runtime. This should be done as early as possible, usually right at the top of the body of a function.  If the preconditions fail, execution of the function should halt and either a sensible value be returned (if possible) or an exception with a descriptive message be raised.

!!! success "**DO** (library code):"

    ```py
    def nth_prime(n: int) -> int:
        if n <= 0:
            raise ValueError(f"n must be at least 1 but was {n}.")

        # compute nth prime
    ```

!!! failure "**DON'T** (library code):"

    ```py
    def nth_prime(n: int) -> int:
        # compute nth prime
    ```

### Raise either Python exceptions or custom exceptions

The user should not have to deal with exceptions that are defined in the wrapper libraries. So, any exceptions that may be raised when a third-party function is called should be caught and a core Python exception or a custom exception should be raised instead. The exception to this rule is when we call a callable created by the user: In this case, we just pass any exceptions thrown by this callable along.

!!! success "**DO** (library code):"

    ```py
    def read_csv(path: str) -> Table:
        try:
            return pd.read_csv(path) # May raise a pd.ParserError
        except pd.ParserError as e:
            raise FileFormatException("The loaded file is not a CSV file.") from e
    ```

!!! failure "**DON'T** (library code):"

    ```py
    def read_csv(path: str) -> Table:
        return pd.read_csv(path) # May raise a pd.ParserError
    ```

### Group API elements by task

Packages should correspond to a specific task like classification or imputation. This eases discovery and makes it easy to switch between different solutions for the same task.

!!! success "**DO** (client code):"

    ```py
    from sklearn.classification import SupportVectorMachine
    ```

!!! failure "**DON'T** (client code):"

    ```py
    from sklearn.svm import SupportVectorMachine
    ```

### Group values that are used together into an object

Passing values that are commonly used together around separately is tedious, verbose, and error prone.

!!! success "**DO** (client code):"

    ```py
    training_data, validation_data = split_rows(full_data)
    ```

!!! failure "**DON'T** (client code):"

    ```py
    training_feature_vectors, validation_feature_vectors, training_target_values, validation_target_values = split_rows(feature_vectors, target_values)
    ```

## Docstrings

The docstrings **should** use the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) format. The
descriptions **should not** start with "this" and **should** use imperative mood. Docstrings **should not** contain
type hints, since they are already specified in the code. Refer to the subsections below for more details on how to
document specific API elements.

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

### Modules

All modules should have

* A one-line description ([short summary][short-summary-section]).
* A longer description if needed ([extended summary][extended-summary-section]).

Example:

```py
"""Containers for tabular data."""
```

### Classes

All classes should have

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

### Functions

All functions should have

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

## Tests

We aim for 100% line coverage,
so automated tests should be added
for any new function.

### File structure

Tests belong in the [`tests`][tests-folder] folder.
The file structure in the tests folder
should mirror the file structure
of the [`src`][src-folder] folder.

### Naming

Names of test functions
shall start with `test_should_`
followed by a description
of the expected behaviour,
e.g. `test_should_add_column`.

!!! success "**DO** (library code):"

    ```py
    def test_should_raise_if_less_than_or_equal_to_0(self, number_of_trees) -> None:
        with pytest.raises(ValueError, match="The parameter 'number_of_trees' has to be greater than 0."):
            ...
    ```

!!! failure "**DON'T** (library code):"

    ```py
    def test_value_error(self, number_of_trees) -> None:
        with pytest.raises(ValueError, match="The parameter 'number_of_trees' has to be greater than 0."):
            ...
    ```

### Parametrization

Tests should be parametrized
using `@pytest.mark.parametrize`,
even if there is only a single test case.
This makes it easier
to add new test cases in the future.
Test cases should be given
descriptive IDs.

!!! success "**DO** (library code):"

    ```py
    @pytest.mark.parametrize("number_of_trees", [0, -1], ids=["zero", "negative"])
    def test_should_raise_if_less_than_or_equal_to_0(self, number_of_trees) -> None:
        with pytest.raises(ValueError, match="The parameter 'number_of_trees' has to be greater than 0."):
            RandomForest(number_of_trees=number_of_trees)
    ```

!!! failure "**DON'T** (library code):"

    ```py
    def test_should_raise_if_less_than_0(self, number_of_trees) -> None:
        with pytest.raises(ValueError, match="The parameter 'number_of_trees' has to be greater than 0."):
            RandomForest(number_of_trees=-1)

    def test_should_raise_if_equal_to_0(self, number_of_trees) -> None:
        with pytest.raises(ValueError, match="The parameter 'number_of_trees' has to be greater than 0."):
            RandomForest(number_of_trees=0)
    ```

## Code style

### Consistency

If there is more than one way
to solve a particular task,
check how it has been solved
at other places in the codebase
and stick to that solution.

### Sort exported classes in `__init__.py`

Classes defined in a module
that other classes shall be able to import
must be defined
in a list named `__all__`
in the module's `__init__.py` file.
This list should be sorted alphabetically,
to reduce the likelihood of merge conflicts
when adding new classes to it.

!!! success "**DO** (library code):"

    ```py
    __all__ = [
        "Column",
        "Row",
        "Table",
        "TaggedTable",
    ]
    ```

!!! failure "**DON'T** (library code):"

    ```py
    __all__ = [
        "Table",
        "TaggedTable",
        "Column",
        "Row",
    ]
    ```

[src-folder]: https://github.com/Safe-DS/Library/tree/main/src
[tests-folder]: https://github.com/Safe-DS/Library/tree/main/tests

[short-summary-section]: https://numpydoc.readthedocs.io/en/latest/format.html#short-summary
[extended-summary-section]: https://numpydoc.readthedocs.io/en/latest/format.html#extended-summary
[parameters-section]: https://numpydoc.readthedocs.io/en/latest/format.html#parameters
[returns-section]: https://numpydoc.readthedocs.io/en/latest/format.html#returns
[raises-section]: https://numpydoc.readthedocs.io/en/latest/format.html#raises
[warns-section]: https://numpydoc.readthedocs.io/en/latest/format.html#warns
[examples-section]: https://numpydoc.readthedocs.io/en/latest/format.html#examples
