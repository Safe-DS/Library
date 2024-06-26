# Design

## Prefer a usable API to simple implementation

It's more important to provide a user-friendly API to many people than to save some of our time when implementing the
functionality.

## Prefer named functions to overloaded operators

The names can better convey the intention of the programmer and enable better auto-completion.

!!! success "**DO** (client code):"

    ```py
    table.keep_columns("name", "age")
    ```

!!! failure "**DON'T** (client code):"

    ```py
    table[["name", "age"]]
    ```

## Prefer methods to global functions

This aids discoverability and again enables better auto-completion. It also supports polymorphism.

!!! success "**DO** (client code):"

    ```py
    model.fit(training_data)
    ```

!!! failure "**DON'T** (client code):"

    ```py
    fit(model, training_data)
    ```

## Prefer separate functions to functions with a flag parameter

Some flag parameters drastically alter the semantics of a function. This can lead to confusion, and, if the parameter is
optional, to errors if the default value is kept unknowingly. In such cases having two separate functions is preferable.

!!! success "**DO** (client code):"

    ```py
    table.drop_columns("name")
    ```

!!! failure "**DON'T** (client code):"

    ```py
    table.drop("name", axis="columns")
    ```

## Return copies of objects

Modifying objects in-place can lead to surprising behaviour and hard-to-find bugs. Methods shall never change the object
they're called on or any of their parameters.

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

The corresponding docstring should explicitly state that a method returns a copy:

!!! success "**DO** (library code):"

    ```py
    """
    Return a new table with the given column added as the last column.

    **Note:** The original table is not modified.
    """
    ```

## Avoid uncommon abbreviations

Write full words rather than abbreviations. The increased verbosity is offset by better readability, better functioning
auto-completion, and a reduced need to consult the documentation when writing code. Common abbreviations like CSV or
HTML, and well-known mathematical terms like min or max are fine though, since they rarely require explanation.

!!! success "**DO** (client code):"

    ```py
    figure.set_color_scheme(ColorScheme.AUTUMN)
    ```

!!! failure "**DON'T** (client code):"

    ```py
    figure.scs(CS.AUT)
    ```

## Place more important parameters first

Parameters that are more important to the user should be placed first. This also applies
to [keyword-only parameters](#consider-marking-optional-parameters-as-keyword-only), since they still have a fixed order
in the documentation. In particular, parameters of model constructors should have the following order:

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
            tree_count: int = 100,
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
            tree_count: int = 100,
            learning_rate: float = 0.1,
        ) -> None:
            ...
    ```

## Consider marking optional parameters as keyword-only

_Keyword-only parameters_ are parameters that can only be passed by name. It prevents users from accidentally passing a
value to the wrong parameter. This can happen easily if several parameters have the same type. Moreover, marking a
parameter as keyword-only allows us to change the order of parameters without breaking client code. Because of this,
strongly consider marking optional parameters as keyword-only. In particular, optional hyperparameters of models should
be keyword-only.

!!! success "**DO** (library code):"

    ```py
    class RandomForest:
        def __init__(self, *, tree_count: int = 100) -> None:
            ...
    ```

!!! failure "**DON'T** (library code):"

    ```py
    class RandomForest:
        def __init__(self, tree_count: int = 100) -> None:
            ...
    ```

## Specify types of parameters and results

Use [type hints](https://docs.python.org/3/library/typing.html) to describe the types of parameters and results of
functions. This enables static type checking of client code.

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

## Use narrow data types

Use data types that can accurately model the legal values of a declaration. This improves static detection of wrong
client code.

!!! success "**DO** (client code):"

    ```py
    SupportVectorMachine(kernel=Kernel.LINEAR) # (enum)
    ```

!!! failure "**DON'T** (client code):"

    ```py
    SupportVectorMachine(kernel="linear") # (string)
    ```

## Check preconditions of functions and fail early

Not all preconditions of functions can be described with type hints but must instead be checked at runtime. This should
be done as early as possible, usually right at the top of the body of a function. If the preconditions fail, execution
of the function should halt and either a sensible value be returned (if possible) or an exception with a descriptive
message be raised.

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

## Raise either Python exceptions or custom exceptions

The user should not have to deal with exceptions that are defined in the wrapper libraries. So, any exceptions that may
be raised when a third-party function is called should be caught and a core Python exception or a custom exception
should be raised instead. The exception to this rule is when we call a callable created by the user: In this case, we
just pass any exceptions thrown by this callable along.

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

## Group API elements by task

Packages should correspond to a specific task like classification or imputation. This eases discovery and makes it easy
to switch between different solutions for the same task.

!!! success "**DO** (client code):"

    ```py
    from sklearn.classification import SupportVectorMachine
    ```

!!! failure "**DON'T** (client code):"

    ```py
    from sklearn.svm import SupportVectorMachine
    ```

## Group values that are used together into an object

Passing values that are commonly used together around separately is tedious, verbose, and error-prone. Group them into
an object instead.

!!! success "**DO** (client code):"

    ```py
    training_data, validation_data = split_rows(full_data)
    ```

!!! failure "**DON'T** (client code):"

    ```py
    training_feature_vectors, validation_feature_vectors, training_target_values, validation_target_values = split_rows(feature_vectors, target_values)
    ```
