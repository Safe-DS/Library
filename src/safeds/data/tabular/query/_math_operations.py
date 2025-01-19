from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Cell


class MathOperations(ABC):
    """
    Namespace for mathematical operations.

    This class cannot be instantiated directly. It can only be accessed using the `math` attribute of a cell.

    Examples
    --------
    >>> from safeds.data.tabular.containers import Column
    >>> column = Column("a", [-1, 0, 1])
    >>> column.transform(lambda cell: cell.math.abs())
    +-----+
    |   a |
    | --- |
    | i64 |
    +=====+
    |   1 |
    |   0 |
    |   1 |
    +-----+
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def __eq__(self, other: object) -> bool: ...

    @abstractmethod
    def __hash__(self) -> int: ...

    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def __sizeof__(self) -> int: ...

    @abstractmethod
    def __str__(self) -> str: ...

    # ------------------------------------------------------------------------------------------------------------------
    # Math operations
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def abs(self) -> Cell:
        """
        Get the absolute value.

        Returns
        -------
        cell:
            The absolute value.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, -2, None])
        >>> column.transform(lambda cell: cell.math.abs())
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |    1 |
        |    2 |
        | null |
        +------+
        """

    @abstractmethod
    def arccos(self) -> Cell:
        """
        Get the inverse cosine.

        Returns
        -------
        cell:
            The inverse cosine.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [-1, 0, 1, None])
        >>> column.transform(lambda cell: cell.math.arccos())
        +---------+
        |       a |
        |     --- |
        |     f64 |
        +=========+
        | 3.14159 |
        | 1.57080 |
        | 0.00000 |
        |    null |
        +---------+
        """

    @abstractmethod
    def arccosh(self) -> Cell:
        """
        Get the inverse hyperbolic cosine.

        Returns
        -------
        cell:
            The inverse hyperbolic cosine.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [-1, 0, 1, None])
        >>> column.transform(lambda cell: cell.math.arccosh())
        +---------+
        |       a |
        |     --- |
        |     f64 |
        +=========+
        |     NaN |
        |     NaN |
        | 0.00000 |
        |    null |
        +---------+
        """

    @abstractmethod
    def arcsin(self) -> Cell:
        """
        Get the inverse sine.

        Returns
        -------
        cell:
            The inverse sine.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [-1, 0, 1, None])
        >>> column.transform(lambda cell: cell.math.arcsin())
        +----------+
        |        a |
        |      --- |
        |      f64 |
        +==========+
        | -1.57080 |
        |  0.00000 |
        |  1.57080 |
        |     null |
        +----------+
        """

    @abstractmethod
    def arcsinh(self) -> Cell:
        """
        Get the inverse hyperbolic sine.

        Returns
        -------
        cell:
            The inverse hyperbolic sine.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [-1, 0, 1, None])
        >>> column.transform(lambda cell: cell.math.arcsinh())
        +----------+
        |        a |
        |      --- |
        |      f64 |
        +==========+
        | -0.88137 |
        |  0.00000 |
        |  0.88137 |
        |     null |
        +----------+
        """

    @abstractmethod
    def arctan(self) -> Cell:
        """
        Get the inverse tangent.

        Returns
        -------
        cell:
            The inverse tangent.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [-1, 0, 1, None])
        >>> column.transform(lambda cell: cell.math.arctan())
        +----------+
        |        a |
        |      --- |
        |      f64 |
        +==========+
        | -0.78540 |
        |  0.00000 |
        |  0.78540 |
        |     null |
        +----------+
        """

    @abstractmethod
    def arctanh(self) -> Cell:
        """
        Get the inverse hyperbolic tangent.

        Returns
        -------
        cell:
            The inverse hyperbolic tangent.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [-1, 0, 1, None])
        >>> column.transform(lambda cell: cell.math.arctanh())
        +---------+
        |       a |
        |     --- |
        |     f64 |
        +=========+
        |    -inf |
        | 0.00000 |
        |     inf |
        |    null |
        +---------+
        """

    @abstractmethod
    def ceil(self) -> Cell:
        """
        Round up to the nearest integer.

        Returns
        -------
        cell:
            The rounded value.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1.1, 3.0, None])
        >>> column.transform(lambda cell: cell.math.ceil())
        +---------+
        |       a |
        |     --- |
        |     f64 |
        +=========+
        | 2.00000 |
        | 3.00000 |
        |    null |
        +---------+
        """

    @abstractmethod
    def cos(self) -> Cell:
        """
        Get the cosine.

        Returns
        -------
        cell:
            The cosine.

        Examples
        --------
        >>> import math
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [0, math.pi / 2, math.pi, 3 * math.pi / 2, None])
        >>> column.transform(lambda cell: cell.math.cos())
        +----------+
        |        a |
        |      --- |
        |      f64 |
        +==========+
        |  1.00000 |
        |  0.00000 |
        | -1.00000 |
        | -0.00000 |
        |     null |
        +----------+
        """

    @abstractmethod
    def cosh(self) -> Cell:
        """
        Get the hyperbolic cosine.

        Returns
        -------
        cell:
            The hyperbolic cosine.

        Examples
        --------
        >>> import math
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [-1, 0, 1, None])
        >>> column.transform(lambda cell: cell.math.cosh())
        +---------+
        |       a |
        |     --- |
        |     f64 |
        +=========+
        | 1.54308 |
        | 1.00000 |
        | 1.54308 |
        |    null |
        +---------+
        """

    @abstractmethod
    def floor(self) -> Cell:
        """
        Round down to the nearest integer.

        Returns
        -------
        cell:
            The rounded value.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1.1, 3.0, None])
        >>> column.transform(lambda cell: cell.math.floor())
        +---------+
        |       a |
        |     --- |
        |     f64 |
        +=========+
        | 1.00000 |
        | 3.00000 |
        |    null |
        +---------+
        """

    @abstractmethod
    def round_to_decimal_places(self, decimal_places: int) -> Cell:
        """
        Round to the specified number of decimal places.

        Parameters
        ----------
        decimal_places:
            The number of decimal places to round to.

        Returns
        -------
        cell:
            The rounded value.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [0.999, 1.123, 3.456, None])
        >>> column.transform(lambda cell: cell.math.round_to_decimal_places(0))
        +---------+
        |       a |
        |     --- |
        |     f64 |
        +=========+
        | 1.00000 |
        | 1.00000 |
        | 3.00000 |
        |    null |
        +---------+

        >>> column.transform(lambda cell: cell.math.round_to_decimal_places(2))
        +---------+
        |       a |
        |     --- |
        |     f64 |
        +=========+
        | 1.00000 |
        | 1.12000 |
        | 3.46000 |
        |    null |
        +---------+
        """

    @abstractmethod
    def round_to_significant_figures(self, significant_figures: int) -> Cell:
        """
        Round to the specified number of significant figures.

        Parameters
        ----------
        significant_figures:
            The number of significant figures to round to.

        Returns
        -------
        cell:
            The rounded value.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [0.999, 1.123, 3.456, None])
        >>> column.transform(lambda cell: cell.math.round_to_significant_figures(1))
        +---------+
        |       a |
        |     --- |
        |     f64 |
        +=========+
        | 1.00000 |
        | 1.00000 |
        | 3.00000 |
        |    null |
        +---------+

        >>> column.transform(lambda cell: cell.math.round_to_significant_figures(2))
        +---------+
        |       a |
        |     --- |
        |     f64 |
        +=========+
        | 1.00000 |
        | 1.10000 |
        | 3.50000 |
        |    null |
        +---------+
        """

    @abstractmethod
    def sign(self) -> Cell:
        """
        Get the sign (-1 for negative numbers, 0 for zero, and 1 for positive numbers).

        Note that IEEE 754 defines a negative zero (-0) and a positive zero (+0). This method return a negative zero
        for -0 and a positive zero for +0.

        Returns
        -------
        cell:
            The sign.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("a", [-1, 0, 1, None])
        >>> column1.transform(lambda cell: cell.math.sign())
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |   -1 |
        |    0 |
        |    1 |
        | null |
        +------+

        >>> column2 = Column("a", [-1.0, -0.0, +0.0, 1.0, None])
        >>> column2.transform(lambda cell: cell.math.sign())
        +----------+
        |        a |
        |      --- |
        |      f64 |
        +==========+
        | -1.00000 |
        | -0.00000 |
        |  0.00000 |
        |  1.00000 |
        |     null |
        +----------+
        """

    @abstractmethod
    def sin(self) -> Cell:
        """
        Get the sine.

        Returns
        -------
        cell:
            The sine.

        Examples
        --------
        >>> import math
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [0, math.pi / 2, math.pi, 3 * math.pi / 2, None])
        >>> column.transform(lambda cell: cell.math.sin())
        +----------+
        |        a |
        |      --- |
        |      f64 |
        +==========+
        |  0.00000 |
        |  1.00000 |
        |  0.00000 |
        | -1.00000 |
        |     null |
        +----------+
        """

    @abstractmethod
    def sinh(self) -> Cell:
        """
        Get the hyperbolic sine.

        Returns
        -------
        cell:
            The hyperbolic sine.

        Examples
        --------
        >>> import math
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [-1, 0, 1, None])
        >>> column.transform(lambda cell: cell.math.sinh())
        +----------+
        |        a |
        |      --- |
        |      f64 |
        +==========+
        | -1.17520 |
        |  0.00000 |
        |  1.17520 |
        |     null |
        +----------+
        """

    @abstractmethod
    def tan(self) -> Cell:
        """
        Get the tangent.

        Returns
        -------
        cell:
            The tangent.

        Examples
        --------
        >>> import math
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [0, math.pi / 4, 3 * math.pi / 4, None])
        >>> column.transform(lambda cell: cell.math.tan())
        +----------+
        |        a |
        |      --- |
        |      f64 |
        +==========+
        |  0.00000 |
        |  1.00000 |
        | -1.00000 |
        |     null |
        +----------+
        """

    @abstractmethod
    def tanh(self) -> Cell:
        """
        Get the hyperbolic tangent.

        Returns
        -------
        cell:
            The hyperbolic tangent.

        Examples
        --------
        >>> import math
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [-1, 0, 1, None])
        >>> column.transform(lambda cell: cell.math.tanh())
        +----------+
        |        a |
        |      --- |
        |      f64 |
        +==========+
        | -0.76159 |
        |  0.00000 |
        |  0.76159 |
        |     null |
        +----------+
        """
