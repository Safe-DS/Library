import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import FunctionalTableTransformer
from safeds.exceptions import ColumnNotFoundError, ColumnTypeError, TransformerNotFittedError


#def invalid_callable(i: int) -> float:
#    return float(i)
    
#def valid_callable(table) -> Table:
#    new_table = table.remove_columns(["col1"])
#    return new_table

class TestInit:
    def invalid_callable(self, i: int) -> float:
        return float(i)
    
    def valid_callable(self, table) -> Table:
        new_table = table.remove_columns(self, ["col1"])
        return new_table

    #def test_should_raise_type_error(self) -> None:
    #    with pytest.raises(TypeError):
    #        transformer = FunctionalTableTransformer(invalid_callable)
    
    def test_should_not_raise_type_error(self) -> None:
            transformer = FunctionalTableTransformer(self.valid_callable)

class TestFit:
    def valid_callable(self, table: Table) -> Table:
        new_table = table.remove_columns(["col1"])
        return new_table
    
    def test_should_return_self(self) -> None:
        table = Table(
            {
                "col1": [1, 2, 3],
                "col2": [1, 2, 3],
            },
        )
        transformer = FunctionalTableTransformer(self.valid_callable)
        assert transformer.fit(table) is transformer

class TestIsFitted:
    def valid_callable(self, table: Table) -> Table:
        new_table = table.remove_columns(["col1"])
        return new_table
    
    def test_should_always_be_fitted(self) -> None:
        transformer = FunctionalTableTransformer(self.valid_callable)
        assert transformer.is_fitted

class TestTransform:
    def valid_callable(self, table: Table) -> Table:
        new_table = table.remove_columns(["col1"])
        return new_table
    
    def test_should_raise_generic_error(self) -> None:
        table = Table(
            {
                "col2": [1, 2, 3],
            
            },
        )
        transformer = FunctionalTableTransformer(self.valid_callable)
        with pytest.raises(Exception, match=r"The underlying function encountered an error"):
            transformer.transform(table)

    def test_should_not_modify_original_table(self) -> None:
        table = Table(
            {
                "col1": [1, 2, 3],
                "col2": [1, 2, 3],
            
            },
        )
        transformer = FunctionalTableTransformer(self.valid_callable)
        transformer.transform(table)
        assert table == Table(
            {
                "col1": [1, 2, 3],
                "col2": [1, 2, 3],
            
            },
        )

    def test_should_return_modified_table(self) -> None:
        table = Table(
            {
                "col1": [1, 2, 3],
                "col2": [1, 2, 3],
            
            },
        )
        transformer = FunctionalTableTransformer(self.valid_callable)
        transformed_table = transformer.transform(table)
        assert transformed_table == Table(
            {
                "col2": [1, 2, 3],
            
            },
        )

class TestFitAndTransform:
    def valid_callable(self, table: Table) -> Table:
        new_table = table.remove_columns(["col1"])
        return new_table

    def test_should_return_self(self) -> None:
        table = Table(
            {
                "col1": [1, 2, 3],
                "col2": [1, 2, 3],
            
            },
        )
        transformer = FunctionalTableTransformer(self.valid_callable)
        assert transformer.fit_and_transform(table)[0] is transformer

    def test_should_not_modify_original_table(self) -> None:
        table = Table(
            {
                "col1": [1, 2, 3],
                "col2": [1, 2, 3],
            
            },
        )
        transformer = FunctionalTableTransformer(self.valid_callable)
        transformer.fit_and_transform(table)
        assert table == Table(
            {
                "col1": [1, 2, 3],
                "col2": [1, 2, 3],
            
            },
        )
