import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import FunctionalTableTransformer
from safeds.exceptions import ColumnNotFoundError, ColumnTypeError, TransformerNotFittedError


class TestInit:
    def should_raise_type_error(self):
        assert None is not None

class TestFit:
    def valid_callable(table: Table) -> Table:
        new_table = table.remove_columns(["col1"])
        return new_table
    
    def should_return_self(self) -> FunctionalTableTransformer:
        table = Table(
            {
                "col1": [1, 2, 3],
                "col2": [1, 2, 3],
            },
        )
        transformer = FunctionalTableTransformer(self.valid_callable())
        assert transformer.fit(table) is transformer

class TestIsFitted:
    def valid_callable(table: Table) -> Table:
        new_table = table.remove_columns(["col1"])
        return new_table
    
    def should_always_be_fitted(self) -> bool:
        transformer = FunctionalTableTransformer(self.valid_callable())
        assert transformer.is_fitted

class TestTransform:
    def valid_callable(table: Table) -> Table:
        new_table = table.remove_columns(["col1"])
        return new_table
    
    def should_raise_generic_error(self):
        table = Table(
            {
                "col2": [1, 2, 3],
            
            },
        )
        transformer = FunctionalTableTransformer(self.valid_callable())
        with pytest.raises(Exception, match=r"The underlying function encountered an error"):
            transformer.transform(table)

    def should_not_modify_original_table(self):
        table = Table(
            {
                "col1": [1, 2, 3],
                "col2": [1, 2, 3],
            
            },
        )
        transformer = FunctionalTableTransformer(self.valid_callable())
        transformer.transform(table)
        assert table == Table(
            {
                "col1": [1, 2, 3],
                "col2": [1, 2, 3],
            
            },
        )

class TestFitAndTransform:
    def valid_callable(table: Table) -> Table:
        new_table = table.remove_columns(["col1"])
        return new_table

    def should_return_self(self):
        table = Table(
            {
                "col1": [1, 2, 3],
                "col2": [1, 2, 3],
            
            },
        )
        transformer = FunctionalTableTransformer(self.valid_callable())
        assert transformer.fit_and_transform(table)[0] is transformer

    def should_not_modify_original_table(self):
        table = Table(
            {
                "col1": [1, 2, 3],
                "col2": [1, 2, 3],
            
            },
        )
        transformer = FunctionalTableTransformer(self.valid_callable())
        transformer.fit_and_transform(table)
        assert table == Table(
            {
                "col1": [1, 2, 3],
                "col2": [1, 2, 3],
            
            },
        )







        

        