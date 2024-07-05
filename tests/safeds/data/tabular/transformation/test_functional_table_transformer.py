import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation._functional_table_transformer import FunctionalTableTransformer
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

class TestTransform:
    def valid_callable(table: Table) -> Table:
        new_table = table.remove_columns(["col1"])
        return new_table
    def sshould_raise_generic_error(self):
        #TODO: implement try catch for main method
        #TODO: test with a callable like remove columns where we can check error returning if we return non generic error
        assert None is not None

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

        





        

        