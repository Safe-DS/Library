from __future__ import annotations

from multiprocessing import Process, Queue
from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import InvertibleTableTransformer, TableTransformer

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Column

class ParallelTableTransformer(InvertibleTableTransformer):
    def __init__(self, *, transformers: list[TableTransformer]):
        super().__init__(None)

        if transformers is None or len(transformers) == 0:
            raise ValueError("Transformers must contain at least one transformer.")

        all_affected_columns = []
        for tf in transformers:
            all_affected_columns.extend(tf._column_names)
        all_affected_columns_unique = set(all_affected_columns)

        for val in all_affected_columns:
            if val in all_affected_columns_unique:
                all_affected_columns_unique.remove(val)
            else:
                raise BaseException("Cannot apply two transformers to the same column at the same time.")

        self._transformers: list[TableTransformer] = transformers
        self._is_fitted: bool = False
        self._q: Queue = Queue()

    def __hash__(self):
        return _structural_hash(
            super().__hash__(),
            self._transformers,
            self._is_fitted,
        )

    @property
    def is_fitted(self):
        return self._is_fitted

    def _worker_fit(self, func: callable[[Table], Table], table: Table) -> None:
        res = func.__call__(table)
        self._q.put(res)

    def _worker_transform(self, func: callable[[Table], Table], table: Table, column_names: str | list[str]) -> None:
        transformed_table = func.__call__(table)
        res = [transformed_table.get_column(i) for i in column_names]
        self._q.put(res)

    def fit(self, table: Table) -> ParallelTableTransformer:

        # Initialize lists for return values and processes
        transformers: list[Column] = []
        procs: list[Process] = []

        # Start a process for every transformer
        for tf in self._transformers:
            p = Process(target=self._worker_fit, args=(tf.fit, table))
            p.start()
            procs.append(p)
        
        for proc in procs:
            transformers.append(self._q.get())
            proc.join()

        return ParallelTableTransformer(transformers=transformers)
            


    def transform(self, table: Table) -> Table:
        # Initialize lists for return values and processes
        cols: list[Column] = []
        procs: list[Process] = []

        # Start a process for every transformer
        for tf in self._transformers:
            p = Process(target=self._worker_transform, args=(tf.transform, table, tf._column_names))
            p.start()
            procs.append(p)
        
        for proc in procs:
            cols.extend(self._q.get())
            proc.join()

        col_names = [col.name for col in cols]
        for name_in_table in table.column_names:
            if name_in_table in col_names:
                pass
            else:
                cols.append(table.get_column(name_in_table))

        return Table.from_columns(cols)

    def inverse_transform(self, table: Table) -> Table:
        pass

