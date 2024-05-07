from timeit import timeit

from safeds.data.tabular.containers import Table

from benchmarks.table.utils import create_synthetic_table

REPETITIONS = 10


def _run_group_rows() -> None:
    table.group_rows(lambda row: row.get_value("column_0") % 2 == 0)


def _run_remove_duplicate_rows() -> None:
    table.remove_duplicate_rows()


def _run_remove_rows_with_missing_values() -> None:
    table.remove_rows_with_missing_values()


def _run_remove_rows_with_outliers() -> None:
    table.remove_rows_with_outliers()


def _run_remove_rows() -> None:
    table.remove_rows(lambda row: row.get_value("column_0") % 2 == 0)


def _run_shuffle_rows() -> None:
    table.shuffle_rows()


def _run_slice_rows() -> None:
    table.slice_rows(end=table.number_of_rows // 2)


def _run_sort_rows() -> None:
    table.sort_rows(lambda row1, row2: row1.get_value("column_0") - row2.get_value("column_0"))


def _run_split_rows() -> None:
    table.split_rows(0.5)


def _run_to_rows() -> None:
    table.to_rows()


def _run_transform_column() -> None:
    table.transform_column("column_0", lambda row: row.get_value("column_0") * 2)


if __name__ == "__main__":
    # Create a synthetic Table
    table = create_synthetic_table(1000, 50)

    # Run the benchmarks
    timings: dict[str, float] = {
        "group_rows": timeit(
            _run_group_rows,
            number=REPETITIONS,
        ),
        "remove_duplicate_rows": timeit(
            _run_remove_duplicate_rows,
            number=REPETITIONS,
        ),
        "remove_rows_with_missing_values": timeit(
            _run_remove_rows_with_missing_values,
            number=REPETITIONS,
        ),
        "remove_rows_with_outliers": timeit(
            _run_remove_rows_with_outliers,
            number=REPETITIONS,
        ),
        "remove_rows": timeit(
            _run_remove_rows,
            number=REPETITIONS,
        ),
        "shuffle_rows": timeit(
            _run_shuffle_rows,
            number=REPETITIONS,
        ),
        "slice_rows": timeit(
            _run_slice_rows,
            number=REPETITIONS,
        ),
        "sort_rows": timeit(
            _run_sort_rows,
            number=REPETITIONS,
        ),
        "split_rows": timeit(
            _run_split_rows,
            number=REPETITIONS,
        ),
        "to_rows": timeit(
            _run_to_rows,
            number=REPETITIONS,
        ),
        "transform_colum": timeit(
            _run_transform_column,
            number=REPETITIONS,
        ),
    }

    # Print the timings
    print(
        Table(
            {  # noqa: T201
                "method": list(timings.keys()),
                "timing": list(timings.values()),
            }
        )
    )
