from timeit import timeit

from safeds.data.tabular.containers import Table

from benchmarks.table.utils import create_synthetic_table_polars

REPETITIONS = 10


def _run_remove_duplicate_rows() -> None:
    table.remove_duplicate_rows()._lazy_frame.collect()


def _run_remove_rows_with_missing_values() -> None:
    table.remove_rows_with_missing_values()._lazy_frame.collect()


# def _run_remove_rows_with_outliers() -> None:
#     table.remove_rows_with_outliers()


def _run_remove_rows() -> None:
    table.remove_rows(lambda row: row.get_value("column_0") % 2 == 0)._lazy_frame.collect()


def _run_shuffle_rows() -> None:
    table.shuffle_rows()._lazy_frame.collect()


def _run_slice_rows() -> None:
    table.slice_rows(length=table.number_of_rows // 2)._lazy_frame.collect()


def _run_sort_rows() -> None:
    table.sort_rows(lambda row: row.get_value("column_0"))._lazy_frame.collect()


def _run_sort_rows_by_column() -> None:
    table.sort_rows_by_column("column_0")._lazy_frame.collect()


def _run_split_rows() -> None:
    table_1, table_2 = table.split_rows(0.5)
    table_1._lazy_frame.collect()
    table_2._lazy_frame.collect()


def _run_transform_column() -> None:
    table.transform_column("column_0", lambda value: value * 2)._lazy_frame.collect()


if __name__ == "__main__":
    # Create a synthetic Table
    table = create_synthetic_table_polars(1000, 50)

    # Run the benchmarks
    timings: dict[str, float] = {
        "remove_duplicate_rows": timeit(
            _run_remove_duplicate_rows,
            number=REPETITIONS,
        ),
        "remove_rows_with_missing_values": timeit(
            _run_remove_rows_with_missing_values,
            number=REPETITIONS,
        ),
        # "remove_rows_with_outliers": timeit(
        #     _run_remove_rows_with_outliers,
        #     number=REPETITIONS,
        # ),
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
        "sort_rows_by_column": timeit(
            _run_sort_rows_by_column,
            number=REPETITIONS,
        ),
        "split_rows": timeit(
            _run_split_rows,
            number=REPETITIONS,
        ),
        "transform_column": timeit(
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

# 0                        group_rows  4.801467
# 1             remove_duplicate_rows  0.287240
# 2   remove_rows_with_missing_values  0.279085
# 3         remove_rows_with_outliers  0.706546
# 4                       remove_rows  3.040592
# 5                      shuffle_rows  0.267581
# 6                        slice_rows  0.143297
# 7                         sort_rows  4.613575
# 8                        split_rows  0.274414
# 9                           to_rows  2.710391
# 10                  transform_colum  3.191619
