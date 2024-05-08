from timeit import timeit

from safeds.data.tabular.containers import ExperimentalTable

from benchmarks.table.utils import create_synthetic_table_polars

REPETITIONS = 10


def _run_remove_columns_with_missing_values() -> None:
    table.remove_columns_with_missing_values()._lazy_frame.collect()


def _run_remove_non_numeric_columns() -> None:
    table.remove_non_numeric_columns()._lazy_frame.collect()


if __name__ == "__main__":
    # Create a synthetic Table
    table = create_synthetic_table_polars(100, 50000)

    # Run the benchmarks
    timings: dict[str, float] = {
        "remove_columns_with_missing_values": timeit(
            _run_remove_columns_with_missing_values,
            number=REPETITIONS,
        ),
        "remove_non_numeric_columns": timeit(
            _run_remove_non_numeric_columns,
            number=REPETITIONS,
        ),
    }

    # Print the timings
    print(
        ExperimentalTable(
            {
                "method": list(timings.keys()),
                "timing": list(timings.values()),
            }
        )
    )
