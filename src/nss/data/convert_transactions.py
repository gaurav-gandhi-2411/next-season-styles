"""Convert the raw H&M `transactions_train.csv` to Parquet, partitioned by year-month.

One-off conversion utility for Phase 1 data acquisition. Reads the raw CSV (written by the
Kaggle download step) and writes a Hive-style partitioned Parquet dataset keyed on `year_month`
(derived from `t_dat`) so downstream panel-building code can prune by date range cheaply.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl


def convert_transactions_to_parquet(csv_path: Path, out_dir: Path) -> tuple[int, int]:
    """Convert `transactions_train.csv` to a year-month partitioned Parquet dataset.

    Args:
        csv_path: Path to the raw `transactions_train.csv` file.
        out_dir: Directory to write the partitioned Parquet dataset into.

    Returns:
        A tuple of (row_count_in, row_count_out) for the caller to sanity-check.
    """
    lf = pl.scan_csv(csv_path, try_parse_dates=True)
    lf = lf.with_columns(pl.col("t_dat").dt.strftime("%Y-%m").alias("year_month"))

    row_count_in = lf.select(pl.len()).collect().item()

    out_dir.mkdir(parents=True, exist_ok=True)
    df = lf.collect(engine="streaming")
    df.write_parquet(
        out_dir,
        use_pyarrow=True,
        pyarrow_options={"partition_cols": ["year_month"]},
    )

    row_count_out = pl.scan_parquet(out_dir / "**/*.parquet").select(pl.len()).collect().item()
    return row_count_in, row_count_out


def main() -> None:
    """CLI entry point: convert `data/raw/transactions_train.csv` to partitioned Parquet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("data/raw/transactions_train.csv"),
        help="Path to raw transactions_train.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/interim/transactions_train_parquet"),
        help="Output directory for the partitioned Parquet dataset",
    )
    args = parser.parse_args()

    start = time.time()
    row_count_in, row_count_out = convert_transactions_to_parquet(args.csv_path, args.out_dir)
    elapsed = time.time() - start

    print(f"Converted {args.csv_path} -> {args.out_dir}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Row count in:  {row_count_in}")
    print(f"Row count out: {row_count_out}")
    print(f"Match: {row_count_in == row_count_out}")


if __name__ == "__main__":
    main()
