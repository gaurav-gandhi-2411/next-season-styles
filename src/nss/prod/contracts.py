"""Data contracts for the three raw inputs and the style-week panel (SPEC Phase C1).

Library: **pandera** with its polars backend. Checks are vectorised, so a 31.8M-row transactions
table validates in one pass, and a failure comes back row by row (column, check, value, row
index). Pydantic validates one object at a time: it is used for API request bodies
(`nss.prod.api`), not for tables.

`validate(frame, name)` raises `ContractViolation` carrying a row-level report of every violation.
Nothing is coerced or dropped: a frame that breaks its contract is rejected as a whole.

Ranges come from profiling the H&M data (2026-09-23), widened only where the quantity is bounded
by meaning rather than by what was observed (e.g. price > 0, not price <= 0.5915).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandera.polars as pa
import polars as pl
from pandera.errors import SchemaErrors

STYLE_KEY_COLS = [
    "index_group_name",
    "product_type_name",
    "garment_group_name",
    "perceived_colour_master_name",
    "graphical_appearance_name",
]
MIN_DATE = date(2018, 1, 1)


def _row_check(fn: object, name: str) -> pa.Check:
    """A vectorised row-level check. pandera-polars hands a check `(lazyframe, key)`, not a Series;
    `fn` maps the column expression to a boolean expression, one value per row. (A check returning
    a single scalar breaks pandera's failure-report assembly, so every check here is row-level.)"""
    return pa.Check(lambda d: d.lazyframe.select(fn(pl.col(d.key))), name=name)  # type: ignore[operator]


_NON_EMPTY = _row_check(lambda c: c.str.strip_chars().str.len_chars() > 0, "non_empty")
_INTEGRAL = _row_check(lambda c: c.cast(pl.Float64) == c.cast(pl.Float64).floor(), "integral")


def _non_empty_str(nullable: bool = False) -> pa.Column:
    return pa.Column(pl.String, _NON_EMPTY, nullable=nullable)


ARTICLES = pa.DataFrameSchema(
    {
        "article_id": pa.Column(pl.String, pa.Check.str_matches(r"^\d{10}$"), unique=True),
        "product_code": pa.Column(pl.String, pa.Check.str_matches(r"^\d{7}$")),
        **{c: _non_empty_str() for c in STYLE_KEY_COLS},
        "detail_desc": pa.Column(pl.String, nullable=True),
    },
    name="articles",
    strict=False,  # the file has 25 columns; the contract covers the ones the pipeline reads
)

CUSTOMERS = pa.DataFrameSchema(
    {
        "customer_id": pa.Column(pl.String, pa.Check.str_matches(r"^[0-9a-f]{64}$"), unique=True),
        "FN": pa.Column(pl.Float64, pa.Check.isin([1.0]), nullable=True),
        "Active": pa.Column(pl.Float64, pa.Check.isin([1.0]), nullable=True),
        "club_member_status": pa.Column(
            pl.String, pa.Check.isin(["ACTIVE", "PRE-CREATE", "LEFT CLUB"]), nullable=True
        ),
        # Known quirk in the source: both "NONE" and "None" occur. Accepted as-is, not normalised.
        "fashion_news_frequency": pa.Column(
            pl.String, pa.Check.isin(["Regularly", "Monthly", "NONE", "None"]), nullable=True
        ),
        "age": pa.Column(pl.Int64, pa.Check.in_range(16, 110), nullable=True),
        "postal_code": _non_empty_str(),
    },
    name="customers",
    strict=True,
)

TRANSACTIONS = pa.DataFrameSchema(
    {
        "t_dat": pa.Column(pl.Date, pa.Check.ge(MIN_DATE)),
        "customer_id": pa.Column(pl.String, pa.Check.str_matches(r"^[0-9a-f]{64}$")),
        "article_id": pa.Column(pl.String, pa.Check.str_matches(r"^\d{10}$")),
        "price": pa.Column(pl.Float64, pa.Check.gt(0.0)),
        "sales_channel_id": pa.Column(pl.Int64, pa.Check.isin([1, 2])),
    },
    name="transactions",
    strict=True,
)

PANEL = pa.DataFrameSchema(
    {
        "style_key": _non_empty_str(),
        **{c: _non_empty_str() for c in STYLE_KEY_COLS},
        "week_start": pa.Column(
            pl.Date,
            [
                pa.Check.ge(MIN_DATE),
                _row_check(lambda c: c.dt.weekday() == 1, "week_starts_monday"),
            ],
        ),
        # Any integer dtype (UInt32 on disk, Int64 from JSON): checked by value, not by dtype.
        "units": pa.Column(None, [_INTEGRAL, pa.Check.ge(0)]),
        "n_active_articles": pa.Column(None, [_INTEGRAL, pa.Check.ge(0)]),
        "units_per_active_article": pa.Column(pl.Float64, pa.Check.ge(0.0)),
        "price_index": pa.Column(pl.Float64, pa.Check.gt(0.0), nullable=True),
        "intensity_shrunk": pa.Column(pl.Float64, pa.Check.ge(0.0), nullable=True),
        "first_week_seen": pa.Column(pl.Date),
    },
    name="style_week_panel",
    strict=False,  # extra panel columns (revenue, channel splits, ...) are allowed
    coerce=False,
    unique=["style_key", "week_start"],
)


@dataclass
class ContractViolation(Exception):
    """A frame broke its contract. `report` has one row per violating (row, column, check)."""

    contract: str
    report: pl.DataFrame

    def __str__(self) -> str:
        head = self.report.head(20)
        return (
            f"contract '{self.contract}' violated: {self.report.height} failure(s); "
            f"first {head.height}:\n{head}"
        )


def _report(contract: str, failure_cases: pl.DataFrame) -> pl.DataFrame:
    cols = [c for c in ("column", "check", "failure_case", "index") if c in failure_cases.columns]
    return failure_cases.select(cols).with_columns(pl.lit(contract).alias("contract"))


def _cross_field(frame: pl.DataFrame, name: str) -> pl.DataFrame:
    """Row-level checks spanning columns, which pandera column checks cannot express."""
    rows = frame.with_row_index("index")
    parts: list[pl.DataFrame] = []

    def add(mask: pl.Expr, check: str, column: str) -> None:
        bad = rows.filter(mask)
        if bad.height:
            parts.append(
                bad.select(
                    pl.lit(column).alias("column"),
                    pl.lit(check).alias("check"),
                    pl.col(column).cast(pl.String).alias("failure_case"),
                    pl.col("index").cast(pl.Int64),
                )
            )

    if name == "style_week_panel":
        key = pl.concat_str(STYLE_KEY_COLS, separator=" || ")
        add(pl.col("style_key") != key, "style_key_matches_attributes", "style_key")
        add(
            pl.col("week_start") < pl.col("first_week_seen"),
            "first_week_seen_le_week",
            "week_start",
        )
        add(
            (pl.col("n_active_articles") == 0) & (pl.col("units") > 0),
            "units_require_active_articles",
            "units",
        )
    if not parts:
        return pl.DataFrame(
            schema={
                "column": pl.String,
                "check": pl.String,
                "failure_case": pl.String,
                "index": pl.Int64,
            }
        )
    return pl.concat(parts)


def validate(frame: pl.DataFrame, name: str) -> pl.DataFrame:
    """Validate `frame` against contract `name`; return it unchanged or raise ContractViolation."""
    schema = {s.name: s for s in (ARTICLES, CUSTOMERS, TRANSACTIONS, PANEL)}[name]
    failures: list[pl.DataFrame] = []
    try:
        schema.validate(frame, lazy=True)
    except SchemaErrors as e:
        fc = e.failure_cases.with_columns(
            pl.col("failure_case").cast(pl.String), pl.col("index").cast(pl.Int64)
        )
        failures.append(_report(name, fc))
    needed = {"style_key", *STYLE_KEY_COLS, "week_start", "first_week_seen", "units"}
    if name == "style_week_panel" and needed | {"n_active_articles"} <= set(frame.columns):
        cross = _cross_field(frame, name)
        if cross.height:
            failures.append(_report(name, cross))
    if failures:
        raise ContractViolation(name, pl.concat(failures, how="diagonal_relaxed"))
    return frame


def check_references(
    transactions: pl.DataFrame | pl.LazyFrame,
    articles: pl.DataFrame,
    customers: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Referential integrity: every transaction's article (and customer) must exist.

    Raises ContractViolation with the offending transaction rows; returns the transactions frame.
    """
    tx = transactions.lazy().with_row_index("index")
    parts = [
        tx.join(articles.lazy().select("article_id"), on="article_id", how="anti").select(
            pl.lit("article_id").alias("column"),
            pl.lit("article_exists").alias("check"),
            pl.col("article_id").alias("failure_case"),
            pl.col("index").cast(pl.Int64),
        )
    ]
    if customers is not None:
        parts.append(
            tx.join(customers.lazy().select("customer_id"), on="customer_id", how="anti").select(
                pl.lit("customer_id").alias("column"),
                pl.lit("customer_exists").alias("check"),
                pl.col("customer_id").alias("failure_case"),
                pl.col("index").cast(pl.Int64),
            )
        )
    bad = pl.concat([p.collect() for p in parts])
    if bad.height:
        raise ContractViolation("transactions_references", _report("transactions_references", bad))
    return transactions.collect() if isinstance(transactions, pl.LazyFrame) else transactions


def read_articles(path: str) -> pl.DataFrame:
    """Articles with ids kept as strings (they have leading zeros)."""
    return pl.read_csv(
        path,
        schema_overrides={"article_id": pl.String, "product_code": pl.String},
        infer_schema_length=10000,
    )


def read_customers(path: str) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema_length=10000)


def read_transactions(path: str) -> pl.LazyFrame:
    """Raw transactions CSV, or the project's partitioned parquet, with typed columns."""
    if path.endswith(".csv"):
        lf = pl.scan_csv(path, schema_overrides={"article_id": pl.String, "t_dat": pl.Date})
    else:
        lf = pl.scan_parquet(path, hive_partitioning=True).drop("year_month", strict=False)
    return lf.with_columns(
        pl.col("article_id").cast(pl.String).str.zfill(10), pl.col("t_dat").cast(pl.Date)
    ).select("t_dat", "customer_id", "article_id", "price", "sales_channel_id")
