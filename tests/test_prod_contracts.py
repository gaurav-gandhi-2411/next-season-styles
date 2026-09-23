from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from nss.prod.contracts import STYLE_KEY_COLS, ContractViolation, check_references, validate

CID_A = "a" * 64
CID_B = "b" * 64


def _articles() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "article_id": ["0108775015", "0108775044"],
            "product_code": ["0108775", "0108775"],
            "index_group_name": ["Ladieswear", "Ladieswear"],
            "product_type_name": ["Vest top", "Vest top"],
            "garment_group_name": ["Jersey Basic", "Jersey Basic"],
            "perceived_colour_master_name": ["Black", "White"],
            "graphical_appearance_name": ["Solid", "Solid"],
            "detail_desc": ["Jersey top", None],
        }
    )


def _customers() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "customer_id": [CID_A, CID_B],
            "FN": [1.0, None],
            "Active": [None, 1.0],
            "club_member_status": ["ACTIVE", None],
            "fashion_news_frequency": ["NONE", "None"],  # both spellings occur in the source
            "age": [24, None],
            "postal_code": ["p1", "p2"],
        }
    )


def _transactions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t_dat": [date(2019, 1, 7), date(2019, 1, 8)],
            "customer_id": [CID_A, CID_B],
            "article_id": ["0108775015", "0108775044"],
            "price": [0.05, 0.03],
            "sales_channel_id": [1, 2],
        }
    )


def _panel() -> pl.DataFrame:
    attrs = ["Ladieswear", "Vest top", "Jersey Basic", "Black", "Solid"]
    return pl.DataFrame(
        {
            "style_key": [" || ".join(attrs)] * 2,
            **{c: [v, v] for c, v in zip(STYLE_KEY_COLS, attrs, strict=True)},
            "week_start": [date(2019, 1, 7), date(2019, 1, 14)],
            "units": pl.Series([10, 0], dtype=pl.UInt32),
            "n_active_articles": pl.Series([2, 1], dtype=pl.UInt32),
            "units_per_active_article": [5.0, 0.0],
            "price_index": [1.0, None],
            "intensity_shrunk": [4.8, 0.3],
            "first_week_seen": [date(2019, 1, 7)] * 2,
        }
    )


@pytest.mark.parametrize(
    ("name", "frame"),
    [
        ("articles", _articles()),
        ("customers", _customers()),
        ("transactions", _transactions()),
        ("style_week_panel", _panel()),
    ],
)
def test_valid_inputs_pass(name: str, frame: pl.DataFrame) -> None:
    assert validate(frame, name) is frame


def _violations(frame: pl.DataFrame, name: str) -> pl.DataFrame:
    with pytest.raises(ContractViolation) as err:
        validate(frame, name)
    return err.value.report


def test_articles_duplicate_and_malformed_id_rejected_row_level() -> None:
    bad = _articles().with_columns(pl.Series("article_id", ["0108775015", "0108775015"]))
    bad = pl.concat([bad, _articles().head(1).with_columns(pl.lit("12345").alias("article_id"))])
    rep = _violations(bad, "articles")
    uniq = rep.filter(pl.col("check") == "field_uniqueness")
    assert sorted(uniq["index"].to_list()) == [0, 1]  # rows 0 and 1 share an id
    # row 2 ("12345") is unique but malformed: caught by the format check, not uniqueness
    assert rep.filter(pl.col("check").str.contains("str_matches"))["index"].to_list() == [2]


def test_customers_unknown_category_and_bad_age_rejected() -> None:
    bad = _customers().with_columns(
        pl.Series("club_member_status", ["ACTIVE", "VIP"]), pl.Series("age", [24, 7])
    )
    rep = _violations(bad, "customers")
    assert set(rep["column"].to_list()) == {"club_member_status", "age"}
    assert rep["index"].to_list() == [1, 1]


def test_transactions_bad_price_and_channel_rejected() -> None:
    bad = _transactions().with_columns(
        pl.Series("price", [0.05, -0.01]), pl.Series("sales_channel_id", [3, 2])
    )
    rep = _violations(bad, "transactions")
    got = {(r["column"], r["index"]) for r in rep.iter_rows(named=True)}
    assert got == {("price", 1), ("sales_channel_id", 0)}


def test_transactions_missing_column_rejected() -> None:
    rep = _violations(_transactions().drop("price"), "transactions")
    assert rep.filter(pl.col("failure_case") == "price").height >= 1


def test_referential_integrity_reports_orphan_rows() -> None:
    tx = pl.concat(
        [
            _transactions(),
            _transactions().head(1).with_columns(pl.lit("0999999999").alias("article_id")),
        ]
    )
    with pytest.raises(ContractViolation) as err:
        check_references(tx, _articles(), _customers())
    rep = err.value.report
    assert rep["check"].to_list() == ["article_exists"]
    assert rep["index"].to_list() == [2]
    assert rep["failure_case"].to_list() == ["0999999999"]


def test_referential_integrity_passes_clean_data() -> None:
    assert check_references(_transactions(), _articles(), _customers()).height == 2


def test_panel_corrupted_rejected_on_every_check() -> None:
    """A deliberately corrupted panel: duplicate key, negative units, fractional units, a Tuesday
    week, a style_key that does not match its attributes, units with no active articles."""
    p = _panel()
    bad = pl.concat(
        [
            p,
            p.head(1),  # duplicate (style_key, week_start)
        ]
    ).with_columns(
        pl.Series("units", [10, -3, 10], dtype=pl.Int64),
        pl.Series("n_active_articles", [2.5, 0, 2], dtype=pl.Float64),
        pl.Series("week_start", [date(2019, 1, 7), date(2019, 1, 15), date(2019, 1, 7)]),
    )
    bad = bad.with_columns(
        pl.when(pl.int_range(pl.len()) == 2)
        .then(pl.lit("wrong key"))
        .otherwise(pl.col("style_key"))
        .alias("style_key")
    )
    rep = _violations(bad, "style_week_panel")
    checks = set(rep["check"].to_list())
    assert "greater_than_or_equal_to(0)" in checks  # negative units, row 1
    assert "integral" in checks  # 2.5 active articles, row 0
    assert "week_starts_monday" in checks  # 2019-01-15 is a Tuesday, row 1
    assert "style_key_matches_attributes" in checks  # row 2
    assert rep.filter(pl.col("check") == "style_key_matches_attributes")["index"].to_list() == [2]


def test_panel_duplicate_key_rejected() -> None:
    p = _panel()
    rep = _violations(pl.concat([p, p.head(1)]), "style_week_panel")
    assert rep.filter(pl.col("check") == "multiple_fields_uniqueness").height == 2


def test_panel_units_without_active_articles_rejected() -> None:
    bad = _panel().with_columns(pl.Series("n_active_articles", [0, 1], dtype=pl.UInt32))
    rep = _violations(bad, "style_week_panel")
    assert rep.filter(pl.col("check") == "units_require_active_articles")["index"].to_list() == [0]
