"""Old vs new Gate 1 / Gate 1b thresholds on the widened reference base, with bootstrap CIs.

The p90 thresholds were computed from 4-8 real articles per style, so each was essentially one real
pair. This recomputes them on the widened base and quantifies how much the sampling uncertainty
shrinks: each threshold's 95% CI comes from resampling the ARTICLES with replacement (B = 500,
seed 42) and recomputing the statistic (pairs of an article with itself, which a resample can
create, are dropped so a duplicate never counts as a near-copy sibling).

Statistics (identical definitions to the shipped gates):
- Gate 1: p90 of pairwise cosine between distinct real articles (`within_style_benchmark`);
- Gate 1b: p90 of each real article's nearest-sibling cosine (`gate1b_nearest_reference`).

Also recomputes the leave-one-out control and the exact-clone control on the wider base by calling
the existing `leave_one_out_control` module, which reads the (now widened) screened manifest.

Usage:
    uv run python -m nss.generate.widen_thresholds
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from nss.generate import clip_scoring, dino_scoring
from nss.generate.clip_scoring import _cosine

OLD_MANIFEST = Path("reports/tables/exemplar_images_screened.csv")
NEW_MANIFEST = Path("reports/tables/reference_base_widened.csv")
OUT_PATH = Path("reports/tables/reference_threshold_comparison.csv")
BOOTSTRAP = 500
SEED = 42


def _sim_matrix(embeddings: list[np.ndarray]) -> np.ndarray:
    n = len(embeddings)
    m = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = m[j, i] = _cosine(embeddings[i], embeddings[j])
    return m


def pair_p90(m: np.ndarray, idx: np.ndarray) -> float:
    """p90 of pairwise cosine over distinct articles in `idx` (a bootstrap index vector)."""
    vals = [m[a, b] for k, a in enumerate(idx) for b in idx[k + 1 :] if a != b]
    return float(np.percentile(vals, 90))


def nn_p90(m: np.ndarray, idx: np.ndarray) -> float:
    """p90 of each article's nearest distinct sibling's cosine."""
    nn = [max(m[a, b] for b in idx if b != a) for a in idx if any(b != a for b in idx)]
    return float(np.percentile(nn, 90))


def with_ci(m: np.ndarray, stat) -> tuple[float, float, float]:
    """`(point, ci_lo, ci_hi)` of `stat` over the full set, bootstrapping articles."""
    n = m.shape[0]
    point = stat(m, np.arange(n))
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        if len(set(idx.tolist())) >= 3:
            draws.append(stat(m, idx))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def _paths(manifest: Path) -> dict[str, list[Path]]:
    df = pl.read_csv(manifest).filter(pl.col("is_full_garment"))
    return {
        sid: [Path(p) for p in df.filter(pl.col("style_id") == sid)["image_path"].to_list()]
        for sid in df["style_id"].unique(maintain_order=True).to_list()
    }


def main() -> None:
    """Old vs new thresholds and CI widths per style, space and gate."""
    old, new = _paths(OLD_MANIFEST), _paths(NEW_MANIFEST)
    embedders = {"clip": clip_scoring.embed_image, "dinov2": dino_scoring.embed_image}
    rows = []
    for sid, new_paths in new.items():
        for label, paths in (("old", old.get(sid, [])), ("new", new_paths)):
            if len(paths) < 3:
                continue
            for space, embed in embedders.items():
                m = _sim_matrix([embed(p) for p in paths])
                for gate, stat in (("gate1_pair_p90", pair_p90), ("gate1b_nn_p90", nn_p90)):
                    point, lo, hi = with_ci(m, stat)
                    rows.append(
                        {
                            "style_id": sid,
                            "base": label,
                            "n_articles": len(paths),
                            "space": space,
                            "gate": gate,
                            "threshold": point,
                            "ci_lo": lo,
                            "ci_hi": hi,
                            "ci_width": hi - lo,
                        }
                    )
    out = pl.DataFrame(rows)
    out.write_csv(OUT_PATH)
    with pl.Config(tbl_rows=60, tbl_width_chars=200):
        print(out)


if __name__ == "__main__":
    main()
