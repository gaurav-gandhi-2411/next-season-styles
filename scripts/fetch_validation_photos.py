"""Fetch the Q2 validation photos and the validation gallery, serially, with 429 cool-downs.

Two sets, both written only to new scratch folders under `data/`:

1. the 40 evaluation photos (`concept_forecast_validation.sample_images()`), missing ones into
   `data/images_q2_eval/` (never into the index);
2. GALLERY photos for the evaluation styles: up to `--gallery-per-style` OTHER real articles of each
   evaluation style (never the evaluation article itself), into `data/images_q2/` (the index
   folder). This is the standard closed-set retrieval protocol -- the gallery must contain the
   classes being queried -- and is what the `gallery_covers_eval` validation condition uses; the
   deployment-coverage condition ignores the difference and reports coverage as it is on disk.

Kaggle answers HTTP 429 when it is hammered (seen with 16 concurrent workers, and the limit then
persisted for 25+ minutes), so this fetches ONE photo at a time and sleeps after consecutive
failures. The first successful/failed attempt times are printed so the fetch cost is recorded.

Usage:
    python scripts/fetch_validation_photos.py --budget-seconds 1800
"""

from __future__ import annotations

import argparse
import time

from nss.data.fetch_images import fetch_one, local_path
from nss.generate import concept_forecast_index as cfi
from nss.generate import concept_forecast_validation as v

COOLDOWN_SECONDS = 90
MAX_CONSECUTIVE_FAILURES = 3


def main() -> None:
    """Fetch until everything is on disk or the wall-clock budget is spent."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget-seconds", type=float, default=1800.0)
    ap.add_argument("--gallery-per-style", type=int, default=2)
    args = ap.parse_args()
    reps = v._sample_reps()
    jobs: list[tuple[int, object]] = [
        (a, v.EVAL_FETCH_DIR)
        for a in reps["article_id"].to_list()
        if not local_path(a, v.IMAGES_DIR).exists()
    ]
    jobs += [(a, cfi.IMAGE_DIRS[1]) for a in v.gallery_article_ids(args.gallery_per_style)]
    jobs = [(a, d) for a, d in jobs if not local_path(a, d).exists()]  # type: ignore[arg-type]
    print(f"to fetch: {len(jobs)} photos", flush=True)
    start = time.time()
    fails = 0
    n_ok = 0
    while jobs and time.time() - start < args.budget_seconds:
        a, d = jobs[0]
        ok = fetch_one(a, d, max_retries=1)  # type: ignore[arg-type]
        if ok:
            jobs.pop(0)
            n_ok += 1
            fails = 0
        else:
            fails += 1
            if fails >= MAX_CONSECUTIVE_FAILURES:
                print(f"{fails} consecutive failures; sleeping {COOLDOWN_SECONDS}s", flush=True)
                time.sleep(COOLDOWN_SECONDS)
                fails = 0
        print(f"{a} ok={ok} elapsed={time.time() - start:.0f}s remaining={len(jobs)}", flush=True)
    print(
        f"DONE fetched={n_ok} remaining={len(jobs)} elapsed={time.time() - start:.0f}s", flush=True
    )


if __name__ == "__main__":
    main()
