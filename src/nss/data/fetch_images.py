"""On-demand, per-article image fetcher for the H&M competition dataset.

Downloads individual article images from Kaggle by `article_id`, instead of the full
~28.46 GiB / 105,100-file competition image bundle. The initial bulk `kaggle competitions
download` of the whole competition (images included -- there is no per-folder download
in the Kaggle API) was killed partway through once we established this project only
needs ~200 exemplar images, not the full tree.

Remote path convention (confirmed empirically against a live Kaggle API file listing via
`kaggle.api.competition_list_files`, not assumed from memory -- see e.g.
`images/010/0108775015.jpg` for article_id 108775015):
`images/<first 3 digits of the article_id zero-padded to 10 digits>/<article_id
zero-padded to 10 digits>.jpg`.

The Kaggle SDK's single-file download (`kaggle.api.competition_download_file`) writes
the file flat into the target directory (just `<article_id>.jpg`, no `images/<prefix>/`
subfolders -- confirmed by a real single-file download), so the local cache mirrors that
flat layout: `data/images/<article_id zero-padded to 10 digits>.jpg`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import kaggle

COMPETITION = "h-and-m-personalized-fashion-recommendations"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0


def remote_path(article_id: int) -> str:
    """Build the remote Kaggle competition-file path for an article_id.

    Args:
        article_id: The H&M article_id (as found in `articles.csv`).

    Returns:
        The remote file path, e.g. "images/010/0108775015.jpg".
    """
    padded = f"{article_id:010d}"
    return f"images/{padded[:3]}/{padded}.jpg"


def local_path(article_id: int, out_dir: Path) -> Path:
    """Build the local cache path for an article_id's image.

    Args:
        article_id: The H&M article_id.
        out_dir: Local directory the image cache lives in.

    Returns:
        The local file path, e.g. `out_dir/0108775015.jpg`.
    """
    return out_dir / f"{article_id:010d}.jpg"


def fetch_one(article_id: int, out_dir: Path, max_retries: int = MAX_RETRIES) -> bool:
    """Fetch a single article's image, reusing the local cache if already present.

    Retries up to `max_retries` times on failure (network error, HTTP error, missing
    remote file, etc.) with a short linear backoff before giving up on this article_id.

    Args:
        article_id: The H&M article_id to fetch.
        out_dir: Local directory to cache images into (created if missing).
        max_retries: Maximum number of download attempts before giving up.

    Returns:
        True if the image is present locally (already cached or freshly downloaded),
        False if every attempt failed.
    """
    dest = local_path(article_id, out_dir)
    if dest.exists() and dest.stat().st_size > 0:
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    remote_name = remote_path(article_id)

    for attempt in range(1, max_retries + 1):
        try:
            kaggle.api.competition_download_file(
                COMPETITION, remote_name, path=str(out_dir), force=True, quiet=True
            )
        except Exception as exc:
            if attempt == max_retries:
                print(f"FAILED {article_id} after {max_retries} attempts: {exc}", file=sys.stderr)
                return False
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue

        if dest.exists() and dest.stat().st_size > 0:
            return True

        # Call succeeded but the file didn't land (or landed empty) -- retry.
        if attempt == max_retries:
            print(
                f"FAILED {article_id}: download call succeeded but file is missing/empty",
                file=sys.stderr,
            )
            return False
        time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    return False


def fetch_images(article_ids: list[int], out_dir: Path) -> dict[str, bool]:
    """Fetch a batch of article images, one at a time, with per-id caching and retry.

    Args:
        article_ids: List of H&M article_ids to fetch.
        out_dir: Local directory to cache images into.

    Returns:
        Mapping of `str(article_id) -> success` for every requested id.
    """
    return {str(article_id): fetch_one(article_id, out_dir) for article_id in article_ids}


def main() -> None:
    """CLI entry point: fetch images for a list of article_ids.

    Article IDs can be passed via `--article-ids` (space-separated) and/or
    `--ids-file` (a text file with one article_id per line); both may be combined.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--article-ids", type=int, nargs="*", default=[], help="Article IDs to fetch"
    )
    parser.add_argument(
        "--ids-file", type=Path, default=None, help="Text file with one article_id per line"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/images"), help="Local image cache directory"
    )
    args = parser.parse_args()

    article_ids = list(args.article_ids)
    if args.ids_file is not None:
        article_ids += [
            int(line.strip()) for line in args.ids_file.read_text().splitlines() if line.strip()
        ]

    if not article_ids:
        parser.error("no article_ids provided (use --article-ids and/or --ids-file)")

    start = time.time()
    results = fetch_images(article_ids, args.out_dir)
    elapsed = time.time() - start

    n_ok = sum(results.values())
    print(f"Fetched {n_ok}/{len(results)} images in {elapsed:.1f}s -> {args.out_dir}")
    failed = [aid for aid, ok in results.items() if not ok]
    if failed:
        print(f"Failed article_ids: {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
