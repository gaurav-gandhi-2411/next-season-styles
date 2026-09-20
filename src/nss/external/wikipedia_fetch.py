"""Fetch daily English-Wikipedia pageviews for product-type and colour articles, into a disk cache.

Source: the official Wikimedia REST API (`/metrics/pageviews/per-article`), user traffic, all
access methods, daily, 2018-09-01 .. 2020-09-30. No credentials are needed; a descriptive
User-Agent is required by the API's policy. It is not rate-limit-free in practice: 0.2 s
spacing hit HTTP 429 at the 11th request, hence the backoff and 1 s spacing below.

This is a much less direct signal than search: an encyclopedia article exists per product type
("Sweater") and per colour ("Beige"), not per colour+product style, so every style of a product
type shares one series. It is evaluated in F1 for comparison; the model experiment uses one
source only (see `reports/EXPERIMENT_external_signals.md`).

Each article is cached to `data/external/wikipedia/<slug>.csv`; existing files are skipped. An
article title that returns 404 is recorded as missing rather than guessed at.

    uv run --no-sync python -m nss.external.wikipedia_fetch
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl

CACHE_DIR = Path("data/external/wikipedia")
MAX_TRIES = 5
SPACING_S = 1.0
API = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/"
    "all-access/user/{title}/daily/20180901/20200930"
)
USER_AGENT = (
    "next-season-styles-research/0.1 (portfolio project; contact via GitHub gaurav-gandhi-2411)"
)

# H&M product_type_name -> English Wikipedia article title (canonical titles, not redirects:
# redirect titles only count views of the redirect itself). Absent = no article assigned.
PRODUCT_ARTICLE: dict[str, str] = {
    "Sweater": "Sweater",
    "Dress": "Dress",
    "T-shirt": "T-shirt",
    "Trousers": "Trousers",
    "Shirt": "Shirt",
    "Shorts": "Shorts",
    "Skirt": "Skirt",
    "Blouse": "Blouse",
    "Hoodie": "Hoodie",
    "Jacket": "Jacket",
    "Underwear bottom": "Underwear",
    "Socks": "Sock",
    "Leggings/Tights": "Leggings",
    "Blazer": "Blazer",
    "Bra": "Bra",
    "Cardigan": "Cardigan (sweater)",
    "Scarf": "Scarf",
    "Pyjama set": "Pajamas",
    "Bikini top": "Bikini",
    "Swimwear bottom": "Bikini",
    "Bag": "Bag",
    "Jumpsuit/Playsuit": "Jumpsuit",
    "Sneakers": "Sneakers",
    "Bodysuit": "Bodysuit",
    "Swimsuit": "Swimsuit",
    "Sunglasses": "Sunglasses",
    "Coat": "Coat (clothing)",
    "Underwear Tights": "Tights",
    "Belt": "Belt (clothing)",
    "Polo shirt": "Polo shirt",
    "Earring": "Earring",
    "Earrings": "Earring",
    "Boots": "Boot",
    "Sandals": "Sandal",
    "Hat/brim": "Hat",
    "Necklace": "Necklace",
    "Cap/peaked": "Baseball cap",
    "Gloves": "Glove",
    "Dungarees": "Dungarees",
    "Night gown": "Nightgown",
    "Bracelet": "Bracelet",
    "Watch": "Watch",
    "Sarong": "Sarong",
    "Tie": "Necktie",
    "Slippers": "Slipper",
    "Robe": "Robe",
    "Wallet": "Wallet",
    "Flip flop": "Flip-flops",
}

COLOUR_ARTICLE: dict[str, str] = {
    "Black": "Black",
    "Blue": "Blue",
    "White": "White",
    "Grey": "Gray",
    "Pink": "Pink",
    "Red": "Red",
    "Beige": "Beige",
    "Yellow": "Yellow",
    "Green": "Green",
    "Khaki green": "Khaki",
    "Orange": "Orange (colour)",
    "Brown": "Brown",
    "Turquoise": "Turquoise (color)",
    "Lilac Purple": "Lilac (color)",
}


def slug(title: str) -> str:
    """Filesystem-safe cache name for an article title."""
    return "".join(c if c.isalnum() else "_" for c in title.lower()).strip("_")


def fetch_article(title: str) -> pl.DataFrame | None:
    """Daily views for one article, or None if the article does not exist (HTTP 404)."""
    url = API.format(title=urllib.parse.quote(title.replace(" ", "_"), safe=""))
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    for attempt in range(1, MAX_TRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as resp:  # noqa: S310
                items = json.loads(resp.read())["items"]
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code == 429 and attempt < MAX_TRIES:
                # Seen in practice at 0.2 s spacing (request 11) despite nominal limits.
                wait = float(exc.headers.get("Retry-After", 10 * attempt))
                print(f"  429 on {title!r}; sleeping {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
            raise
    return pl.DataFrame(
        {
            "day": [pl.Series([i["timestamp"][:8]]).str.to_date("%Y%m%d")[0] for i in items],
            "views": [int(i["views"]) for i in items],
        }
    )


def main() -> None:
    """Fetch every uncached article; write missing titles to `_missing.csv`."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    titles = sorted({*PRODUCT_ARTICLE.values(), *COLOUR_ARTICLE.values()})
    missing: list[str] = []
    t0 = time.time()
    for title in titles:
        path = CACHE_DIR / f"{slug(title)}.csv"
        if path.exists():
            continue
        df = fetch_article(title)
        if df is None:
            missing.append(title)
            print(f"MISSING (404): {title}")
            continue
        df.write_csv(path)
        print(f"{title}: {df.height} days, total views {int(df['views'].sum())}")
        time.sleep(SPACING_S)
    if missing:
        (CACHE_DIR / "_missing.csv").write_text("\n".join(missing) + "\n", encoding="utf-8")
    print(f"{len(titles)} titles, {len(missing)} missing, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
