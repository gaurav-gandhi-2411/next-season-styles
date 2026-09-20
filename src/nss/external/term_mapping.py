"""Map style attributes to public-search query terms, and say plainly which styles cannot be mapped.

A style's search term is "<colour word> <product word>", e.g. Beige + Sweater -> "beige sweater".
Only attributes with an unambiguous everyday search word are mapped. A product type or colour
with no such word (H&M catch-alls such as "Other accessories" or "Garment Set", H&M-specific
colours such as "Mole") is left UNMAPPED and the style gets no signal: no term is invented to
force coverage.

The graphical appearance (Solid, Stripe, Melange, ...) is deliberately not part of the term.
Pattern words make queries too sparse for Google Trends, and it means several styles share one
term (styles that differ only in pattern get the same signal). That is a real resolution limit,
reported in `reports/EXPERIMENT_external_signals.md`.
"""

from __future__ import annotations

import polars as pl

# H&M product_type_name -> everyday search word. Absent = unmapped (no invented substitute).
PRODUCT_WORD: dict[str, str] = {
    "Sweater": "sweater",
    "Dress": "dress",
    "T-shirt": "t-shirt",
    "Trousers": "trousers",
    "Top": "top",
    "Vest top": "vest top",
    "Shirt": "shirt",
    "Shorts": "shorts",
    "Skirt": "skirt",
    "Blouse": "blouse",
    "Hoodie": "hoodie",
    "Jacket": "jacket",
    "Underwear bottom": "underwear",
    "Socks": "socks",
    "Leggings/Tights": "leggings",
    "Swimwear bottom": "bikini bottoms",
    "Blazer": "blazer",
    "Bra": "bra",
    "Cardigan": "cardigan",
    "Scarf": "scarf",
    "Pyjama set": "pyjamas",
    "Bikini top": "bikini top",
    "Hair/alice band": "headband",
    "Bag": "bag",
    "Hat/beanie": "beanie",
    "Jumpsuit/Playsuit": "jumpsuit",
    "Sneakers": "sneakers",
    "Bodysuit": "bodysuit",
    "Swimsuit": "swimsuit",
    "Sunglasses": "sunglasses",
    "Coat": "coat",
    "Underwear Tights": "tights",
    "Belt": "belt",
    "Polo shirt": "polo shirt",
    "Pyjama bottom": "pyjama bottoms",
    "Earring": "earrings",
    "Earrings": "earrings",
    "Boots": "boots",
    "Ballerinas": "ballet flats",
    "Sandals": "sandals",
    "Hat/brim": "hat",
    "Necklace": "necklace",
    "Cap/peaked": "cap",
    "Gloves": "gloves",
    "Pumps": "pumps",
    "Heeled sandals": "heeled sandals",
    "Dungarees": "dungarees",
    "Night gown": "nightgown",
    "Hair clip": "hair clip",
    "Swimwear set": "bikini",
    "Bracelet": "bracelet",
    "Watch": "watch",
    "Sarong": "sarong",
    "Tie": "tie",
    "Tailored Waistcoat": "waistcoat",
    "Slippers": "slippers",
    "Hair ties": "hair ties",
    "Robe": "robe",
    "Beanie": "beanie",
    "Wallet": "wallet",
    "Flip flop": "flip flops",
    "Heels": "heels",
}

# H&M perceived_colour_master_name -> everyday colour word. Absent = unmapped.
COLOUR_WORD: dict[str, str] = {
    "Black": "black",
    "Blue": "blue",
    "White": "white",
    "Grey": "grey",
    "Pink": "pink",
    "Red": "red",
    "Beige": "beige",
    "Yellow": "yellow",
    "Green": "green",
    "Khaki green": "khaki",
    "Orange": "orange",
    "Brown": "brown",
    "Turquoise": "turquoise",
    "Lilac Purple": "lilac",
}

COLOUR_COL = "perceived_colour_master_name"
PRODUCT_COL = "product_type_name"


def map_styles(styles: pl.DataFrame) -> pl.DataFrame:
    """One row per style_key with `term` (null if unmapped) and `unmapped_reason`.

    Args:
        styles: A frame with `style_key`, `product_type_name` and `perceived_colour_master_name`
            (duplicates allowed; the first row per style_key is used).
    """
    base = styles.select("style_key", PRODUCT_COL, COLOUR_COL).unique("style_key", keep="first")
    rows = []
    for key, product, colour in base.sort("style_key").iter_rows():
        product_word = PRODUCT_WORD.get(product)
        colour_word = COLOUR_WORD.get(colour)
        if product_word is None and colour_word is None:
            reason = "product type and colour both unmapped"
        elif product_word is None:
            reason = "product type unmapped"
        elif colour_word is None:
            reason = "colour unmapped"
        else:
            reason = None
        term = f"{colour_word} {product_word}" if reason is None else None
        rows.append({"style_key": key, "term": term, "unmapped_reason": reason})
    return pl.DataFrame(
        rows, schema={"style_key": pl.String, "term": pl.String, "unmapped_reason": pl.String}
    )
