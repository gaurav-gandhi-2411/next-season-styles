"""The final three style ids, their order and plain names -- single source of truth (task M2/M4).

The final three come from `nss.models.reselect_final_three` (one incumbent, two emerging, colour-
distinct, no intimates): the black jersey T-shirt, the beige melange sweater and the red dress.
Order is the selection order (incumbent first, then emerging by rank). Every figure, table and page
builder imports these instead of restating style keys.
"""

from __future__ import annotations

TSHIRT = "Ladieswear || T-shirt || Jersey Basic || Black || Solid"
SWEATER = "Ladieswear || Sweater || Knitwear || Beige || Melange"
DRESS = "Ladieswear || Dress || Dresses Ladies || Red || Solid"

STYLE_ORDER: tuple[str, ...] = (TSHIRT, SWEATER, DRESS)

DISPLAY_NAMES: dict[str, str] = {
    TSHIRT: "Black Jersey Basic T-Shirt",
    SWEATER: "Beige Melange Knitwear Sweater",
    DRESS: "Red Dress",
}
PLAIN_NAMES: dict[str, str] = {
    TSHIRT: "Black jersey T-shirt",
    SWEATER: "Beige knit sweater",
    DRESS: "Red dress",
}
