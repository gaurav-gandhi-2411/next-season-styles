"""The final three style ids, their order and plain names -- single source of truth (tasks M2/N6).

The final three come from `nss.models.reselect_final_three` (task N6: all three from the emerging
table, category and visual-ambiguity exclusions, no shared colour or product type): the beige
melange sweater, the red dress and the white jersey top. Order is the emerging-table growth rank.
Every figure, table and page builder imports these instead of restating style keys.
"""

from __future__ import annotations

SWEATER = "Ladieswear || Sweater || Knitwear || Beige || Melange"
DRESS = "Ladieswear || Dress || Dresses Ladies || Red || Solid"
TOP = "Ladieswear || Top || Jersey Basic || White || Solid"

SUMMER = "Ladieswear || Bikini top || Swimwear || Orange || All over pattern"

STYLE_ORDER: tuple[str, ...] = (SWEATER, DRESS, TOP)

DISPLAY_NAMES: dict[str, str] = {
    SWEATER: "Beige Melange Knitwear Sweater",
    DRESS: "Red Dress",
    TOP: "White Jersey Top",
    SUMMER: "Orange Patterned Bikini Top",
}
PLAIN_NAMES: dict[str, str] = {
    SWEATER: "Beige knit sweater",
    DRESS: "Red dress",
    TOP: "White jersey top",
    SUMMER: "Orange patterned bikini top",
}
