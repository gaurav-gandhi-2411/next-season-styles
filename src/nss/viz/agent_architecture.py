"""Render the Track D agent-delegation architecture diagram (task D2).

Draws the orchestrator, its 5 sub-agents (`data-analyst`, `forecaster`, `style-profiler`,
`concept-designer`, `critic`), the delegation/return arrows between the orchestrator and each
sub-agent, and the `critic` <-> `concept-designer` QC retry loop as an explicit two-arc cycle (not
a single straight line) so the retry mechanism -- REJECT with an adjusted parameter, capped at 2
retries before escalation -- is visually obvious rather than implied.

TOOLING CHOICE (documented per task instructions): matplotlib, not graphviz. `dot` is not on PATH
and the `graphviz` Python package is not installed in this project's `.venv` (verified at
development time; this module does not depend on either). matplotlib is already a pinned project
dependency (see `pyproject.toml`) and is what every other `reports/figures/*.png` in this repo is
generated with (`nss.viz.panel_eda`, `nss.viz.channel_check`) -- reusing it keeps this diagram
reproducible with zero new external binary dependencies, at the cost of manually laying out node
positions (acceptable for a fixed 6-node diagram).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless -- this module only writes a PNG, never shows a window.
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DEFAULT_OUT_PATH = Path("reports/figures/agent_architecture.png")

# Layout constants (data coordinates). Kept as named constants rather than inline literals so the
# hand-tuned positions below read as intentional, not magic numbers.
FIG_WIDTH, FIG_HEIGHT = 15.0, 9.5
ORCHESTRATOR_CENTER = (7.4, 8.0)
ORCHESTRATOR_SIZE = (4.4, 1.15)
SUBAGENT_Y = 4.6
SUBAGENT_SIZE = (2.2, 1.15)
SUBAGENT_X = {
    "data-analyst": 1.3,
    "forecaster": 4.1,
    "style-profiler": 7.0,
    "concept-designer": 10.0,
    "critic": 13.1,
}
SUBAGENT_ROLE = {
    "data-analyst": "historical sales\n& trend Q&A",
    "forecaster": "reads frozen\nforecast tables",
    "style-profiler": "profile -> design\nbrief (skill)",
    "concept-designer": "generate_concept\n(SDXL / Gemini)",
    "critic": "score_concept\n(Gate 1/1b/2 + human)",
}

ORCHESTRATOR_COLOR = "#2C3E50"
SUBAGENT_COLOR = "#2E86AB"
DELEGATE_COLOR = "#5D6D7E"
RETRY_COLOR = "#D35400"
ESCALATE_COLOR = "#C0392B"


def _draw_box(
    ax: plt.Axes,
    center: tuple[float, float],
    size: tuple[float, float],
    label: str,
    sublabel: str | None,
    facecolor: str,
) -> None:
    """Draw one rounded agent box with a bold name and an optional smaller role line.

    Args:
        ax: Target axes.
        center: `(x, y)` center of the box in data coordinates.
        size: `(width, height)` of the box.
        label: Bold agent-name text.
        sublabel: Optional smaller descriptive text drawn below `label`, or `None` to omit it.
        facecolor: Box fill color (hex string).
    """
    x, y = center
    w, h = size
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        facecolor=facecolor,
        edgecolor="black",
        linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(box)
    text_y = y + 0.16 if sublabel else y
    ax.text(
        x,
        text_y,
        label,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="white",
        zorder=4,
    )
    if sublabel:
        ax.text(
            x,
            y - 0.26,
            sublabel,
            ha="center",
            va="center",
            fontsize=7.5,
            color="white",
            zorder=4,
        )


def _draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    connectionstyle: str = "arc3,rad=0.0",
    linestyle: str = "-",
    label: str | None = None,
    label_offset: tuple[float, float] = (0.0, 0.0),
    fontsize: float = 7.5,
) -> None:
    """Draw one curved/straight arrow with an optional label near its midpoint.

    Args:
        ax: Target axes.
        start: Arrow tail `(x, y)`.
        end: Arrow head `(x, y)`.
        color: Arrow color (hex string).
        connectionstyle: matplotlib `FancyArrowPatch` connection style (e.g. an arc radius for
            curved arrows, used here to draw the retry loop as two visually distinct arcs).
        linestyle: Line style (`"-"` solid, `"--"` dashed for the retry/escalation paths).
        label: Optional text placed near the arrow's midpoint.
        label_offset: `(dx, dy)` nudge applied to the label position, to avoid overlapping the
            arrow line itself.
        fontsize: Label font size.
    """
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        color=color,
        linewidth=1.6,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        zorder=2,
    )
    ax.add_patch(arrow)
    if label:
        mid_x = (start[0] + end[0]) / 2 + label_offset[0]
        mid_y = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(
            mid_x,
            mid_y,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            fontweight="bold",
            zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
        )


def build_figure() -> plt.Figure:
    """Build the full agent-architecture figure: orchestrator, 5 sub-agents, delegation arrows,
    and the critic <-> concept-designer retry loop.

    Returns:
        The constructed `matplotlib.figure.Figure`, ready to save.
    """
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.set_xlim(0, FIG_WIDTH)
    ax.set_ylim(1.0, FIG_HEIGHT)
    ax.axis("off")
    ax.set_title(
        "next-season-styles -- Agent Delegation Architecture (Track D)",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    _draw_box(
        ax,
        ORCHESTRATOR_CENTER,
        ORCHESTRATOR_SIZE,
        "orchestrator",
        "receives request, delegates, aggregates -- calls no MCP tool directly",
        ORCHESTRATOR_COLOR,
    )

    ox, oy = ORCHESTRATOR_CENTER
    ow, oh = ORCHESTRATOR_SIZE
    sw, sh = SUBAGENT_SIZE
    for name, x in SUBAGENT_X.items():
        _draw_box(ax, (x, SUBAGENT_Y), SUBAGENT_SIZE, name, SUBAGENT_ROLE[name], SUBAGENT_COLOR)
        # Delegate down / return up, drawn as one double-headed arrow between box edges.
        top_of_subagent = (x, SUBAGENT_Y + sh / 2)
        bottom_of_orchestrator = (
            x if abs(x - ox) < ow / 2 else ox + (ow / 2 if x > ox else -ow / 2),
            oy - oh / 2,
        )
        arrow = FancyArrowPatch(
            bottom_of_orchestrator,
            top_of_subagent,
            arrowstyle="<|-|>",
            mutation_scale=13,
            color=DELEGATE_COLOR,
            linewidth=1.4,
            zorder=2,
        )
        ax.add_patch(arrow)

    # Explicit critic -> concept-designer retry cycle: two opposite-bulging arcs BELOW both boxes
    # (rather than one straight line, and kept clear of the top-edge delegate/escalate arrows) so
    # the loop reads as a cycle at a glance.
    cd_x = SUBAGENT_X["concept-designer"]
    cr_x = SUBAGENT_X["critic"]
    cd_bottom = (cd_x + 0.15, SUBAGENT_Y - sh / 2)
    cr_bottom = (cr_x - 0.15, SUBAGENT_Y - sh / 2)

    _draw_arrow(
        ax,
        cd_bottom,
        cr_bottom,
        color=DELEGATE_COLOR,
        connectionstyle="arc3,rad=-0.3",
        label="candidate image + params",
        label_offset=(0.0, -0.72),
        fontsize=7.5,
    )
    _draw_arrow(
        ax,
        cr_bottom,
        cd_bottom,
        color=RETRY_COLOR,
        connectionstyle="arc3,rad=-0.75",
        linestyle="--",
        label="REJECT: adjusted param (max 2 retries)",
        label_offset=(0.0, -1.35),
        fontsize=7.5,
    )

    # Escalation path: critic -> orchestrator, dashed red, only fires after the retry cap. Routed
    # further right and with a wider curve than the plain delegate/return line so the two never
    # visually merge.
    _draw_arrow(
        ax,
        (cr_x + sw / 2 - 0.1, SUBAGENT_Y + sh / 2),
        (ox + ow / 2 + 0.05, oy - oh / 2 + 0.1),
        color=ESCALATE_COLOR,
        connectionstyle="arc3,rad=0.4",
        linestyle="--",
        label="escalate: QC FAILED\nafter 2 retries",
        label_offset=(1.35, -0.1),
        fontsize=7.5,
    )

    # Legend.
    legend_x, legend_y = 0.3, 1.55
    legend_entries = [
        (DELEGATE_COLOR, "-", "delegate / return"),
        (RETRY_COLOR, "--", "critic REJECT -> retry (capped at 2)"),
        (ESCALATE_COLOR, "--", "escalate to orchestrator after cap exhausted"),
    ]
    for i, (color, style, text) in enumerate(legend_entries):
        y = legend_y - i * 0.32
        ax.plot([legend_x, legend_x + 0.5], [y, y], color=color, linestyle=style, linewidth=2.0)
        ax.text(legend_x + 0.65, y, text, fontsize=8.5, va="center")

    fig.tight_layout()
    return fig


def main(out_path: Path = DEFAULT_OUT_PATH) -> Path:
    """Build and save the agent-architecture diagram.

    Args:
        out_path: Destination PNG path; parent directories are created if missing.

    Returns:
        `out_path`, for convenience chaining.
    """
    fig = build_figure()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


if __name__ == "__main__":
    main()
