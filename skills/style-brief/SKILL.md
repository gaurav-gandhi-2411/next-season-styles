---
name: style-brief
description: Convert a generic style profile (attributes, historical/predicted performance signal, ranked feature drivers, reference images) into a structured JSON design brief and a rendered text-to-image prompt. Use when a winning product/style needs to become an image-generation brief that deliberately preserves what made it a winner while introducing bounded, tasteful novelty axes.
---

# Style Brief

Turns "this style won" (a row of attributes + a performance signal + why the model thinks it
won) into "here is what a design concept derived from it should preserve, what it should vary,
and the actual prompt text to generate it." The skill is dataset-agnostic: it never hardcodes a
specific product taxonomy's column names. Any catalogue (fashion, footwear, home goods,
consumer packaged goods with a "style"/SKU-family concept) can use it by writing a small adapter
that maps its own schema onto the generic `StyleProfile` input below.

## When to use this skill

- A forecasting/ranking pipeline has surfaced a winning style (or product concept) and the next
  step is "generate a novel-but-related design concept" via text-to-image or similar generative
  tooling.
- You need the generation prompt to be traceable back to *why* the source style won (its
  attributes, its performance drivers) rather than hand-written ad hoc.
- You need an explicit, auditable split between "must stay the same" (`preserve`) and "should
  differ" (`change`) axes, so a downstream reviewer (or a novelty/fidelity scoring gate) can
  check the generated concept against the brief mechanically.

## Input schema: `StyleProfile`

A plain JSON-serializable object. Field names are generic on purpose — they describe *roles*
(what a garment/product's shape is called, what its material family is, what its color is),
never a specific dataset's literal column name.

```jsonc
{
  // Opaque identifier for the style. Any string; not interpreted.
  "style_id": "<string>",

  // Garment/product-identity attributes. All four are required; free-text values.
  "attributes": {
    "garment_category": "<broad product type, e.g. 'T-shirt', 'Sweater', 'Sneaker', 'Mug'>",
    "construction_group": "<material/construction family, e.g. 'Jersey Basic', 'Knitwear'>",
    "colour_name": "<dominant colour name, e.g. 'Black', 'Beige'>",
    "pattern_or_finish": "<surface pattern/finish, e.g. 'Solid', 'Melange', 'Striped'>"
  },

  // Historical/predicted performance signal for this style. `growth_ratio` may be null when a
  // trailing trend isn't defined for this style (e.g. it's the top style by absolute level, not
  // by growth).
  "performance_signal": {
    "predicted_intensity": 33.7,
    "growth_ratio": null
  },

  // Free-text verdict on WHY the style is predicted to perform -- already synthesized by
  // whatever upstream analysis produced it (e.g. a SHAP driver read-out). This skill does not
  // compute this; it only checks whether "persist" or "season" occurs FIRST in the narrative
  // (the convention: the leading/dominant mechanism is named first, e.g. "persistence
  // dominates: ..." or "seasonal recovery dominates over persistence: ...") to decide whether
  // the brief should treat the style's identity itself as the thing preserving performance, or
  // whether a seasonal/occasion cue is doing the work. Neither word present defaults to the
  // persistence framing (conservative).
  "dominant_mechanism": "<free-text narrative, e.g. 'persistence (lag_1) dominates: SHAP=0.49...'>",

  // Ranked feature drivers, in descending importance order. Passed through for provenance;
  // `feature` labels are opaque strings (may be raw model feature names -- the skill does not
  // interpret them beyond `dominant_mechanism`).
  "top_drivers": [
    {"feature": "<label>", "importance": 0.49}
  ],

  // Local or remote paths/URLs to exemplar images of the style, for a human or an
  // image-to-image/ControlNet step to reference. Not consumed by the text synthesis itself.
  "reference_image_paths": ["<path or URL>", "..."],

  // Optional freeform tag for categories needing tasteful handling (e.g. "intimate_apparel").
  // The skill's OUTPUT is already garment/product-focused by default regardless of this tag (see
  // Design notes below) -- this field exists for downstream consumers (e.g. a generation-time
  // guard enforcing a flat-lay/product-photography framing constraint) rather than to change the
  // brief's own text. May be null.
  "sensitivity_tag": "<string or null>"
}
```

## Output schema: `DesignBrief`

```jsonc
{
  "style_id": "<pass-through from input>",
  "silhouette": "<garment shape/fit description>",
  "fabric_and_hand": "<material, texture, drape/weight qualities>",
  "colour_direction": "<the winning colour plus an adjacent/complementary suggestion>",
  "detail_and_graphic_treatment": "<surface treatment, trims, graphic elements>",
  "preserve": ["<attribute or narrative that made this style a winner>", "..."],
  "change": ["<novelty axis -- a dimension a derived concept should meaningfully vary>", "..."],
  "rendered_prompt": "<the actual text-to-image prompt string, synthesized from the above>",
  "negative_prompt": "<things to explicitly avoid>"
}
```

`preserve` and `change` are deliberately disjoint: everything in `preserve` is an identity
attribute (category, construction family, colour, surface treatment, silhouette, and — when
`dominant_mechanism` signals persistence rather than seasonality — the winning combination as a
whole). `change` never touches those; it lists bounded novelty axes (a motif/print accent, a
trim/construction detail, a small proportion tweak) that a derived concept should vary instead.

## Design notes

- **No human-model framing, by default, for every style.** `rendered_prompt` describes the
  garment/product itself (silhouette, fabric, colour, surface treatment) and closes with
  "product photography of the garment itself, clean studio background" — never "worn by a
  model." `negative_prompt` explicitly excludes human models, faces, and skin. This is a
  general design-brief-writing choice (a brief describes the product, not a photoshoot cast),
  not a special case for any one category — it happens to also satisfy sensitive-category
  handling (see the underwear example below) without any category-specific branching.
- **Novelty axes are chosen to not collide with `preserve`.** If a style's surface treatment is
  already a strong print, the `change` list's graphic-motif axis says to skip itself rather than
  compounding the existing print.
- **`dominant_mechanism` changes what `preserve` argues for, not what it lists.** Every brief
  preserves the same four identity attributes plus silhouette; what differs is the closing
  sentence: a persistence-dominated style additionally preserves "the winning combination as a
  whole" (a reinterpretation risks losing the demand driver), while a seasonally-dominated style
  instead preserves "the seasonal/occasion cue" driving it.

## Worked example 1 — persistence-driven basic (no special handling needed)

Input `StyleProfile` (values from a real H&M forecasting run's winning T-shirt style, adapted by
the calling code's H&M-specific mapper — see "Adapting a new dataset" below for what that mapper
does):

```json
{
  "style_id": "Ladieswear || T-shirt || Jersey Basic || Black || Solid",
  "attributes": {
    "garment_category": "T-shirt",
    "construction_group": "Jersey Basic",
    "colour_name": "Black",
    "pattern_or_finish": "Solid"
  },
  "performance_signal": {"predicted_intensity": 33.68792198187379, "growth_ratio": null},
  "dominant_mechanism": "persistence (lag_1) dominates over seasonal: lag_1 SHAP=0.7451 vs. fourier_sin_1 SHAP=-0.1043 (also the single largest driver overall)",
  "top_drivers": [
    {"feature": "lag_1", "importance": 0.7451153123173221},
    {"feature": "n_active_articles_level", "importance": 0.212953116355365},
    {"feature": "perceived_colour_master_name", "importance": 0.11955561709379584},
    {"feature": "fourier_sin_1", "importance": -0.1042796712395504},
    {"feature": "garment_group_name", "importance": 0.10111252504372029}
  ],
  "reference_image_paths": [
    "data/images/0800691008.jpg", "data/images/0554598001.jpg", "data/images/0778064001.jpg",
    "data/images/0827968004.jpg", "data/images/0865076001.jpg", "data/images/0767862001.jpg",
    "data/images/0843685001.jpg"
  ],
  "sensitivity_tag": null
}
```

Output `DesignBrief` (actual `generate_design_brief(...)` output for the input above):

```json
{
  "style_id": "Ladieswear || T-shirt || Jersey Basic || Black || Solid",
  "silhouette": "Jersey Basic t-shirt: relaxed, straight-body silhouette with a simple crew or scoop neckline. No directional fit change is signalled by the source data -- silhouette is a `preserve`, not a `change`, axis.",
  "fabric_and_hand": "Jersey Basic fabrication with a soft, stretch single-knit jersey hand with a relaxed, body-skimming drape.",
  "colour_direction": "Black as the anchor colour (the attribute that won); charcoal or ink navy as an optional adjacent/complementary accent, not a replacement for the anchor.",
  "detail_and_graphic_treatment": "clean solid-ground treatment with no print or graphic; surface interest, if any, comes from construction details (ribbing, seaming, trims) rather than graphics",
  "preserve": [
    "garment category: T-shirt",
    "construction/fabrication family: Jersey Basic",
    "anchor colour: Black",
    "surface treatment: Solid",
    "overall silhouette proportions (relaxed, straight-body silhouette with a simple crew or scoop neckline)",
    "the winning combination as a whole -- performance is driven by persistence of demand for this exact combination, not a seasonal/calendar cue, so the concept must read as recognisably the same style rather than a reinterpretation of a seasonal theme"
  ],
  "change": [
    "one subtle graphic motif, print placement, or embroidery accent not present in the source style (skip this axis if the source is already a strong graphic/print treatment)",
    "a trim or construction detail (topstitch colour, binding, rib width, hardware finish)",
    "a small proportion tweak within the category's normal range (hem length, cuff width, rise)"
  ],
  "rendered_prompt": "T-shirt, jersey basic construction. Silhouette: relaxed, straight-body silhouette with a simple crew or scoop neckline. Fabric: a soft, stretch single-knit jersey hand with a relaxed, body-skimming drape. Colour: Black, anchor tone (optional accent: charcoal or ink navy). Surface treatment: clean solid-ground treatment with no print or graphic; surface interest, if any, comes from construction details (ribbing, seaming, trims) rather than graphics. Novel accents to introduce: one subtle graphic motif, print placement, or embroidery accent not present in the source style (skip this axis if the source is already a strong graphic/print treatment); a trim or construction detail (topstitch colour, binding, rib width, hardware finish); a small proportion tweak within the category's normal range (hem length, cuff width, rise). Product photography of the garment itself, clean studio background, even lighting, no styling props.",
  "negative_prompt": "blurry, distorted proportions, extra limbs, warped seams, low-resolution, watermark, text overlay, logo, duplicate garments, mismatched colourway, worn by a human model, face, skin, lifestyle photography"
}
```

## Worked example 2 — persistence-driven, sensitive category (underwear)

This is the case where the plausible narrative ("a red underwear style must be Christmas-window
seasonal") is checked against the actual driver data and turns out to be wrong: `lag_1`
(persistence) dominates, not any seasonal/fourier term. The brief below reflects what the data
actually shows, not the plausible-sounding story.

Input `StyleProfile`:

```json
{
  "style_id": "Ladieswear || Underwear bottom || Under-, Nightwear || Red || Solid",
  "attributes": {
    "garment_category": "Underwear bottom",
    "construction_group": "Under-, Nightwear",
    "colour_name": "Red",
    "pattern_or_finish": "Solid"
  },
  "performance_signal": {"predicted_intensity": 16.247146540920095, "growth_ratio": 3.6715010706959976},
  "dominant_mechanism": "persistence (lag_1) dominates: SHAP=0.4880 (also the single largest driver overall); no seasonal fourier/lag_52 term appears in the top-5 at all",
  "top_drivers": [
    {"feature": "lag_1", "importance": 0.4879718268272778},
    {"feature": "perceived_colour_master_name", "importance": -0.12350384274081928},
    {"feature": "n_active_articles_level", "importance": 0.10687098439354158},
    {"feature": "slope_13w", "importance": 0.06188135329019789},
    {"feature": "share_garment_group", "importance": 0.06170162968258172}
  ],
  "reference_image_paths": [
    "data/images/0803986005.jpg", "data/images/0854384005.jpg", "data/images/0822311003.jpg",
    "data/images/0798407002.jpg", "data/images/0826955010.jpg", "data/images/0736531016.jpg",
    "data/images/0798416002.jpg", "data/images/0841185002.jpg"
  ],
  "sensitivity_tag": "intimate_apparel"
}
```

Output `DesignBrief` (actual `generate_design_brief(...)` output for the input above — note the
`rendered_prompt` stays product/garment-focused with no human-model language, the same as every
other style, simply because that is how this skill always writes a brief):

```json
{
  "style_id": "Ladieswear || Underwear bottom || Under-, Nightwear || Red || Solid",
  "silhouette": "Under-, Nightwear underwear bottom: brief/hipster-style silhouette, low- to mid-rise, following the body's natural line without structural embellishment. No directional fit change is signalled by the source data -- silhouette is a `preserve`, not a `change`, axis.",
  "fabric_and_hand": "Under-, Nightwear fabrication with a lightweight, skin-friendly hand prioritising comfort and breathability over structure.",
  "colour_direction": "Red as the anchor colour (the attribute that won); burgundy or brick as an optional adjacent/complementary accent, not a replacement for the anchor.",
  "detail_and_graphic_treatment": "clean solid-ground treatment with no print or graphic; surface interest, if any, comes from construction details (ribbing, seaming, trims) rather than graphics",
  "preserve": [
    "garment category: Underwear bottom",
    "construction/fabrication family: Under-, Nightwear",
    "anchor colour: Red",
    "surface treatment: Solid",
    "overall silhouette proportions (brief/hipster-style silhouette, low- to mid-rise, following the body's natural line without structural embellishment)",
    "the winning combination as a whole -- performance is driven by persistence of demand for this exact combination, not a seasonal/calendar cue, so the concept must read as recognisably the same style rather than a reinterpretation of a seasonal theme"
  ],
  "change": [
    "one subtle graphic motif, print placement, or embroidery accent not present in the source style (skip this axis if the source is already a strong graphic/print treatment)",
    "a trim or construction detail (topstitch colour, binding, rib width, hardware finish)",
    "a small proportion tweak within the category's normal range (hem length, cuff width, rise)"
  ],
  "rendered_prompt": "Underwear bottom, under-, nightwear construction. Silhouette: brief/hipster-style silhouette, low- to mid-rise, following the body's natural line without structural embellishment. Fabric: a lightweight, skin-friendly hand prioritising comfort and breathability over structure. Colour: Red, anchor tone (optional accent: burgundy or brick). Surface treatment: clean solid-ground treatment with no print or graphic; surface interest, if any, comes from construction details (ribbing, seaming, trims) rather than graphics. Novel accents to introduce: one subtle graphic motif, print placement, or embroidery accent not present in the source style (skip this axis if the source is already a strong graphic/print treatment); a trim or construction detail (topstitch colour, binding, rib width, hardware finish); a small proportion tweak within the category's normal range (hem length, cuff width, rise). Product photography of the garment itself, clean studio background, even lighting, no styling props.",
  "negative_prompt": "blurry, distorted proportions, extra limbs, warped seams, low-resolution, watermark, text overlay, logo, duplicate garments, mismatched colourway, worn by a human model, face, skin, lifestyle photography"
}
```

Note: a later generation-time step is expected to layer a mandatory flat-lay/mannequin
product-photography framing constraint on top of `rendered_prompt` for sensitive categories
(identified via `sensitivity_tag`) — that enforcement is out of scope for this skill and lives
in whatever calls the image generator, not here. This skill's job stops at producing a brief
that is already tasteful and product-focused on its own terms.

## Adapting a new dataset

Write a small adapter (calling code, not part of this skill) that:

1. Maps the dataset's own product-taxonomy columns onto the four `attributes` keys
   (`garment_category`, `construction_group`, `colour_name`, `pattern_or_finish`). For a
   non-fashion catalogue, pick the closest analogues (e.g. `construction_group` might be a
   material or manufacturing process family).
2. Maps whatever performance/forecast signal the dataset already computed onto
   `performance_signal.predicted_intensity` / `growth_ratio`.
3. Supplies `dominant_mechanism` as a pre-synthesized free-text string (this skill does not
   compute feature attribution itself — it only reads the resulting narrative).
4. Passes through `top_drivers` and `reference_image_paths` as-is.
5. Calls `generate_brief.generate_design_brief(profile)` and persists the result.

`skills/style-brief/generate_brief.py`'s `generate_design_brief` function is the only entry
point calling code needs; `validate_style_profile` is exposed separately for callers that want
to validate a batch of profiles before generating briefs for all of them.
