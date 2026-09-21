# Agent workflow, fully live

Every tool call below is a real MCP call over stdio (`ClientSession.call_tool` against a live `nss.mcp_server` subprocess; generation on the GPU server, everything else on a CPU server), with the calling agent's allowlist from `agents/*.md` enforced by the driver. Nothing is replayed. `INPUT` = a fixed human-authored input; `LOCAL` = an in-process pure function (prompt builder, the critic's decision rule, the router); `NOT PERFORMED` = the human visual check, which is a person's judgment and is not done here. Rule committed before the run: `reports/v3/PREREGISTRATION.md`, H3 / 4a.

**This run is not evidence about the submitted concepts.** The MCP `generate_concept` tool calls `backends.generate_concept`, which exposes no negative prompt and no concat reference mode, and does not use the weighted compel prompt; the deliverables were made by `levers.generate_variant` (concat mode, negative prompt, weighted prompt). The images here are a different, weaker configuration, and none is a candidate for the submission.

Wall time for the whole run: 336 s. Cost: $0 (local).

## Step 1 -- orchestrator delegates to `forecaster`

**[LIVE] (0.1 s)** **[LIVE] forecaster** calls MCP tool `forecast_styles` (real call over the MCP stdio protocol -- `ClientSession.call_tool` against a live `nss.mcp_server` subprocess):

Request:
```json
{
  "origin_date": "2020-09-21",
  "horizon_weeks": 13,
  "table": "emerging",
  "top_n": 10
}
```

Response (projected to rank, style, intensity, growth ratio and guard flags):
```json
[
  {
    "rank": 1,
    "style_key": "Ladieswear || Underwear bottom || Under-, Nightwear || Red || Solid",
    "predicted_intensity": 16.25523950275885,
    "growth_ratio": 3.6733299037148455,
    "guard1_pass": true,
    "guard2_pass": true,
    "guard3_pass": true
  },
  {
    "rank": 2,
    "style_key": "Ladieswear || Sweater || Knitwear || Beige || Melange",
    "predicted_intensity": 17.783148281064506,
    "growth_ratio": 1.5858516618457832,
    "guard1_pass": true,
    "guard2_pass": true,
    "guard3_pass": true
  },
  {
    "rank": 3,
    "style_key": "Ladieswear || Dress || Dresses Ladies || Red || Solid",
    "predicted_intensity": 10.996816441902132,
    "growth_ratio": 1.4122021450859839,
    "guard1_pass": true,
    "guard2_pass": true,
    "guard3_pass": true
  },
  {
    "rank": 4,
    "style_key": "Ladieswear || Sweater || Knitwear || Grey || Melange",
    "predicted_intensity": 10.51041900701676,
    "growth_ratio": 1.170933656620168,
    "guard1_pass": true,
    "guard2_pass": true,
    "guard3_pass": true
  },
  {
    "rank": 5,
    "style_key": "Ladieswear || Top || Jersey Basic || White || Solid",
    "predicted_intensity": 13.853842573439003,
    "growth_ratio": 1.168139356707085,
    "guard1_pass": true,
    "guard2_pass": true,
    "guard3_pass": true
  },
  {
    "rank": 6,
    "style_key": "Divided || Jacket || Outdoor || Black || Solid",
    "predicted_intensity": 10.83696526190775,
    "growth_ratio": 1.0922741668356215,
    "guard1_pass": true,
    "guard2_pass": true,
    "guard3_pass": true
  },
  {
    "rank": 7,
    "style_key": "Ladieswear || Top || Jersey Basic || Black || Solid",
    "predicted_intensity": 26.806200859583825,
    "growth_ratio": 1.0727944363461772,
    "guard1_pass": true,
    "guard2_pass": true,
    "guard3_pass": true
  },
  {
    "rank": 8,
    "style_key": "Ladieswear || Sweater || Knitwear || Mole || Melange",
    "predicted_intensity": 15.793351132507922,
    "growth_ratio": 1.0727105170135174,
    "guard1_pass": true,
    "guard2_pass": true,
    "guard3_pass": true
  },
  {
    "rank": 9,
    "style_key": "Ladieswear || Sweater || Knitwear || Pink || Solid",
    "predicted_intensity": 13.053416182892029,
    "growth_ratio": 1.0019795543026173,
    "guard1_pass": true,
    "guard2_pass": true,
    "guard3_pass": true
  },
  {
    "rank": 10,
    "style_key": "Divided || Dress || Jersey Basic || Black || Solid",
    "predicted_intensity": 18.931747132235127,
    "growth_ratio": 0.9847878713523339,
    "guard1_pass": true,
    "guard2_pass": true,
    "guard3_pass": true
  }
]
```


**[LOCAL]** orchestrator reads the forecaster's table and selects the three styles named in `final_registry.STYLE_ORDER` (the submitted three; selection rule `nss.models.reselect_final_three`):

| rank | style_key | predicted_intensity |
|---|---|---|
| 2 | Ladieswear || Sweater || Knitwear || Beige || Melange | 17.7831 |
| 3 | Ladieswear || Dress || Dresses Ladies || Red || Solid | 10.9968 |
| 5 | Ladieswear || Top || Jersey Basic || White || Solid | 13.8538 |



## Step 2 -- `data-analyst`: not invoked (optional in `orchestrator.md`, not asked)


## Style `Ladieswear || Sweater || Knitwear || Beige || Melange`

**[LIVE] (0.0 s)** style-profiler calls `get_style_profile` for `Ladieswear || Sweater || Knitwear || Beige || Melange`; response summary: trajectory available = True, SHAP source `reports\tables\final_three_shap_verdict.csv`.

**[INPUT]** design brief (human-authored, `concept_generation.CHANGES`): ['a high funnel neck collar', 'dark brown contrast rib cuffs and hem']

**[LOCAL]** `natural_prompt` builds the prompt: `flat-lay product photo of a beige melange sweater with a high funnel neck collar and dark brown contrast rib cuffs and hem, plain light background`; 8 references.

**[LIVE] (33.7 s)** attempt 1: concept-designer calls `generate_concept` (ip_adapter_scale 0.35, seed 42, n 1) -> `data/generated/local_sdxl/seed42_00.png` (the tool names files `seed<seed>_<i>.png` with no style or scale, so the next call with the same seed overwrites it; the driver archived this one at `data/generated/agent_live/sweater_a1_s0.35_seed42.png`)

**[LIVE] (33.0 s)** attempt 1: critic calls `score_concept` (include_fidelity, brief changes):

- Gate 1: CLIP 0.909 <= 0.966, DINOv2 0.788 <= 0.904 -> pass
- Gate 1b: CLIP 0.935 <= 0.983, DINOv2 0.865 <= 0.947 -> pass (exact-clone control failed as required: True)
- Integrity (GLOBAL floor gates): closest reference 0.865 >= 0.779 -> pass; per-style floor (advisory) 0.835 -> pass
- Gate 2: smolvlm (gating) 0.000 vs threshold 0.384 -> FAIL; florence2 (advisory) 0.567 vs threshold 0.337 -> pass -> **FAIL**
- Gate 3 (briefed changes visible, SmolVLM): answers YN -> **FAIL**
- Human visual check: required = True
- automated_gates_pass = False; verdict: `REJECT: failed gate2, gate3`


**[LOCAL]** critic decision `REJECT` (failing gates: ['gate2', 'gate3']); router: RETRY -> `concept-designer`, adjust `ip_adapter_scale`

**[LIVE] (23.9 s)** attempt 2: concept-designer calls `generate_concept` (ip_adapter_scale 0.25, seed 42, n 1) -> `data/generated/local_sdxl/seed42_00.png` (the tool names files `seed<seed>_<i>.png` with no style or scale, so the next call with the same seed overwrites it; the driver archived this one at `data/generated/agent_live/sweater_a2_s0.25_seed42.png`)

**[LIVE] (23.9 s)** attempt 2: critic calls `score_concept` (include_fidelity, brief changes):

- Gate 1: CLIP 0.885 <= 0.966, DINOv2 0.803 <= 0.904 -> pass
- Gate 1b: CLIP 0.913 <= 0.983, DINOv2 0.876 <= 0.947 -> pass (exact-clone control failed as required: True)
- Integrity (GLOBAL floor gates): closest reference 0.876 >= 0.779 -> pass; per-style floor (advisory) 0.835 -> pass
- Gate 2: smolvlm (gating) 0.000 vs threshold 0.384 -> FAIL; florence2 (advisory) 0.567 vs threshold 0.337 -> pass -> **FAIL**
- Gate 3 (briefed changes visible, SmolVLM): answers YY -> **pass**
- Human visual check: required = True
- automated_gates_pass = False; verdict: `REJECT: failed gate2`


**[LOCAL]** critic decision `REJECT` (failing gates: ['gate2']); router: RETRY -> `concept-designer`, adjust `ip_adapter_scale`

**[LIVE] (23.6 s)** attempt 3: concept-designer calls `generate_concept` (ip_adapter_scale 0.15, seed 42, n 1) -> `data/generated/local_sdxl/seed42_00.png` (the tool names files `seed<seed>_<i>.png` with no style or scale, so the next call with the same seed overwrites it; the driver archived this one at `data/generated/agent_live/sweater_a3_s0.15_seed42.png`)

**[LIVE] (30.8 s)** attempt 3: critic calls `score_concept` (include_fidelity, brief changes):

- Gate 1: CLIP 0.892 <= 0.966, DINOv2 0.819 <= 0.904 -> pass
- Gate 1b: CLIP 0.939 <= 0.983, DINOv2 0.893 <= 0.947 -> pass (exact-clone control failed as required: True)
- Integrity (GLOBAL floor gates): closest reference 0.893 >= 0.779 -> pass; per-style floor (advisory) 0.835 -> pass
- Gate 2: smolvlm (gating) 0.283 vs threshold 0.384 -> FAIL; florence2 (advisory) 0.567 vs threshold 0.337 -> pass -> **FAIL**
- Gate 3 (briefed changes visible, SmolVLM): answers YN -> **FAIL**
- Human visual check: required = True
- automated_gates_pass = False; verdict: `REJECT: failed gate2, gate3`


**[LOCAL]** critic decision `REJECT` (failing gates: ['gate2', 'gate3']); router: FAILED

**[NOT PERFORMED]** human visual check for `Ladieswear || Sweater || Knitwear || Beige || Melange` (outcome so far: FAILED (retry cap exhausted)). A person must open `data/generated/agent_live/sweater_a3_s0.15_seed42.png`; this driver cannot perform it and records no result for it.


## Style `Ladieswear || Dress || Dresses Ladies || Red || Solid`

**[LIVE] (0.0 s)** style-profiler calls `get_style_profile` for `Ladieswear || Dress || Dresses Ladies || Red || Solid`; response summary: trajectory available = True, SHAP source `reports\tables\final_three_shap_verdict.csv`.

**[INPUT]** design brief (human-authored, `concept_generation.CHANGES`): ['a square neckline with puff sleeves', 'a wide self belt tied at the waist']

**[LOCAL]** `natural_prompt` builds the prompt: `flat-lay product photo of a red dress with a square neckline with puff sleeves and a wide self belt tied at the waist, plain light background`; 8 references.

**[LIVE] (23.2 s)** attempt 1: concept-designer calls `generate_concept` (ip_adapter_scale 0.35, seed 42, n 1) -> `data/generated/local_sdxl/seed42_00.png` (the tool names files `seed<seed>_<i>.png` with no style or scale, so the next call with the same seed overwrites it; the driver archived this one at `data/generated/agent_live/dress_a1_s0.35_seed42.png`)

**[LIVE] (38.9 s)** attempt 1: critic calls `score_concept` (include_fidelity, brief changes):

- Gate 1: CLIP 0.857 <= 0.951, DINOv2 0.749 <= 0.840 -> pass
- Gate 1b: CLIP 0.916 <= 0.974, DINOv2 0.876 <= 0.901 -> pass (exact-clone control failed as required: True)
- Integrity (GLOBAL floor gates): closest reference 0.876 >= 0.779 -> pass; per-style floor (advisory) 0.754 -> pass
- Gate 2: smolvlm (gating) 0.567 vs threshold 0.384 -> pass; florence2 (advisory) 0.567 vs threshold 0.337 -> pass -> **pass**
- Gate 3 (briefed changes visible, SmolVLM): answers YY -> **pass**
- Human visual check: required = True
- automated_gates_pass = True; verdict: `PASS on all automatic gates; the human visual check is still REQUIRED`


**[LOCAL]** critic decision `PASS_PENDING_HUMAN` (failing gates: none); router: FORWARD -> `forecaster`, adjust `None`

**[LIVE] (16.8 s)** **[LIVE] forecaster** calls MCP tool `forecast_concept` (real call over the MCP stdio protocol -- `ClientSession.call_tool` against a live `nss.mcp_server` subprocess):

Request:
```json
{
  "concept_path": "data/generated/local_sdxl/seed42_00.png"
}
```

Response (top-5 as one line each; `>>` marks the intended style):
```json
{
  "sentence": "maps to Ladieswear || Dress || Blouses || Orange || Solid; forecast 3.0 units/product/week; rank 1443 of 1,980; confidence high",
  "style_key": "Ladieswear || Dress || Blouses || Orange || Solid",
  "forecast_units_per_product_per_week": 3.0354507422717028,
  "rank": 1443,
  "n_styles": 1980,
  "match_level": "retrieval",
  "confidence": "high",
  "similarity": 0.91301429271698,
  "margin": 0.040419697761535645,
  "n_indexed_styles": 1980,
  "top5 (>> = the style this concept was designed from)": [
    "   1. Ladieswear || Dress || Blouses || Orange || Solid  (similarity 0.913, forecast 3.0, rank 1443)",
    ">> 2. Ladieswear || Dress || Dresses Ladies || Red || Solid  (similarity 0.892, forecast 11.0, rank 282)",
    "   3. Baby/Children || Dress || Dresses/Skirts girls || Pink || Solid  (similarity 0.881, forecast 1.9, rank 1752)",
    "   4. Ladieswear || Dress || Blouses || Pink || Solid  (similarity 0.880, forecast 2.9, rank 1473)",
    "   5. Divided || Dress || Dresses Ladies || Yellow || Solid  (similarity 0.877, forecast 4.6, rank 1012)"
  ]
}
```


**[NOT PERFORMED]** human visual check for `Ladieswear || Dress || Dresses Ladies || Red || Solid` (outcome so far: PASS_PENDING_HUMAN). A person must open `data/generated/agent_live/dress_a1_s0.35_seed42.png`; this driver cannot perform it and records no result for it.


## Style `Ladieswear || Top || Jersey Basic || White || Solid`

**[LIVE] (0.0 s)** style-profiler calls `get_style_profile` for `Ladieswear || Top || Jersey Basic || White || Solid`; response summary: trajectory available = True, SHAP source `reports\tables\final_three_shap_verdict.csv`.

**[INPUT]** design brief (human-authored, `concept_generation.CHANGES`): ['a square neckline', 'long balloon sleeves with wide ribbed cuffs']

**[LOCAL]** `natural_prompt` builds the prompt: `flat-lay product photo of a white top with a square neckline and long balloon sleeves with wide ribbed cuffs, plain light background`; 8 references.

**[LIVE] (25.2 s)** attempt 1: concept-designer calls `generate_concept` (ip_adapter_scale 0.35, seed 42, n 1) -> `data/generated/local_sdxl/seed42_00.png` (the tool names files `seed<seed>_<i>.png` with no style or scale, so the next call with the same seed overwrites it; the driver archived this one at `data/generated/agent_live/top_a1_s0.35_seed42.png`)

**[LIVE] (40.1 s)** attempt 1: critic calls `score_concept` (include_fidelity, brief changes):

- Gate 1: CLIP 0.871 <= 0.977, DINOv2 0.726 <= 0.928 -> pass
- Gate 1b: CLIP 0.900 <= 0.986, DINOv2 0.799 <= 0.956 -> pass (exact-clone control failed as required: True)
- Integrity (GLOBAL floor gates): closest reference 0.799 >= 0.779 -> pass; per-style floor (advisory) 0.922 -> fail
- Gate 2: smolvlm (gating) 0.850 vs threshold 0.384 -> pass; florence2 (advisory) 0.567 vs threshold 0.337 -> pass -> **pass**
- Gate 3 (briefed changes visible, SmolVLM): answers YY -> **pass**
- Human visual check: required = True
- automated_gates_pass = True; verdict: `PASS on all automatic gates; the human visual check is still REQUIRED`


**[LOCAL]** critic decision `PASS_PENDING_HUMAN` (failing gates: none); router: FORWARD -> `forecaster`, adjust `None`

**[LIVE] (4.2 s)** **[LIVE] forecaster** calls MCP tool `forecast_concept` (real call over the MCP stdio protocol -- `ClientSession.call_tool` against a live `nss.mcp_server` subprocess):

Request:
```json
{
  "concept_path": "data/generated/local_sdxl/seed42_00.png"
}
```

Response (top-5 as one line each; `>>` marks the intended style):
```json
{
  "sentence": "maps to Divided || Top || Jersey Fancy || White || Solid; forecast 6.4 units/product/week; rank 651 of 1,980; confidence high",
  "style_key": "Divided || Top || Jersey Fancy || White || Solid",
  "forecast_units_per_product_per_week": 6.433257938528905,
  "rank": 651,
  "n_styles": 1980,
  "match_level": "retrieval",
  "confidence": "high",
  "similarity": 0.8626073002815247,
  "margin": 0.041119158267974854,
  "n_indexed_styles": 1980,
  "top5 (>> = the style this concept was designed from)": [
    "   1. Divided || Top || Jersey Fancy || White || Solid  (similarity 0.863, forecast 6.4, rank 651)",
    "   2. Divided || Top || Knitwear || White || Solid  (similarity 0.839, forecast 12.6, rank 215)",
    "   3. Divided || T-shirt || Jersey Fancy || White || Solid  (similarity 0.839, forecast 4.8, rank 955)",
    ">> 4. Ladieswear || Top || Jersey Basic || White || Solid  (similarity 0.834, forecast 13.9, rank 174)",
    "   5. Ladieswear || Top || Knitwear || White || Solid  (similarity 0.826, forecast 7.0, rank 561)"
  ]
}
```


**[NOT PERFORMED]** human visual check for `Ladieswear || Top || Jersey Basic || White || Solid` (outcome so far: PASS_PENDING_HUMAN). A person must open `data/generated/agent_live/top_a1_s0.35_seed42.png`; this driver cannot perform it and records no result for it.


## Steps by label

| Label | Steps |
|---|---|
| LIVE | 16 |
| INPUT | 3 |
| LOCAL | 9 |
| NOT PERFORMED | 3 |

## Outcomes

- `Ladieswear || Sweater || Knitwear || Beige || Melange`: FAILED (retry cap exhausted)
- `Ladieswear || Dress || Dresses Ladies || Red || Solid`: PASS_PENDING_HUMAN
- `Ladieswear || Top || Jersey Basic || White || Solid`: PASS_PENDING_HUMAN
