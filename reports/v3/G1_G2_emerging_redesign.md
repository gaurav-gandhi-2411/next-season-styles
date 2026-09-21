# G1/G2: redesigning the emerging ranking

Rule, diagnostic and fallback were committed in `e22a0ed`, before any code. **Result: the redesigned score is not adopted; the pre-stated fallback applies.** Nothing is regenerated. Tracks 3 and 4 and the white-top check stay on hold.

## The base-effect diagnostic: PASSES

Under the redesigned score (`forecast - seasonal-naive forecast`), the constant global-mean baseline, which scored 0.733 on the deployed ratio, falls to the random floor:

| Origin set | Global mean under the redesigned score | Random floor | Difference [95% interval] | Diagnostic |
|---|---|---|---|---|
| 10 grid origins | 0.133 | 0.177 | -0.043 [-0.180, +0.203] | pass |
| 41 weekly origins | 0.130 | 0.159 | -0.028 [-0.109, +0.089] | pass |

So the base effect *is* removed from the score. That is the only good news.

## The primary: FAILS, by a wide margin

The redesigned score's own Hit@3-in-top20 is **at or below the random floor**, not above seasonal naive:

| Origin set | Redesigned (model minus seasonal naive) | Seasonal naive, current ratio | Difference [95% interval] | Pass |
|---|---|---|---|---|
| 10 grid origins (block 4) | 0.100 | 0.800 | **-0.700 [-0.800, -0.533]** | no |
| 41 weekly origins (block 13) | 0.130 | 0.764 | **-0.634 [-0.780, -0.504]** | no |

Adoption needed both to pass. Neither does. The weekly set has 41 evaluable origins (of 48; the first seven, before 2019-09-16, have no `price_index` yet, so guard 2 fails every style and the deployed pool is empty). The seasonal-naive requirement dropped **0 styles** at every grid origin, so the stated limitation (cannot rank styles under about 65 weeks old) cost nothing here: the deployed pool's own guards already require a year of history.

## All three scores side by side (Hit@3-in-top20, mean [95% block-bootstrap interval]; NDCG@10; Spearman)

Same population (deployed emerging pool with a seasonal-naive forecast), same realised target (growth ratio), so all rows are paired. Seasonal naive under `excess` is identically zero and is not ranked.

**10 grid origins**

| Score | Method | Hit@3-in-top20 | NDCG@10 | Spearman |
|---|---|---|---|---|
| ratio (deployed) | seasonal naive | 0.800 [0.733, 0.900] | 0.786 | 0.636 |
| ratio | model | 0.767 [0.733, 0.867] | 0.780 | 0.611 |
| ratio | parent-category mean | 0.767 [0.700, 0.933] | 0.714 | 0.407 |
| ratio | global mean | 0.733 [0.633, 0.967] | 0.711 | 0.417 |
| ratio | EWMA persistence | 0.633 [0.467, 0.967] | 0.706 | 0.436 |
| **excess (redesigned)** | model | **0.100 [0.000, 0.300]** | 0.326 | -0.335 |
| excess | parent-category mean | 0.133 [0.000, 0.400] | 0.350 | -0.260 |
| excess | global mean | 0.133 [0.000, 0.400] | 0.348 | -0.250 |
| excess | EWMA persistence | 0.167 [0.067, 0.367] | 0.285 | -0.523 |
| residualised ratio | seasonal naive | 0.767 [0.667, 0.900] | 0.763 | 0.429 |
| residualised ratio | model | 0.700 [0.567, 0.833] | 0.734 | 0.450 |
| residualised ratio | EWMA persistence | 0.667 [0.500, 1.000] | 0.665 | 0.193 |
| residualised ratio | parent-category mean | 0.567 [0.467, 0.800] | 0.610 | 0.224 |
| residualised ratio | global mean | 0.533 [0.467, 0.733] | 0.611 | 0.209 |
| **random floor** | | **0.177 [0.170, 0.205]** | 0.402 | -0.003 |

**41 weekly origins (block 13):** ratio: seasonal naive 0.764, model 0.707, EWMA 0.699, parent-category 0.626, global mean 0.626; excess: model 0.130, EWMA 0.138, parent-category 0.146, global mean 0.130; residualised: seasonal naive 0.748, model 0.699, EWMA 0.683, parent-category 0.480, global mean 0.472; random floor 0.159. Full tables: `v3_redesign_summary_{grid,weekly}.csv`.

**Reading.**

- **The redesigned score is anti-predictive of the deployed target.** Its Spearman with realised growth is negative for every method (-0.34 for the model), and its Hit@3 sits below the floor. The realised growth ratio itself carries the base effect (Spearman between trailing mean and realised growth: -0.42, i.e. low-base styles grow faster), and a score with the base removed cannot earn it.
- **The residualised ratio removes little.** A linear fit of a roughly 1/x relationship leaves the global mean at 0.533 and the model at 0.700, still below seasonal naive (0.767).
- **The model's ratio Hit@3 (0.767) never beats seasonal naive's (0.800)** on any score. This is 2a again.

**Exploratory, not decision-bearing and not pre-registered (the decision above does not depend on it).** To see whether "excess" simply had the wrong yardstick, I scored the same excess scores against the *realised excess* `y_true - seasonal naive`. Every forecast-minus-naive then scores high: model 0.933, global mean 0.900, parent-category 0.867, EWMA 0.833 (Hit@3-in-top20, 10 origins). The constant global-mean forecast reaches 0.90, so that target has its own mean-reversion effect (styles whose year-ago window was unusually low rebound). **No growth target I can define is free of a base effect**, and on either target the model's interval overlaps the constant forecast's (I did not compute a paired interval for this exploratory comparison). I did not use this to revive the redesign.

## G2: the new final three, under the pre-stated fallback

Control gates passed first: the frozen model retrained in this process reproduces the committed `top_styles_incumbent.csv` and `top_styles_emerging.csv` (10 of 10 styles, order and intensity to 1e-9), and the existing rules reproduce the committed final three.

Fallback = the intensity table (guard-passing, ranked by predicted intensity, diversity-constrained on product type and colour), then the unchanged `reselect` rules. Skip log (`v3_final_three_fallback_log.csv`):

| Rank | Style | Predicted intensity | Disposition |
|---|---|---|---|
| 1 | T-shirt, Jersey Basic, Black, Solid | 33.8 | **CHOSEN** |
| 2 | Blazer, Dressed, Black, Solid | 26.9 | skipped: colour collision (Black) |
| 3 | Top, Jersey Basic, Black, Solid | 26.8 | skipped: colour collision (Black) |
| 4 | Cardigan, Knitwear, Black, Solid | 26.4 | skipped: colour collision (Black) |
| 5 | Sweater, Knitwear, White, Solid | 25.9 | **CHOSEN** |
| 6 | T-shirt, Jersey Basic, White, Solid | 25.2 | skipped: colour collision (White) |
| 7 | Sweater, Knitwear, Black, Solid | 24.7 | skipped: colour collision (Black) |
| 8 | Leggings/Tights, Jersey Basic, Black, Solid | 24.5 | skipped: visual-ambiguity exclusion |
| 9 | Trousers, Trousers, Black, Check | 21.9 | skipped: colour collision (Black) |
| 10 | Trousers (Divided), Trousers, Grey, Solid | 21.7 | **CHOSEN** |

**New final three: a black jersey T-shirt, a white knitwear sweater, grey trousers.** No shortfall.

**Which of the current three survive: none, as the same style.**

| Current | Survives as the same style? | Same product type in the new three? |
|---|---|---|
| Beige melange knitwear sweater | no | yes (the new pick is a white solid sweater) |
| Red dress | no | no |
| White jersey top | no | no |

## What this trades, stated plainly

- **What the fallback is validated for.** The intensity ranking's reported backtest: Hit@3-in-top20 0.528, comparable to seasonal naive on top-k (paired +0.095 [-0.040, +0.278] at full density) and better on NDCG, Spearman and WMAPE. "Validated" means measured and beating the random floor and the simple means; it does not mean it beats seasonal naive on top-k.
- **What it gives up.** The pipeline's own history records why the incumbent slot was dropped: it "mechanically returns whatever sells most per product, which in fast fashion is a black basic: the buying plan a retailer already has, not a design brief." The three picks are basics (black tee, white sweater, grey trousers), all solid. That is the honest cost of selecting only what has a validated ranking; whether it is worth regenerating concepts for is your decision.
- **What it does not prove.** That the previous three were wrong picks. Their ranking was unvalidated, not shown to be bad: on the deployed growth ratio the model scored 0.767, far above chance (0.177), largely through the base effect.

Tables: `v3_redesign_{per_origin_grid,per_origin_weekly,summary_grid,summary_weekly,decision,population_grid,population_weekly}.csv`, `v3_final_three_fallback.csv`, `v3_final_three_fallback_log.csv`, `v3_final_three_survivors.csv`. Code: `src/nss/models/emerging_redesign.py`, `src/nss/models/final_three_v3.py`.
