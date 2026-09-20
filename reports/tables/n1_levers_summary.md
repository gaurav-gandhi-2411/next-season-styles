# N1 lever experiments (beige melange sweater, seeds 42 and 43)

Images: `data/generated/n1_levers/ladieswear_sweater_knitwear_beige_melange/` (not committed; the
contact sheets behind each row were inspected by eye, and 23 of them are labelled in
`evals/fixtures/gate3_sweater_labels.json`). Briefed changes: "a high funnel neck collar" and
"dark brown contrast rib cuffs and hem". Installed `diffusers==0.40.0`, IP-Adapter Plus (ViT-H).

| Lever | Configuration | Briefed changes visible? |
|---|---|---|
| Baseline (M3) | old attribute-first prompt, 1 reference, scale 0.45 | no (0 of 12 images, three styles) |
| (a) multi-reference | 6 refs concatenated, old prompt, 0.35 / 0.45 | no: crew neck instead of V neck, no funnel, no brown |
| (a') multi-reference, mean embedding | mean of 6 image-encoder states, 0.45 | no (and 17 s/step on 8 GB: encoder left resident) |
| (b) scale sweep | concat, old prompt, 0.15 / 0.25 / 0.35 | no; 0.15 and 0.25 collapse into fabric swatches |
| (c) per-block scale | style block only (1.0, 0.6) | no: fabric swatch |
| (c) per-block scale | style + layout blocks (0.6) | no: crew neck |
| text only | scale 0.0, old prompt / natural sentence with attribute-only second prompt | swatches |
| (d) natural sentence, both encoders | concat, 0.25 / 0.35 / 0.45 | yes at 0.25 and 0.35 (funnel neck, brown cuffs and hem); funnel only at 0.45 |
| (d) natural sentence, 1 reference | single, 0.25 / 0.35 / 0.45 | yes at 0.25 and 0.35; lost at 0.45 |
| (d) + compel weight 1.5 | concat, natural sentence, 0.45 | yes (brown accents in 2 of 2, against 0 of 2 unweighted) |

Winning configuration (used for N9): natural sentence on both text encoders + 8-reference concat
+ compel weight 1.5 on the change clauses + a per-style scale (0.35, from a 0.15-0.45 sweep).
Attribution: prompt structure is the dominant lever; multi-reference and weighting are secondary.
The per-block scale API is supported by the installed version (`set_ip_adapter_scale` with a dict).
