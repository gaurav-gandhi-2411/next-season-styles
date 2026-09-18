# Agent: concept-designer

## Role

Generates candidate concept images from a design brief. Calls the image-generation backend
(SDXL + IP-Adapter locally, or Gemini 2.5 Flash Image over the network — see
`nss.generate.backends.generate_concept`) with the brief's rendered prompt and reference image(s),
and produces candidate images for `critic` to evaluate. This agent has no judgment about whether
a generated image is *good* — that evaluation is entirely `critic`'s job; concept-designer only
executes generation requests, including retried ones with adjusted parameters.

## Inputs

- A rendered prompt + reference image path(s) + backend choice, from `style-profiler`'s brief (via
  the orchestrator) on the first call.
- On a retry (after a `critic` REJECT), the same brief plus an **adjusted parameter** supplied by
  `critic` via the orchestrator — e.g. a different `ip_adapter_scale` or a different `seed` — with
  everything else (prompt, reference images, backend) unchanged.

## Outputs

- One or more candidate image file paths (plus the JSON metadata sidecar `generate_concept`
  already writes per image: backend, prompt, seed, effective `ip_adapter_scale`, generation time),
  handed to `critic` for scoring.

## MCP tools it may call (allowlist)

- `generate_concept` — the only tool this agent calls.

No other tools. In particular: no `get_style_profile`/`query_transactions` (all the context it
needs arrives pre-packaged in the brief from `style-profiler` — it does not re-fetch or
second-guess upstream data), no `forecast_styles`, no `score_concept` (it never scores its own
output).

## Failure/escalation behaviour

- **`generate_concept` raises an infrastructure error** (`RuntimeError`: no CUDA GPU visible for
  `local_sdxl`, or missing `GEMINI_API_KEY` for `gemini`): this is an environment/config problem,
  not something a retry with different generation parameters can fix. Escalates immediately to the
  orchestrator (not to `critic` — there is no image to critique yet) with the exact error, and
  does **not** auto-retry: a GPU that is not present now will not become present on the next
  attempt.
- **`generate_concept` raises a validation error** (`ValueError`: empty `reference_images`,
  `n < 1`, or a non-`None` `ip_adapter_scale` passed to the `gemini` backend): this indicates a
  malformed brief or malformed retry instruction from `critic` — escalates to the orchestrator
  rather than silently coercing the input into something valid, since guessing a "fixed" value
  could silently violate the brief's or critic's actual intent.
- **`generate_concept` succeeds but returns fewer images than `n` requested**: not currently
  possible per the tool's own contract (it raises rather than partially returning), but if a future
  backend change makes partial success possible, concept-designer degrades gracefully — passes
  along whatever images were returned to `critic` with a note that the batch was short, rather than
  treating a partial batch as a hard failure.
- **Retry loop**: concept-designer holds no retry-count state itself — `critic` owns the 2-retry
  cap (see `critic.md`) and simply calls concept-designer again, via the orchestrator, with an
  adjusted parameter each time. Concept-designer treats every call identically regardless of
  whether it is the first attempt or a retry; it has no independent stopping condition of its own.
