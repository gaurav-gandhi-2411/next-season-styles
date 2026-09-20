# next-season-styles: submission bundle

Start with `DEMO.html` (open it in any browser; it is self-contained, needs no server or internet).

| File | What it is |
|---|---|
| `DEMO.html` | The concepts, how each traces to its forecast, the model evidence, the seasonal view, and what could not be verified, for a non-specialist |
| `FINAL_concepts.png` | The three generated concepts, one per forecast style |
| `evidence_chain.png` | Per concept: references, brief, checks, verdict |
| `seasonal_comparison.png` | The same pipeline for autumn/winter and for summer |
| `WRITEUP.md` | The full argument and its limits (about 2,000 words) |
| `SUBMISSION_CHECKLIST.md` | Each required item mapped to the file that satisfies it |

**Repository:** https://github.com/gaurav-gandhi-2411/next-season-styles (private; access can be
granted on request). This bundle is cut from its `main` branch; the code, tests, agents, skills, MCP
server and every table cited here live there.

- Reviewers without a GPU: `uv run --no-sync python scripts/run_pipeline.py --dry-run` (about 1 to
  2.5 minutes, writes only to a scratch directory; needs the H&M data in `data/`).
- Full reproduction (CUDA GPU): `uv run --no-sync python scripts/run_pipeline.py`.
- MCP server check: `uv run --no-sync python scripts/mcp_smoke_test.py` (expected output is in the
  repository README).
