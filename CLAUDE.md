# next-season-styles

Read SPEC.md before starting any work. Update its Section 9 status table at the end of every session.

## DATA SAFETY

This project's `data/` directory was destroyed on 2026-09-23 when `git worktree remove --force`
was run on a worktree nested inside this repo (`.claude/worktrees/agent-a739ec72e4d5c25b3`). The
plain `git worktree remove` refused first ("contains modified or untracked files"); `--force` was
then used to override that refusal, and the recursive delete behind it destroyed this repo's own
`data/raw/`, `data/interim/`, `data/processed/`, `data/retrieval_cache/`, `data/rembg_models/`,
`data/external/` and `data/images/` — not just the worktree's copy. This was the second time this
project lost its `data/` directory. Recovery required re-sourcing raw CSVs from a sibling project's
Kaggle copy, rebuilding derived data from a sibling worktree's independent copy, and regenerating
the retrieval embedding cache.

Rules, binding on every session working in this repo:

- **Never use `git worktree remove --force`, anywhere, for any reason.** If the plain
  `git worktree remove` refuses (uncommitted changes, untracked files, or any other reason), stop
  and report the refusal to the user. Do not override it with `--force`, and do not work around it
  by deleting the worktree directory manually instead.
- **Before removing any worktree, scan it for reparse points** (junctions and symlinks) with
  `Get-ChildItem -Recurse -Force -Directory | Where-Object { $_.Attributes -band
  [System.IO.FileAttributes]::ReparsePoint }` or `dir /AL /S` on the worktree's path specifically.
  If any reparse point is found anywhere under the worktree, do not remove it without explicit
  confirmation from the user, even if the reparse point's target looks unrelated to `data/`.
- **Never create a junction or symlink from inside a worktree (or anywhere in this repo's working
  tree) into this repo's own `data/` directory**, or into any other repo's data. If a worktree
  needs access to the catalogue data, use `NSS_HM_IMAGE_TREE` (or add a new env var / config entry)
  to point at the read-only source directly — never a filesystem-level alias that a generic
  recursive-delete tool could follow.
- **A destructive command's refusal is a stop signal, never an obstacle to route around.** This
  applies to every destructive git operation in this repo (`git worktree remove`, `git clean`,
  `git reset --hard`, `git checkout --`/`restore` over modified files), not just worktree removal.
  If a command refuses or warns, stop and report to the user; do not retry with a more forceful
  flag, and do not find an alternate command that achieves the same result without the safety
  check.

See the recovery session's report for the full incident writeup, the reparse-point scan across
`ml-projects`, and the proposal for a single canonical read-only H&M data location.

### Canonical H&M data: `C:\Users\gaura\hm-data\`

The master dataset (4 CSVs + `images/`, 105,104 files, 34,558,584,597 bytes as copied on
2026-09-23, plus one `ACL_TEST_SENTINEL.txt`) lives outside every git repository, under an NTFS
ACL that denies delete to the current user. This project reads images from it via
`NSS_HM_IMAGE_TREE=C:\Users\gaura\hm-data\images` (set as a User environment variable; also the
first entry of `TREE_CANDIDATES` in `nss.generate.concept_forecast_index`). The copy in
`multimodal-fashion-recommender` stays untouched as a redundant backup.

- **Apply** (inherits to every file and folder beneath):
  `icacls C:\Users\gaura\hm-data /deny "LEGION\gaura:(OI)(CI)(DE,DC)"`
- **Undo** (removes every deny ACE for that user, including the inherited ones):
  `icacls C:\Users\gaura\hm-data /remove:d "LEGION\gaura"`
- **Check:** `icacls C:\Users\gaura\hm-data` must show `LEGION\gaura:(OI)(CI)(DENY)(DE,DC)`.

Tested on 2026-09-23 on a scratch folder first: under the deny, reads succeeded, and file delete,
recursive folder delete and rename all failed with "Access ... is denied"; after the undo, the same
deletes succeeded. On `hm-data`: reads succeed, and deleting `ACL_TEST_SENTINEL.txt` fails. The ACL
does **not** block creating or overwriting files — it protects against deletion only. Only remove it
with the user's explicit instruction, and re-apply it immediately after.
