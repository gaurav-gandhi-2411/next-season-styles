"""N3: catalogue-wide colour priors by pattern class (R1-R4: a resource-safe, resumable redesign
after the first attempt was killed by the harness for pushing the machine into low memory).

Method and class definition pre-registered in `reports/v3/PREREGISTRATION.md` (section N3,
commit 635ac6d), before this scored anything. Universe: the 1,980 autumn forecast-eligible styles
(`reports/tables/forecast_all_styles.csv`), not the five calibrated ones.

**R1 (diagnosis).** rembg's onnxruntime session already runs on `CPUExecutionProvider` only: the
project venv has plain `onnxruntime`, not `onnxruntime-gpu`, so `ort.get_available_providers()`
never contains a CUDA provider -- there is nothing to force. The GPU (and 5+ GB of system RAM) is
held by another local process (Ollama's `llama-server.exe`) and this script never touches it.

**R2 (memory design).** The first attempt used 12 single-threaded worker processes, each loading
its own ~1.1 GB rembg model copy -- memory scaled with worker count. This version uses at most 2
worker processes (`--workers`), each with multiple onnxruntime intra-op threads instead of many
processes (rembg's `new_session` reads `OMP_NUM_THREADS` and sets both `inter_op_num_threads` and
`intra_op_num_threads` from it -- see `rembg.session_factory.new_session`), so `workers *
threads_per_worker` never exceeds the R3 core budget.

**R3 (resource limits).** Each worker: `psutil` `BELOW_NORMAL_PRIORITY_CLASS` (so the OS prefers
any other process automatically) and CPU affinity restricted to half the physical cores' logical
siblings (assumes consecutive HT pairing, the common case on Windows/Intel; reported, not assumed
silently). Before writing each checkpoint batch, the worker checks system-available RAM; below
`--mem-guard-gb` (default 4) it sleeps 30 s and re-checks, logging every pause, instead of
proceeding. This is a system-wide guard (catches contention from ANY process, not just our own),
which is the actual OOM-prevention mechanism; the main process also samples and reports this run's
own peak RSS against the 6 GB target, but that number is informational, not the safety net.

**R4 (checkpoint/resume).** Each worker appends its per-photo results to its own JSONL checkpoint
file under `data/retrieval_cache/` (gitignored -- a resumable cache, not a report artifact),
flushed and fsynced every `--checkpoint-every` photos (default 500). On start, a worker reads its
checkpoint file, builds the set of paths already recorded (a truncated last line from a kill is
dropped and redone) and skips them. Rerunning this script after any kill resumes from the last
flushed checkpoint.

    uv run --no-sync python scripts/colour_catalogue_priors.py \
        [--limit N] [--workers 2] [--threads-per-worker 2] [--checkpoint-every 500] \
        [--mem-guard-gb 4] [--fresh]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

import polars as pl
import psutil

TABLES = "reports/tables"
FORECAST = f"{TABLES}/forecast_all_styles.csv"
MIN_STYLES_FOR_PRIOR = 20
CHECKPOINT_DIR = Path("data/retrieval_cache")
MEM_CEILING_GB = 6.0  # R3: informational target for this run's own peak RSS


def _checkpoint_path(shard: int, workers: int) -> Path:
    return CHECKPOINT_DIR / f"colour_catalogue_checkpoint_shard{shard}_of_{workers}.jsonl"


def _read_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    """`path -> record` for every intact line already written; a truncated last line (a kill
    mid-write) is silently dropped so that photo is redone rather than counted as done."""
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[rec["path"]] = rec
    return done


def _worker(
    shard_id: int,
    workers: int,
    paths: list[str],
    threads_per_worker: int,
    affinity_cpus: list[int] | None,
    checkpoint_every: int,
    mem_guard_gb: float,
) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)  # R2: rembg's new_session reads this

    proc = psutil.Process()
    try:
        proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception as exc:  # not fatal: still runs, just without the priority hint
        print(f"[shard {shard_id}] could not set BELOW_NORMAL priority: {exc}", flush=True)
    if affinity_cpus:
        try:
            proc.cpu_affinity(affinity_cpus)
        except Exception as exc:
            print(f"[shard {shard_id}] could not set CPU affinity {affinity_cpus}: {exc}", flush=True)

    from nss.generate import colour_check as cc

    ckpt_path = _checkpoint_path(shard_id, workers)
    done = _read_checkpoint(ckpt_path)
    todo = [p for p in paths if p not in done]
    print(f"[shard {shard_id}] {len(done)} already done, {len(todo)} remaining", flush=True)
    if not todo:
        print(f"[shard {shard_id}] nothing to do, complete", flush=True)
        return

    buf: list[dict[str, Any]] = []
    n_paused = 0
    n_total_done_before = len(done)
    with ckpt_path.open("a", encoding="utf-8") as f:
        for i, path_str in enumerate(todo):
            while True:
                avail_gb = psutil.virtual_memory().available / (1024**3)
                if avail_gb >= mem_guard_gb:
                    break
                n_paused += 1
                print(
                    f"[shard {shard_id}] low memory ({avail_gb:.2f} GB available < "
                    f"{mem_guard_gb} GB guard) -- pausing 30s (pause #{n_paused})",
                    flush=True,
                )
                time.sleep(30)
            try:
                dom = cc.dominant_colour(Path(path_str))
                rec = {
                    "path": path_str,
                    "L": dom.lab[0],
                    "a": dom.lab[1],
                    "b": dom.lab[2],
                    "fallback": dom.used_fallback,
                    "error": None,
                }
            except Exception as exc:  # a handful of catalogue photos are corrupt/unreadable
                rec = {
                    "path": path_str,
                    "L": None,
                    "a": None,
                    "b": None,
                    "fallback": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            buf.append(rec)
            done_so_far = n_total_done_before + i + 1
            if len(buf) >= checkpoint_every or i == len(todo) - 1:
                for r in buf:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                os.fsync(f.fileno())
                print(
                    f"[shard {shard_id}] checkpoint: {done_so_far}/{len(paths)} done "
                    f"({n_paused} memory pauses so far)",
                    flush=True,
                )
                buf = []
    print(f"[shard {shard_id}] complete", flush=True)


def _resource_plan(args: argparse.Namespace) -> tuple[int, int, list[list[int]] | None]:
    """R3: workers, threads/worker and each worker's CPU affinity, from half the physical cores."""
    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count(logical=True) or physical
    budget_physical = max(1, physical // 2)
    workers = min(args.workers, budget_physical)
    threads_per_worker = max(1, budget_physical // workers)
    ht_ratio = max(1, logical // physical)
    budget_logical = budget_physical * ht_ratio
    if logical >= budget_logical:
        # assumes consecutive HT pairing (logical cpu 2i, 2i+1 share physical core i on Windows) --
        # the common case; not verified per-machine, so this is reported, not assumed silently.
        affinity_pool = list(range(budget_logical))
        chunk = max(1, len(affinity_pool) // workers)
        affinities = [affinity_pool[i * chunk : (i + 1) * chunk] for i in range(workers)]
    else:
        affinities = None
    return workers, threads_per_worker, affinities


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap total photos processed (pilot/test runs)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--mem-guard-gb", type=float, default=4.0)
    ap.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint files")
    args = ap.parse_args()

    from nss.generate import colour_check as cc
    from nss.generate import concept_forecast_index as cfi

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    workers, threads_per_worker, affinities = _resource_plan(args)
    physical, logical = psutil.cpu_count(logical=False), psutil.cpu_count(logical=True)
    mem = psutil.virtual_memory()
    print(
        f"R1/R3 snapshot: {physical} physical / {logical} logical cores, "
        f"{mem.available / 1024**3:.2f} GB available / {mem.total / 1024**3:.2f} GB total RAM"
    )
    print(
        f"R2/R3 plan: {workers} worker(s) x {threads_per_worker} intra-op thread(s) "
        f"= {workers * threads_per_worker} of {physical} physical cores; affinity={affinities}"
    )
    print(f"R3: expected peak RSS well under {MEM_CEILING_GB:.0f} GB (2 model copies x ~1.1 GB + buffers)")

    if args.fresh:
        for w in range(workers):
            p = _checkpoint_path(w, workers)
            if p.exists():
                p.unlink()

    styles_df = pl.read_csv(FORECAST).select("style_key", "graphical_appearance_name")
    style_class = {
        r["style_key"]: ("solid" if r["graphical_appearance_name"] in ("Solid", "Melange") else "patterned")
        for r in styles_df.to_dicts()
    }
    print(f"universe: {len(style_class)} styles")

    t0 = time.time()
    by_style = cfi.collect_index_images(style_class.keys(), max_per_style=8)
    print(f"collected images for {len(by_style)} styles in {time.time() - t0:.1f}s")

    all_paths = sorted({str(p) for paths in by_style.values() for p in paths})
    if args.limit is not None:
        all_paths = all_paths[: args.limit]
        wanted = set(all_paths)
        by_style = {s: [p for p in ps if str(p) in wanted] for s, ps in by_style.items()}
    print(f"{len(all_paths)} distinct photos to process across {workers} shard(s)")

    shards = [all_paths[i::workers] for i in range(workers)]
    procs = []
    t0 = time.time()
    for i, shard_paths in enumerate(shards):
        aff = affinities[i] if affinities else None
        proc = mp.Process(
            target=_worker,
            args=(i, workers, shard_paths, threads_per_worker, aff, args.checkpoint_every, args.mem_guard_gb),
        )
        proc.start()
        procs.append(proc)

    peak_rss_gb = 0.0
    while any(p.is_alive() for p in procs):
        rss = 0
        for p in procs:
            if p.pid is None:
                continue
            try:
                rss += psutil.Process(p.pid).memory_info().rss
            except psutil.NoSuchProcess:
                pass
        peak_rss_gb = max(peak_rss_gb, rss / 1024**3)
        time.sleep(15)
    for proc in procs:
        proc.join()
    dt = time.time() - t0
    print(f"all shards complete in {dt:.1f}s; this run's peak worker RSS ~{peak_rss_gb:.2f} GB")

    lab_by_path: dict[str, tuple[float, float, float]] = {}
    fallback_by_path: dict[str, bool] = {}
    errors: list[dict[str, Any]] = []
    for w in range(workers):
        for rec in _read_checkpoint(_checkpoint_path(w, workers)).values():
            if rec["error"] is not None:
                errors.append(rec)
                continue
            lab_by_path[rec["path"]] = (rec["L"], rec["a"], rec["b"])
            fallback_by_path[rec["path"]] = rec["fallback"]
    if errors:
        print(f"{len(errors)} photos failed and were skipped, e.g. {errors[0]}")

    rows: list[dict[str, Any]] = []
    for style, paths in by_style.items():
        labs = [lab_by_path[str(p)] for p in paths if str(p) in lab_by_path]
        n = len(labs)
        row: dict[str, Any] = {
            "style_key": style,
            "class": style_class[style],
            "n_photos_requested": len(paths),
            "n_photos_ok": n,
            "n_fallback": sum(fallback_by_path.get(str(p), False) for p in paths),
        }
        row["raw_threshold"] = cc.threshold_from(labs) if n >= 2 else None
        rows.append(row)
    df = pl.DataFrame(rows)
    df.write_csv(f"{TABLES}/v3_catalogue_colour_priors_by_style.csv")

    summary_rows = []
    for cls in ("solid", "patterned"):
        sub = df.filter((pl.col("class") == cls) & pl.col("raw_threshold").is_not_null())
        n_styles = sub.height
        excluded = df.filter((pl.col("class") == cls) & pl.col("raw_threshold").is_null()).height
        used_pooled_fallback = n_styles < MIN_STYLES_FOR_PRIOR
        if used_pooled_fallback:
            all_valid = df.filter(pl.col("raw_threshold").is_not_null())
            median = float(all_valid["raw_threshold"].median()) if all_valid.height else None
        else:
            median = float(sub["raw_threshold"].median())
        summary_rows.append(
            {
                "class": cls,
                "n_styles_with_threshold": n_styles,
                "n_excluded_lt_2_photos": excluded,
                "median_threshold": median,
                "used_pooled_fallback": used_pooled_fallback,
            }
        )
    summary = pl.DataFrame(summary_rows)
    summary.write_csv(f"{TABLES}/v3_catalogue_colour_priors_by_class.csv")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(summary)


if __name__ == "__main__":
    main()
