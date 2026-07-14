#!/usr/bin/env python3
"""One-at-a-time (OAT) hyperparameter screen around the current config.

Sweeps each Tier-1/Tier-2 knob independently against a single shared BASELINE
(the current sin/cos bundle = no overrides), N seeds per point, and prints one
ranked comparison table plus a summary JSON. Each run is a fresh run_once.py
subprocess (the trainer snapshots CONFIG at import, so per-run --set overrides
need their own process), sequential to keep the timing honest.

WHY OAT, not a grid: a full grid over ~12 knobs is thousands of runs; OAT finds
which knobs individually move the needle in ~dozens, and the winners get a
proper multi-seed / interaction follow-up.

Screen defaults chosen for SIGNAL, not final numbers:
  • probes OFF (--probe 0 to run_once): the periodic probe draws perturb the RNG
    trajectory and muddy seed-to-seed comparison (this bit us on the sin/cos
    seed-0 result); run_once still emits ONE final probe for the reach/grip
    diagnostic.
  • 3 seeds: enough to catch a knob that helps/hurts, cheap enough to be fast.
    Promote winners to a 5-seed confirm (scripts/sweep.py) before trusting them.

BUDGET ACCOUNTING (the load-bearing bit): browser time = load + batches ×
per-batch-cost. The plain [budget] estimate assumes a FIXED per-batch cost
calibrated at imgSize 64 / batchSize 32, so it is WRONG for the imgSize and
batchSize knobs — those change per-batch compute. This script recomputes a
cost-weighted estimate: cost_factor ≈ (imgSize/64)² · (batchSize/32) (conv cost
is ~quadratic in resolution, ~linear in batch), so a knob that cuts batch count
but quadruples per-batch cost is correctly shown as a budget REGRESSION.

    python scripts/param_sweep.py --dry-run              # list the plan, no training
    python scripts/param_sweep.py --tiers 1 --seeds 3    # Tier-1 screen
    python scripts/param_sweep.py --tiers both --include-ceiling
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_ONCE = ROOT / "scripts" / "run_once.py"
RESULTS_DIR = ROOT / "results"
sys.path.insert(0, str(ROOT))

from train import (  # noqa: E402  (single source for the budget calibration)
    BROWSER_BATCHES_PER_SEC,
    BROWSER_BUDGET_SECONDS,
    BROWSER_LOAD_SECONDS,
)

# ── the OAT grid ────────────────────────────────────────────────────────────
# Each knob maps to a list of variant CONFIGS to try against the baseline. A
# config is a dict of {dotted CONFIG path: value} — usually one override, but a
# few carry a paired override (imgSize must scale renderSize to keep the tuned
# antialiasing headroom, renderSize ≈ 4× imgSize). The baseline VALUE of each
# knob is the shared BASELINE row and is NOT re-run here.
TIER1: dict[str, list[dict]] = {
    # imgSize DOWN — the primary per-batch-cost lever (cheaper batches). Open
    # question: does grasp precision hold at lower res now that sin/cos +
    # attention carry more of the load (the "32 was too blurry" finding predates
    # both)? renderSize 256 stays ≥4× at 48/56, so no pairing needed going down.
    "imgSize": [{"model.imgSize": 48}, {"model.imgSize": 56}],
    # LR schedule peak — mirror-balanced batches de-risked the high-LR collapse,
    # so there may be headroom above 0.008 to converge in fewer batches.
    "lrPeak": [{"trainer.lrSchedule.peak": 0.010}, {"trainer.lrSchedule.peak": 0.012}],
    # convergence stop dial — maps the grasp-vs-batches frontier directly (looser
    # = fewer batches / coarser grasp; tighter = more batches / better grasp).
    "convergeLoss": [{"trainer.converge.loss": 0.012}, {"trainer.converge.loss": 0.020}],
    # attention-map loss weight — sharper map faster → faster reach convergence.
    "mapLossWeight": [{"model.mapLossWeight": 2.0}, {"model.mapLossWeight": 2.5}],
    # gripper loss weight — the remaining worst-seed failures are gripper, not
    # reach, so this targets reliability (pairs with graspFrac below).
    "gripperLossWeight": [{"model.gripperLossWeight": 0.8}, {"model.gripperLossWeight": 1.0}],
}

TIER2: dict[str, list[dict]] = {
    # batchSize UP — may buy reliability + fewer batches, but raises per-batch
    # cost 1.5× (cost-weighted below, so a false "fewer batches" win is caught).
    "batchSize": [{"trainer.batchSize": 48}],
    # LR schedule shape.
    "lrWarmup": [{"trainer.lrSchedule.warmupBatches": 20}, {"trainer.lrSchedule.warmupBatches": 60}],
    "lrDecayHalfLife": [{"trainer.lrSchedule.decayHalfLife": 100}, {"trainer.lrSchedule.decayHalfLife": 250}],
    # soft-argmax coordinate gain — reach precision knob.
    "attnCoordGain": [{"model.attnCoordGain": 24}, {"model.attnCoordGain": 48}],
    # near-target sampling — how much vision sees close-in views (closed-loop
    # precision near the grasp).
    "nearTargetFrac": [{"trainer.nearTargetFrac": 0.35}, {"trainer.nearTargetFrac": 0.65}],
    "nearTargetStd": [{"trainer.nearTargetStd": 0.35}, {"trainer.nearTargetStd": 0.7}],
    # grasp-positive fraction — co-lever with gripperLossWeight for reliability.
    "graspFrac": [{"trainer.graspFrac": 0.2}, {"trainer.graspFrac": 0.45}],
}

# imgSize UP — a QUALITY-CEILING datapoint only (~4× per-batch cost → cannot meet
# the browser budget; renderSize paired to 512 to keep 4× headroom). Off unless
# --include-ceiling. See the 128 discussion: down is the shippable direction.
CEILING: dict[str, list[dict]] = {
    "imgSizeCeiling": [{"model.imgSize": 128, "trainer.renderSize": 512}],
}

# CONFIRM (--tiers confirm) — the OAT screen's winners STACKED, for a 5-seed
# check that the two independent wins compose: imgSize=48 (budget: cost×0.56)
# + attnCoordGain=48/64 (grasp+reliability) + gripperLossWeight=1.0. Two gains
# to find where that knob peaks. Hypothesis: ~26s browser AND ~80% grasp.
CONFIRM: dict[str, list[dict]] = {
    "combined": [
        {"model.imgSize": 48, "model.attnCoordGain": 48, "model.gripperLossWeight": 1.0},
        {"model.imgSize": 48, "model.attnCoordGain": 64, "model.gripperLossWeight": 1.0},
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OAT hyperparameter screen for mini-vla.")
    p.add_argument("--preset", choices=["desktop", "mobile"], default="desktop")
    p.add_argument("--seeds", type=int, default=3, help="seeds per config (0..N-1)")
    p.add_argument("--tiers", choices=["1", "2", "both", "confirm"], default="both")
    p.add_argument("--include-ceiling", action="store_true",
                   help="also run the imgSize=128 quality-ceiling datapoint")
    p.add_argument("--max-batches", type=int, default=800,
                   help="per-run converge.maxBatches (high → see true convergence)")
    p.add_argument("--eval-episodes", type=int, default=32)
    p.add_argument("--name", default="param-sweep", help="results/<name>/ label")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan (configs, run count, rough time) and exit")
    return p.parse_args()


def build_grid(args: argparse.Namespace) -> dict[str, list[dict]]:
    grid: dict[str, list[dict]] = {}
    if args.tiers in ("1", "both"):
        grid.update(TIER1)
    if args.tiers in ("2", "both"):
        grid.update(TIER2)
    if args.tiers == "confirm":
        grid.update(CONFIRM)
    if args.include_ceiling:
        grid.update(CEILING)
    return grid


def cost_factor(overrides: dict) -> float:
    """Per-batch compute relative to the baseline (imgSize 64, batchSize 32):
    ~quadratic in resolution, ~linear in batch size. Approximate (ignores the
    dense/attention terms that don't scale with imgSize), but far better than
    pretending per-batch cost is constant across these two knobs."""
    img = overrides.get("model.imgSize", 64)
    bs = overrides.get("trainer.batchSize", 32)
    return (img / 64.0) ** 2 * (bs / 32.0)


def est_browser_seconds(median_batches: float, overrides: dict) -> float:
    """Cost-weighted browser train estimate for this config."""
    per_batch = (1.0 / BROWSER_BATCHES_PER_SEC) * cost_factor(overrides)
    return BROWSER_LOAD_SECONDS + median_batches * per_batch


def run_group(label: str, overrides: dict, seeds: list[int], args, out_dir: Path) -> dict:
    """Run `seeds` fresh run_once subprocesses for one config; aggregate."""
    rows = []
    for seed in seeds:
        out = out_dir / f"{label}_seed{seed}.json"
        cmd = [
            sys.executable, str(RUN_ONCE),
            "--preset", args.preset,
            "--seed", str(seed),
            "--max-batches", str(args.max_batches),
            "--probe", "0",  # screen: no periodic probes (RNG-clean); final probe still emitted
            "--eval-episodes", str(args.eval_episodes),
            "--out", str(out),
        ]
        if overrides:
            cmd += ["--set", *[f"{path}:{val}" for path, val in overrides.items()]]
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"[param-sweep] {label} seed {seed} FAILED (exit {r.returncode}) — aborting")
            sys.exit(1)
        rows.append(json.loads(out.read_text()))

    batches = [r["batches"] for r in rows]
    grasps = [r["graspRate"] for r in rows]
    med_batches = statistics.median(batches)
    return {
        "label": label,
        "overrides": overrides,
        "medianBatches": med_batches,
        "minBatches": min(batches),
        "maxBatches": max(batches),
        "fallbackCount": sum(r["hitFallback"] for r in rows),
        "meanGrasp": statistics.mean(grasps),
        "worstGrasp": min(grasps),
        "medianReach": statistics.median(r["reachLoss"] for r in rows),
        "medianGrip": statistics.median(r["gripAcc"] for r in rows),
        "estBrowser": est_browser_seconds(med_batches, overrides),
        "costFactor": cost_factor(overrides),
    }


def fmt_overrides(overrides: dict) -> str:
    return "baseline" if not overrides else ", ".join(f"{k.split('.')[-1]}={v}" for k, v in overrides.items())


def print_dry_run(grid: dict, seeds: list[int], args) -> None:
    configs = [("baseline", {})] + [(k, o) for k, ovs in grid.items() for o in ovs]
    n_runs = len(configs) * len(seeds)
    # rough wall estimate: ~12s/run at baseline cost, scaled by per-batch cost
    est = sum(len(seeds) * 12.0 * cost_factor(o) for _, o in configs)
    print(f"[param-sweep] DRY RUN — preset={args.preset} tiers={args.tiers} "
          f"seeds={seeds} ({len(configs)} configs × {len(seeds)} = {n_runs} runs)")
    print(f"[param-sweep] rough wall ≈ {est / 60:.0f} min (sequential)\n")
    cur_group = None
    for label, ov in configs:
        group = label  # label IS the knob for our grid
        if group != cur_group:
            print(f"── {group} ──" if label != "baseline" else "")
            cur_group = group
        print(f"    {fmt_overrides(ov):48s}  cost×{cost_factor(ov):.2f}")


def main() -> None:
    args = parse_args()
    grid = build_grid(args)
    seeds = list(range(args.seeds))

    if args.dry_run:
        print_dry_run(grid, seeds, args)
        return

    out_dir = RESULTS_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # shared baseline first, then each knob's variants
    baseline = run_group("baseline", {}, seeds, args, out_dir)
    results: list[tuple[str, dict]] = []  # (knob, stats)
    for knob, variants in grid.items():
        for ov in variants:
            label = knob + "__" + "_".join(f"{k.split('.')[-1]}{v}" for k, v in ov.items())
            results.append((knob, run_group(label, ov, seeds, args, out_dir)))

    # ── comparison table ────────────────────────────────────────────────────
    def row(tag: str, s: dict) -> str:
        d_batches = s["medianBatches"] - baseline["medianBatches"]
        d_grasp = s["meanGrasp"] - baseline["meanGrasp"]
        over = "!" if s["estBrowser"] > BROWSER_BUDGET_SECONDS else " "
        return (f"{tag:38s} {s['medianBatches']:>6.0f} ({d_batches:+5.0f})  "
                f"{s['meanGrasp'] * 100:>4.0f}% ({d_grasp * 100:+4.0f})  "
                f"{s['worstGrasp'] * 100:>4.0f}%  {s['medianReach']:>7.4f}  "
                f"{s['medianGrip'] * 100:>4.0f}%  {s['estBrowser']:>5.1f}s{over}")

    print(f"\n[param-sweep] {args.name}: preset={args.preset} seeds={seeds}")
    hdr = (f"{'config':38s} {'batches (Δ)':>14}  {'grasp (Δ)':>11}  "
           f"{'worst':>5}  {'reach':>7}  {'grip':>5}  {'browser':>7}")
    print(hdr)
    print("-" * len(hdr))
    print(row("baseline (sin/cos)", baseline))
    cur = None
    for knob, s in results:
        if knob != cur:
            print(f"── {knob} " + "─" * max(0, 30 - len(knob)))
            cur = knob
        print(row("  " + fmt_overrides(s["overrides"]), s))
    print(f"\n('!' = over the {BROWSER_BUDGET_SECONDS:.0f}s budget after cost-weighting; "
          f"batches Δ / grasp Δ vs baseline)")
    print(f"[param-sweep] total wall {time.time() - t0:.0f}s → {out_dir}/")

    (out_dir / "summary.json").write_text(json.dumps(
        {"baseline": baseline, "variants": [s for _, s in results]}, indent=2
    ))


if __name__ == "__main__":
    main()
