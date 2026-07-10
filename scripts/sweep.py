#!/usr/bin/env python3
"""Multi-seed sweep: N run_once.py subprocesses → per-seed table + summary JSON.

One subprocess per run because the trainer/model snapshot CONFIG at import
time — a fresh process is the only clean way to apply per-run --set overrides.
Runs are sequential on purpose: parallel TF processes contend for the same
GPU/accelerator and would skew the wall-time numbers.

The summary is what config comparisons read: median/min/max batches (the
browser-budget currency), how many seeds hit the maxBatches fallback, and the
worst-seed grasp rate (the reliability dial — side-binding collapse shows up
as one seed cratering, not as a bad mean).

    python scripts/sweep.py --preset desktop --seeds 5 --name baseline-desktop
    python scripts/sweep.py --seeds 0,3,7 --set model.learningRate:0.008 --name lr8
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-seed mini-vla training sweep.")
    p.add_argument("--preset", choices=["desktop", "mobile"], default="desktop")
    p.add_argument("--seeds", default="5",
                   help="seed count ('5' → 0..4) or explicit list ('0,3,7')")
    p.add_argument("--set", nargs="*", default=[], metavar="path:value",
                   help="CONFIG knob overrides forwarded to every run")
    p.add_argument("--max-batches", type=int, default=800,
                   help="per-run converge.maxBatches override (high default so the "
                        "sweep sees TRUE convergence, not the 800 fallback)")
    p.add_argument("--probe", type=int, default=25)
    p.add_argument("--eval-episodes", type=int, default=32)
    p.add_argument("--name", required=True,
                   help="sweep label — results land in results/<name>/")
    return p.parse_args()


def parse_seeds(spec: str) -> list[int]:
    if "," in spec:
        return [int(s) for s in spec.split(",") if s.strip() != ""]
    return list(range(int(spec)))


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    out_dir = RESULTS_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    t0 = time.time()
    for seed in seeds:
        out = out_dir / f"seed{seed}.json"
        cmd = [
            sys.executable, str(RUN_ONCE),
            "--preset", args.preset,
            "--seed", str(seed),
            "--max-batches", str(args.max_batches),
            "--probe", str(args.probe),
            "--eval-episodes", str(args.eval_episodes),
            "--out", str(out),
        ]
        if args.set:
            cmd += ["--set", *args.set]
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"[sweep] seed {seed} FAILED (exit {r.returncode}) — aborting")
            sys.exit(1)
        results.append(json.loads(out.read_text()))

    # ── per-seed table ────────────────────────────────────────────────────────
    print(f"\n[sweep] {args.name}: preset={args.preset} overrides={args.set or '—'}")
    hdr = (f"{'seed':>4}  {'batches':>7}  {'stop':>8}  {'reach':>7}  {'carry':>7}  "
           f"{'color':>6}  {'grip':>6}  {'grasp':>6}  {'browser':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['seed']:>4}  {r['batches']:>7}  "
              f"{'FALLBACK' if r['hitFallback'] else 'loss':>8}  "
              f"{r['reachLoss']:>7.4f}  {r['carryLoss']:>7.4f}  "
              f"{r['colorAcc'] * 100:>5.0f}%  {r['gripAcc'] * 100:>5.0f}%  "
              f"{r['graspRate'] * 100:>5.0f}%  {r['estBrowserSeconds']:>7.1f}s")

    # ── summary ───────────────────────────────────────────────────────────────
    batches = [r["batches"] for r in results]
    grasps = [r["graspRate"] for r in results]
    summary = {
        "preset": args.preset,
        "overrides": args.set,
        "seeds": seeds,
        "medianBatches": statistics.median(batches),
        "minBatches": min(batches),
        "maxBatches": max(batches),
        "fallbackCount": sum(r["hitFallback"] for r in results),
        "meanGraspRate": statistics.mean(grasps),
        "worstGraspRate": min(grasps),
        "medianReachLoss": statistics.median(r["reachLoss"] for r in results),
        "medianEstBrowserSeconds": statistics.median(
            r["estBrowserSeconds"] for r in results
        ),
        "overBudgetCount": sum(r["overBudget"] for r in results),
    }
    print(f"\nbatches median {summary['medianBatches']:.0f} "
          f"(min {summary['minBatches']}, max {summary['maxBatches']}, "
          f"{summary['fallbackCount']}/{len(seeds)} hit fallback)")
    print(f"grasp   mean {summary['meanGraspRate'] * 100:.0f}%  "
          f"worst-seed {summary['worstGraspRate'] * 100:.0f}%")
    print(f"budget  median est. browser {summary['medianEstBrowserSeconds']:.0f}s "
          f"— {summary['overBudgetCount']}/{len(seeds)} seeds over 30s")
    print(f"[sweep] total wall {time.time() - t0:.0f}s → {out_dir}/")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
