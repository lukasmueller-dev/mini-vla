#!/usr/bin/env python3
"""One seeded training run → one JSON result file. The sweep unit.

Runs the same pipeline as train.py (preset → warmup → fit → closed-loop eval)
but emits a machine-readable result instead of prose, and adds the two numbers
train.py doesn't surface: a FINAL held-out probe at convergence (reach/carry
bucket losses — the mixed convergence loss is diluted by trivial carry samples,
so the reach bucket is the honest skill dial) and whether the run hit the
maxBatches fallback instead of genuinely converging.

Meant to be spawned by scripts/sweep.py (one subprocess per run — the trainer
snapshots CONFIG at import, so per-run overrides need a fresh process), but
works standalone:

    python scripts/run_once.py --preset desktop --seed 3 --out /tmp/r.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

# repo root importable regardless of CWD (scripts/ is not a package)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import (  # noqa: E402  (single source for the budget calibration)
    BROWSER_BATCHES_PER_SEC,
    BROWSER_BUDGET_SECONDS,
    BROWSER_LOAD_SECONDS,
)

# Bigger held-out sample for the one FINAL probe (the periodic in-training
# probes keep the cheap default) — 64/bucket halves the noise of the
# at-convergence reading the sweep compares configs on.
FINAL_PROBE_N = 64


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One seeded mini-vla training run → JSON.")
    p.add_argument("--preset", choices=["desktop", "mobile"], default="desktop")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", nargs="*", default=[], metavar="path:value",
                   help="CONFIG knob overrides, e.g. model.mapLossWeight:2.5")
    p.add_argument("--max-batches", type=int, default=None,
                   help="override converge.maxBatches (raise past 800 to see true convergence)")
    p.add_argument("--probe", type=int, default=25, help="probe cadence in batches (0=off)")
    p.add_argument("--eval-episodes", type=int, default=32)
    p.add_argument("--out", default=None, help="result JSON path (default: stdout)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Import-order contract (see train.py): overrides + preset BEFORE the
    # modules that snapshot CONFIG.
    from mini_vla import config as cfg
    from mini_vla.run_config import PRESETS, set_run_config

    for kv in args.set:
        path, _, val = kv.partition(":")
        cfg.override(path, float(val))
    rc = PRESETS[args.preset]
    set_run_config(rc)

    import tensorflow as tf

    from mini_vla.eval import closed_loop_eval
    from mini_vla.trainer import VLATrainer

    tf.keras.utils.set_random_seed(args.seed)

    trainer = VLATrainer()
    trainer.probe_every_n = args.probe
    if args.max_batches is not None:
        trainer.max_batches_override = args.max_batches

    t0 = time.time()
    trainer.fit()
    train_seconds = time.time() - t0

    # the at-convergence held-out reading (larger sample than the cadence probes)
    trainer.probe_n = FINAL_PROBE_N
    trainer.run_probe()
    final = trainer.probes[-1]

    ev = closed_loop_eval(trainer, args.eval_episodes)

    max_batches = trainer.max_batches_override or cfg.CONFIG.trainer.converge.maxBatches
    est_browser = BROWSER_LOAD_SECONDS + trainer.batches / BROWSER_BATCHES_PER_SEC

    result = {
        "preset": args.preset,
        "numColors": rc.numColors,
        "maxBlocks": rc.maxBlocks,
        "seed": args.seed,
        "overrides": args.set,
        # convergence
        "batches": trainer.batches,
        "hitFallback": trainer.batches >= max_batches,  # stopped by budget, not by loss
        "smoothLoss": trainer.smooth_loss,
        # final held-out probe (FINAL_PROBE_N per bucket)
        "reachLoss": final.buckets.get("reach"),
        "carryLoss": final.buckets.get("carry"),
        "colorAcc": final.colorAcc,
        "gripAcc": final.gripAcc,
        # closed-loop eval
        "graspRate": ev.graspRate,
        "meanGraspFrames": ev.meanGraspFrames,
        "reachJitter": ev.reachJitter,
        "evalEpisodes": ev.episodes,
        # budget
        "estBrowserSeconds": est_browser,
        "overBudget": est_browser > BROWSER_BUDGET_SECONDS,
        "trainWallSeconds": train_seconds,
        # curves, for later plotting / convergence-criterion tuning
        "lossHistory": trainer.loss_history,
        "probes": [dataclasses.asdict(p) for p in trainer.probes],
    }

    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"[run_once] seed={args.seed} batches={trainer.batches} "
              f"reach={result['reachLoss']:.4f} grasp={ev.graspRate * 100:.0f}% "
              f"→ {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
