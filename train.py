#!/usr/bin/env python3
"""Train the mini-vla policy headless, with optional wandb tracking.

Examples:
    python train.py                                  # desktop preset: 8 colors, 4 blocks
    python train.py --preset mobile --wandb          # mobile preset: 4 colors, 3 blocks
    python train.py --set model.mapLossWeight:2.5 model.learningRate:0.003
    python train.py --max-batches 600 --probe 25 --eval-episodes 32

IMPORT ORDER IS LOAD-BEARING (mirrors js/eval/main.ts): the trainer + model
snapshot CONFIG values into module constants the first time they import, so
--set overrides and the run config are applied to CONFIG/run_config BEFORE the
heavy modules are imported below.
"""

from __future__ import annotations

import argparse
import os
import time

# Browser budget calibration (see CLAUDE.md + the converge/eta notes in
# js/src/config.ts): the live in-browser demo must train-to-converge + roll out in
# under 60 s (raised from 30 s in 2026-07 for headroom — 0.015 convergence's slow
# tail ran ~37 s, over the old ceiling). WebGL does ~10 gradient batches/s on a mid
# laptop GPU; tfjs load + language warm-up ≈ 2 s. train.py prints the projected
# browser time so an architecture change that blows the budget is visible
# immediately.
BROWSER_BATCHES_PER_SEC = 10.0
BROWSER_LOAD_SECONDS = 2.0
BROWSER_BUDGET_SECONDS = 60.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the mini-vla policy.")
    p.add_argument(
        "--preset",
        choices=["desktop", "mobile"],
        default="desktop",
        help="scene-difficulty profile (desktop=8c/4b hardest, mobile=4c/3b)",
    )
    p.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="path:value",
        help="CONFIG knob overrides, e.g. model.mapLossWeight:2.5",
    )
    p.add_argument("--max-batches", type=int, default=None, help="override converge.maxBatches")
    p.add_argument("--probe", type=int, default=0, help="probe cadence in batches (0=off)")
    p.add_argument("--eval-episodes", type=int, default=24, help="closed-loop eval episodes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="checkpoints/vla.weights.h5", help="checkpoint path")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    p.add_argument("--wandb-project", default="mini-vla")
    p.add_argument("--wandb-run", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1) Apply overrides BEFORE importing the heavy modules (see module docstring).
    from mini_vla import config as cfg
    from mini_vla.run_config import PRESETS, set_run_config

    for kv in args.set:
        path, _, val = kv.partition(":")
        cfg.override(path, float(val))
    rc = PRESETS[args.preset]
    set_run_config(rc)

    # 2) Now the modules that snapshot CONFIG can be imported.
    import tensorflow as tf

    from mini_vla.trainer import VLATrainer
    from mini_vla.eval import closed_loop_eval

    tf.keras.utils.set_random_seed(args.seed)  # seeds python random, numpy, tf

    run = None
    if args.wandb:
        import dataclasses

        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config={
                "preset": args.preset,
                "colors": rc.numColors,
                "blocks": rc.maxBlocks,
                "seed": args.seed,
                "overrides": args.set,
                "model": dataclasses.asdict(cfg.CONFIG.model),
                "trainer": dataclasses.asdict(cfg.CONFIG.trainer),
            },
        )

    trainer = VLATrainer()
    trainer.probe_every_n = args.probe
    if args.max_batches is not None:
        trainer.max_batches_override = args.max_batches

    print(f"[train] building model (preset={args.preset} "
          f"colors={rc.numColors} blocks={rc.maxBlocks})…")
    t0 = time.time()
    last_print = [0.0]

    def on_update(tr: VLATrainer) -> None:
        if run is not None:
            log = {"batch": tr.batches, "loss": tr.loss, "smooth_loss": tr.smooth_loss}
            if tr.probes and tr.probes[-1].batch == tr.batches:
                pr = tr.probes[-1]
                log.update(
                    {
                        "probe/reach_loss": pr.buckets.get("reach"),
                        "probe/carry_loss": pr.buckets.get("carry"),
                        "probe/color_acc": pr.colorAcc,
                        "probe/grip_acc": pr.gripAcc,
                    }
                )
            run.log(log, step=tr.batches)
        now = time.time()
        if now - last_print[0] > 1.0 or tr.status == "converged":
            last_print[0] = now
            print(
                f"\r[train] {tr.status} b={tr.batches} loss={tr.loss:.4f} "
                f"smooth={tr.smooth_loss:.4f}",
                end="",
                flush=True,
            )

    trainer.fit(on_update)
    print(f"\n[train] {trainer.status} at {trainer.batches} batches "
          f"({time.time() - t0:.1f}s)")

    print(f"[eval] closed-loop over {args.eval_episodes} episodes…")
    result = closed_loop_eval(trainer, args.eval_episodes)
    print(f"[eval] graspRate={result.graspRate * 100:.1f}%  "
          f"meanGraspFrames={result.meanGraspFrames}  reachJitter={result.reachJitter}")

    # Projected in-browser training time — the hard < 60 s product budget the
    # ported js/ demo must meet (see CLAUDE.md). Treat OVER BUDGET as a regression.
    # The 10 batches/s calibration was measured at imgSize 64 / batchSize 32, and
    # per-batch cost is ~quadratic in imgSize · ~linear in batchSize — so weight
    # by that factor or the estimate is wrong whenever those knobs move (e.g.
    # imgSize 48 makes each batch ~0.56× the cost). Mirrors param_sweep.py.
    cost_factor = (cfg.CONFIG.model.imgSize / 64.0) ** 2 * (cfg.CONFIG.trainer.batchSize / 32.0)
    est_browser = BROWSER_LOAD_SECONDS + trainer.batches * cost_factor / BROWSER_BATCHES_PER_SEC
    over = est_browser > BROWSER_BUDGET_SECONDS
    print(f"[budget] {trainer.batches} batches → est. browser train "
          f"≈ {est_browser:.0f}s (budget {BROWSER_BUDGET_SECONDS:.0f}s) "
          f"[{'OVER BUDGET' if over else 'OK'}]")
    if over:
        print("[budget] ⚠ over the 60s in-browser budget — cut batches-to-converge "
              "or per-batch compute (imgSize / conv / batchSize). See CLAUDE.md.")

    if run is not None:
        run.log(
            {
                "eval/grasp_rate": result.graspRate,
                "eval/mean_grasp_frames": result.meanGraspFrames,
                "eval/reach_jitter": result.reachJitter,
                "budget/est_browser_seconds": est_browser,
                "budget/over": int(over),
            }
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    trainer.models.model.save_weights(args.out)
    print(f"[train] saved weights → {args.out}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
