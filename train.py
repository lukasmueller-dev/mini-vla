#!/usr/bin/env python3
"""Train the mini-vla policy headless, with optional wandb tracking.

Examples:
    python train.py                                  # defaults: 8 colors, 4 blocks
    python train.py --colors 4 --blocks 3 --wandb
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the mini-vla policy.")
    p.add_argument("--colors", type=int, default=8, choices=[2, 4, 8], help="palette size")
    p.add_argument("--blocks", type=int, default=4, choices=[2, 3, 4], help="scene density cap")
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
    from mini_vla.run_config import RunConfig, set_run_config

    for kv in args.set:
        path, _, val = kv.partition(":")
        cfg.override(path, float(val))
    set_run_config(RunConfig(numColors=args.colors, maxBlocks=args.blocks))

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
                "colors": args.colors,
                "blocks": args.blocks,
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

    print(f"[train] building model (colors={args.colors} blocks={args.blocks})…")
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
    if run is not None:
        run.log(
            {
                "eval/grasp_rate": result.graspRate,
                "eval/mean_grasp_frames": result.meanGraspFrames,
                "eval/reach_jitter": result.reachJitter,
            }
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    trainer.models.model.save_weights(args.out)
    print(f"[train] saved weights → {args.out}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
