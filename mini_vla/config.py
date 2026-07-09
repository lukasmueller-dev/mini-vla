# ─────────────────────────────────────────────────────────────────────────
# VLA — the ONE place to tune the model + task.  Python port of js/src/config.ts.
#
# Every knob the pipeline exposes (model architecture, optimizer, convergence,
# task difficulty, arm geometry, rollout control) lives here. The other
# mini_vla/*.py modules read their constants from CONFIG instead of hard-coding
# literals, so tuning is a single-file edit shared by every environment.
#
# Field names are kept camelCase to mirror js/src/config.ts 1:1 — the knob sheet
# is the contract the Python→JS port skill maps across, so keeping the two
# schemas identical makes that mapping mechanical. Deep rationale for each value
# lives next to it as a comment — read those before turning a knob, several are
# load-bearing.
#
# NOTE: the demo-only knobs from config.ts (`demo`, `eta`, and the display-side
# of `render`/`rollout`) are portfolio-render concerns and live JS-side only.
# What remains here is what the MODEL + TRAINING + headless eval read.
# ─────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional


@dataclass
class ConvLayer:
    """One convolution stage of the vision encoder (relu activation)."""

    filters: int
    kernel: int
    # conv stride (default 1).
    stride: int = 1
    # "same" keeps the spatial size (default); "valid" shrinks it by the kernel.
    padding: str = "same"
    # apply 2x2 max-pool after this conv.
    pool: bool = False


@dataclass
class ModelConfig:
    # Square input resolution fed to the CNN. Raised 32→64 lets the policy
    # resolve WHERE in its placement band a block sits (block ~8px, band ~12px)
    # so it reaches precisely; at 32 the position blurred to ~3px and it could
    # only learn the per-band mean target. The CNN and attention grid adapt
    # symbolically, so this is the only downstream knob — but per-batch vision
    # compute scales ~4x vs 32. Keep renderSize ≈ 4x this.
    imgSize: int = 64
    # Adam learning rate. 0.005 won the 2026-07 sweep at batchSize 32 / imgSize
    # 64: 0.008 was collapse-prone (side-binding failure on bad seeds) and 0.003
    # measurably slower without being more reliable.
    learningRate: float = 0.005
    # Weight of the auxiliary color-classification loss vs. the action loss.
    # 0.4 was the most collapse-resistant setting in the sweep (0 side-binding
    # collapses across 7 seeds vs 1/7 at 0.2); 0.2 peaked slightly higher on
    # lucky seeds but is riskier.
    colorLossWeight: float = 0.6
    # Weight of the auxiliary attention-map loss (see model.py): cross-entropy
    # between the spatial attention map and the commanded block's grid cell.
    # WHY IT EXISTS: the action loss alone cannot train the attention — with a
    # near-uniform 16×16 map the softmax Jacobian dilutes its gradient by
    # ~1/256, and the map never sharpens. CE through the softmax has an
    # undiluted (map − label) gradient, and the supervision is free — the expert
    # already knows which block it labeled.
    mapLossWeight: float = 1.5
    # Scale of the frozen soft-argmax coordinate kernel: the fusion sees the
    # gaze as (imageCoord − 0.5) × this gain. WHY: in raw [0,1] units the
    # within-band position signal spans only ~0.16 while the other ~74 fusion
    # inputs swing ~1.0, so the coordinate pathway's gradients are ~10x smaller
    # and the action head parks on a per-side-mean policy. Plain feature
    # standardization — the kernel is frozen, so the gain is exact.
    attnCoordGain: float = 32
    # Huber transition point for the action loss. The two IK target clusters
    # (commanded block left vs. right) sit ~4.3 rad apart, so plain MSE lets the
    # rare (~1%) wrong-side pick dominate over regression precision. Huber is
    # quadratic below DELTA (precise on correct-side ~0.1-rad jitter) and linear
    # above (caps a wrong-side pick), dropping the floor to ~0.025.
    actionHuberDelta: float = 0.6
    # Weight of the auxiliary gripper-command loss (BCE on the sigmoid
    # "close now" head, see model.py). Small like the color head: the gripper is
    # a near-deterministic function of "is the effector over the block", so it
    # needs only a light nudge — but non-zero, or the head has no gradient and
    # collapses to a constant. Raised 0.3→0.6 (and trainer.graspFrac 0.15→0.3, per
    # this comment's own playbook): mirror augmentation (trainer.synth_batch)
    # halves the batch's UNIQUE scene draws (each is duplicated as its exact
    # mirror), and the already-sparse grasp-now positive class collapsed onto
    # the carry_flag shortcut ("closed iff carrying", ignoring the harder visual
    # "over the block" cue) at the old weight/frac — measured: positive-class
    # mean prediction 0.09 vs negative 0.06 (no discrimination) and 0% closed-
    # loop grasp rate. Both raised together restored discrimination (0.51 vs
    # 0.12) and grasp rate (42%, above the pre-mirror baseline's 35%).
    gripperLossWeight: float = 0.6
    # Vision CNN stack, in order. The LAST stage's output map is what the
    # language-conditioned spatial attention scores (see model.py) — its spatial
    # size sets the attention grid (64 → two pools → 16×16 here), and its
    # `filters` sets the attention query width.
    conv: list[ConvLayer] = field(
        default_factory=lambda: [
            ConvLayer(filters=8, kernel=3, pool=True),
            ConvLayer(filters=16, kernel=3, pool=True),
            ConvLayer(filters=24, kernel=3),
        ]
    )
    # Units in the fused hidden layer before the heads. The fusion input is now
    # small and structured — soft-argmax (x̂,ŷ) + attended features + language
    # vector, not a flattened feature map — so this mostly learns the
    # coordinate→angles map.
    fusionUnits: int = 64


@dataclass
class LRScheduleConfig:
    # Adam LR at batch 0 — conservative, since the side-binding collapse risk
    # (see model.learningRate's history) lives in this fragile opening phase.
    start: float = 0.003
    # Ramp target, reached at warmupBatches. 0.008 flat was collapse-prone in
    # the 2026-07 sweep; mirror-balanced batches (synth_batch pairs every scene
    # with its exact mirror) remove the side-binding failure mode that made it
    # risky, so the fast regime should be reachable once past the opening phase.
    peak: float = 0.008
    # Batches to linearly ramp start→peak.
    warmupBatches: int = 40
    # Floor the post-peak decay asymptotically approaches (inverse-time decay
    # — never fully reaches it, by design).
    floor: float = 0.004
    # Inverse-time decay half-life (batches) after the peak: lr(t) = floor +
    # (peak-floor)/(1+t/decayHalfLife), t = batches since warmupBatches.
    decayHalfLife: int = 150


@dataclass
class ConvergeConfig:
    # Handoff threshold on the trailing-window HUBER action loss. Healthy runs
    # cross 0.02 at 150-280 batches; pick-up 8c/4b converges ~0.012 at ~410.
    loss: float = 0.015
    # Trailing window (batches) the convergence mean is taken over.
    window: int = 10
    # Consecutive in-threshold batches required before declaring converged.
    streak: int = 8
    # Hard floor of batches before convergence can fire.
    minBatches: int = 100
    # Fixed-budget fallback: converge regardless of loss at this batch.
    maxBatches: int = 450


@dataclass
class TrainerConfig:
    # Samples synthesized + gradient-stepped per batch. 32 is load-bearing for
    # RELIABILITY, not just speed: batchSize 16 collapsed onto always-one-side
    # policies on bad seeds where 32 stayed healthy. Don't lower it.
    batchSize: int = 32
    # Silhouettes are drawn at this px then averaged down to imgSize — drawn at
    # target size directly the sub-pixel arm strokes alias away. Keep ≈4x imgSize
    # to preserve the tuned antialiasing headroom.
    renderSize: int = 256
    # Fraction of samples posed NEAR the commanded block's IK solution (rest
    # uniform over the full pose range). The label is pose-independent, but the
    # rendered silhouette isn't — this keeps vision trained on what the scene
    # looks like as the rollout closes in.
    nearTargetFrac: float = 0.5
    # Gaussian spread (rad) of that near-target pose jitter.
    nearTargetStd: float = 0.5
    # Fraction of samples synthesized MID-CARRY: the commanded block is rendered
    # at the effector of the sampled pose, the carry_flag input is 1, and the
    # label is the carry-phase target (REST — bring the grasped block home).
    # Lowered from 0.5: REST is a CONSTANT label regardless of scene/command, so
    # carry samples are near-trivial next to the reach subtask (language-
    # conditioned localization + coordinate→angle regression) — at 0.5 they ate
    # half the batch's gradient budget for one of the easiest facts to learn.
    # 0.3 still gives the carry-phase attention target (tracking the effector)
    # steady coverage without starving the hard subtask.
    carryFrac: float = 0.3
    # Fraction of the EMPTY-HANDED samples posed as "grasp-now" positives: the
    # commanded block's IK pose, tightly jittered so the effector sits fully over
    # the block (gripper.radius) and the gripper label is 1 ("close"). Guarantees
    # a steady stream of clean close-now examples for the gripper head. Raised
    # 0.15→0.3 alongside model.gripperLossWeight — see that field's comment for
    # why (mirror augmentation halves this class's unique-scene diversity).
    graspFrac: float = 0.30
    # Gaussian spread (rad) of the grasp-class pose jitter — tight so the
    # effector reliably stays fully over the (possibly smallest) block.
    graspJitterStd: float = 0.05
    # Chance a non-color token becomes <unk> in training, so the encoder learns
    # to shrug off unknown words in free user text.
    wordDropout: float = 0.1
    # LANGUAGE WARM-UP: text-only gradient steps run BEFORE the main
    # vision→action loop. They train ONLY the pure-text color head plus its conv
    # scorer, so the color decoding is already correct when the coupled policy
    # starts. This is the CAP: warm-up early-stops once the head's loss plateaus.
    # Set 0 to disable.
    warmupBatches: int = 200
    # Batch size for warm-up steps ONLY. Bigger on purpose: warm-up is a tiny
    # text-only graph with NO images, bound by fixed per-step dispatch overhead,
    # so a larger batch does many more samples per dispatch at almost no cost.
    warmupBatchSize: int = 256
    converge: ConvergeConfig = field(default_factory=ConvergeConfig)
    lrSchedule: LRScheduleConfig = field(default_factory=LRScheduleConfig)


@dataclass
class ArmConfig:
    # Upper-/fore-arm link lengths. Sized so the full reach circle
    # (base ± l1+l2 = ±0.58) stays inside the rendered canvas.
    l1: float = 0.32
    l2: float = 0.26
    # Arm base anchor in the y-up unit workspace.
    base: tuple[float, float] = (0.5, 0.2)
    # Upright rest pose [θ1, θ2] (straight up).
    rest: tuple[float, float] = (math.pi / 2, 0.0)
    # Pose-sampling ranges for synthesized training states. θ2 spans BOTH elbow
    # configs: floor-block IK solutions sit near |θ2|≈2.
    theta1Range: tuple[float, float] = (-0.3, math.pi + 0.3)
    theta2Range: tuple[float, float] = (-2.4, 2.4)


@dataclass
class BlockConfig:
    # Reference side length — default + fallback when a block has no size.
    ref: float = 0.12
    # Per-scene blocks randomize their side length in [min, max]. FLOOR raised to
    # 0.12 for the LEARNED gripper: below it the in-block tolerance is tighter
    # than the policy's reach floor and closed-loop grasps miss small blocks
    # (measured graspRate ~0.13 at 0.08).
    min: float = 0.12
    max: float = 0.16


@dataclass
class GripperConfig:
    # Radius (workspace units) of the effector "disk" the grasp predicate
    # (effectorOverBlock, geometry.py) must fit fully inside a block's footprint
    # before a close counts. Kept well under the smallest block's half-width.
    radius: float = 0.025
    # Sigmoid threshold above which the gripper head's output counts as
    # "closed". Used identically by the rollout grasp gate and the eval.
    threshold: float = 0.5


@dataclass
class TaskConfig:
    # Token slots per command (padded/truncated to this).
    maxSeqLen: int = 14
    # The two cleanly-reachable floor BANDS [lo, hi] blocks are placed in per
    # side (the centre is a near-singular dead zone). Inner edges (0.31/0.69) are
    # set for the LARGEST block's elbow limit.
    placeLeft: tuple[float, float] = (0.11, 0.31)
    placeRight: tuple[float, float] = (0.69, 0.89)
    # Extra clearance (workspace units) required between two same-side blocks'
    # silhouettes, on top of their (boosted) half-widths.
    minBlockGap: float = 0.03
    # Grammar sampling probabilities: chance a sentence gets a leading filler
    # word, and a trailing "please".
    fillerProb: float = 0.25
    pleaseProb: float = 0.2
    # FORM augmentation: chance each scaffolding element is DROPPED from a
    # sampled sentence (verb per sentence; article/noun per occurrence), yielding
    # compressed forms like "grab red" or bare "red". Colors never drop.
    dropVerbProb: float = 0.15
    dropArticleProb: float = 0.25
    dropNounProb: float = 0.25


@dataclass
class RolloutConfig:
    # Rollout control the headless closed-loop eval reads (mini_vla/eval.py).
    # (The display-timing knobs from config.ts stay JS-side.)
    # Proportional gain toward the predicted target each frame (0..1).
    stepGain: float = 0.08
    # Consecutive close frames required to register a grasp.
    nearFrames: int = 4
    # Joint-space closeness (rad, per joint) to the predicted carry-phase target
    # that counts as "settled".
    settleEps: float = 0.05


@dataclass
class RenderConfig:
    # Model's-eye rendering (mini_vla/render.py).
    # Isotropic workspace→canvas scale (× canvas height).
    sceneScale: float = 0.8
    # Floor line position (× canvas height).
    floorY: float = 0.86
    # Blocks render this much larger in the model's-eye silhouette than in the
    # display scene — a display-size block is only a few px after the downsample.
    silBlockScale: float = 1.3


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    arm: ArmConfig = field(default_factory=ArmConfig)
    block: BlockConfig = field(default_factory=BlockConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    render: RenderConfig = field(default_factory=RenderConfig)


# The single shared instance every other module reads (mirrors `export const
# CONFIG` in config.ts). train.py mutates fields on it before the model/trainer
# snapshot their constants — same import-order contract as the JS eval harness.
CONFIG = Config()


def override(path: str, value: float) -> None:
    """Apply a dotted `model.mapLossWeight`-style override to CONFIG in place.

    Mirrors the `?set=` knob overrides the JS eval harness accepts. Must run
    BEFORE the trainer/model read their constants.
    """
    keys = path.split(".")
    obj = CONFIG
    for k in keys[:-1]:
        obj = getattr(obj, k)
    cur = getattr(obj, keys[-1])
    # preserve int-ness for integer knobs (batchSize, imgSize, …)
    setattr(obj, keys[-1], type(cur)(value) if isinstance(cur, (int, float)) else value)
