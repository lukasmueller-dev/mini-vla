// The real thing behind the hero's "Start Training" button: an asynchronous
// behavioral-cloning loop over pick-up commands (2..4 blocks from a 2/4/8-
// color palette — see src/run-config.ts). Each batch synthesizes (scene
// layout, pose, command) states — random block placements, sentences from
// the slot grammar with ~10% word-dropout to <unk> — renders each state
// through the same silhouette pipeline the live rollout uses, labels it with
// the analytical-IK expert's ABSOLUTE target joint angles — as CIRCULAR coords
// (cosθ,sinθ per joint, recovered with atan2 at readout; see model.ts), plus
// color for the auxiliary head — and runs one trainOnBatch step. Every batch is
// MIRROR-PAIRED (synthBatch): each synthesized scene is duplicated as its exact
// horizontal mirror, side-balancing the batch by construction (countering the
// side-binding collapse) at half the render cost. Grasping is a LEARNED action:
// a sigmoid gripper head is trained (BCE) to close exactly when the effector is
// fully over the commanded block — the shared effectorOverBlock predicate
// (geometry.ts) — and the rollout turns its rising edge into the physical grasp
// (see Hero.tsx), instead of the old bare proximity snap.
//
// The carry phase is policy-driven, so samples are CARRY-CONDITIONED: a
// carryFrac share render the commanded block IN THE GRIPPER of the sampled
// pose, set the carry_flag input to 1, and the label flips to the carry-
// phase target — REST, bringing the grasped block home. "Is a block in my
// hand" reaches the network as gripper proprioception (the flag) AND as the
// carried block's pixels; the rollout feeds the same flag from its own grasp
// state (see model.ts on why the flag exists). Apart from the carried block
// riding at the effector, labels
// do NOT depend on the (randomized, for robustness) rendered pose — the
// network learns "given this scene, command and carry state, where does the
// arm need to end up," not also implicitly infer the current pose and
// subtract. The Rollout arm computes its own delta from the predicted target
// against its actual known current pose (see Hero.tsx's drawArm) and steps
// toward it; the plotted curve is the genuine action loss (Huber regression
// to that absolute target), and the Rollout is driven purely by
// model.predict on its own live silhouette view.
//
// Training stops by itself once the trailing-window action loss crosses the
// convergence threshold — that switches the hero into "try it" mode where
// user-typed sentences drive the policy. It can also be paused/resumed.
//
// THREADING: this class is environment-agnostic — it normally runs inside
// src/trainer.worker.ts (OffscreenCanvas, off the main thread, so the
// gradient loop never contends with the 60fps display), with src/
// trainer.ts as the main-thread proxy Hero talks to. On browsers without
// module-worker/OffscreenCanvas support the proxy instantiates it inline on
// the main thread instead (DOM canvases, the pre-worker behavior).
//
// tfjs is dynamically imported on first start() so the landing page loads
// without the ~1MB bundle; batches are paced with awaits between steps so
// whichever thread hosts the loop stays responsive.

import { CONFIG } from "./config";
import {
  REST,
  THETA1_RANGE,
  THETA2_RANGE,
  clamp,
  effectorOverBlock,
  fk,
  ikToX,
} from "./geometry";
import { paintSilhouette } from "./scene";
import {
  COLORS,
  LABEL_TOKEN_IDS,
  MAX_SEQ_LEN,
  PAD,
  UNK,
  blockOfColor,
  randomLayout,
  sampleCommand,
  type Layout,
} from "./examples";
import {
  ACTION_HUBER_DELTA,
  ATTN_GRID,
  IMG_SIZE,
  buildVLAModel,
  type VLAModels,
  type TF,
} from "./model";
import { EMBED_DIM } from "./vocab.gen";
import { embeddingMatrix, loadEmbeddings } from "./embeddings";

import type * as tfType from "@tensorflow/tfjs";

// All tuning knobs live in src/config.ts — the rationale for each value is
// documented there. Aliased to locals so the loop body below reads the same.
const BATCH_SIZE = CONFIG.trainer.batchSize;
const RENDER_SIZE = CONFIG.trainer.renderSize;
const BATCH_GAP_MS = CONFIG.trainer.batchGapMs;
const NEAR_TARGET_FRAC = CONFIG.trainer.nearTargetFrac;
const NEAR_TARGET_STD = CONFIG.trainer.nearTargetStd;
const WORD_DROPOUT = CONFIG.trainer.wordDropout;
const CARRY_FRAC = CONFIG.trainer.carryFrac;
const GRASP_FRAC = CONFIG.trainer.graspFrac;
const GRASP_JITTER_STD = CONFIG.trainer.graspJitterStd;
const GRIP_RADIUS = CONFIG.gripper.radius;
const GRIP_THRESHOLD = CONFIG.gripper.threshold;
const WARMUP_BATCHES = CONFIG.trainer.warmupBatches;
const WARMUP_BATCH_SIZE = CONFIG.trainer.warmupBatchSize;

// Convergence: the mean action loss over a short trailing WINDOW of batches
// stays under CONVERGE_LOSS for CONVERGE_STREAK consecutive batches (after
// MIN_BATCHES warmup) → training ends and unlocks "try it" mode. MAX_BATCHES is
// the fixed-budget fallback. Threshold is on the HUBER action loss (model.ts).
export const CONVERGE_LOSS = CONFIG.trainer.converge.loss;
const CONVERGE_WINDOW = CONFIG.trainer.converge.window;
const CONVERGE_STREAK = CONFIG.trainer.converge.streak;
const MIN_BATCHES = CONFIG.trainer.converge.minBatches;
const MAX_BATCHES = CONFIG.trainer.converge.maxBatches;

// Zero-loss / context-loss guard (see the training loop + installGLWatchdog).
// A real Huber action loss for this task floors ~0.012 and is never exactly 0,
// so a loss at/below LOSS_FLOOR (or non-finite) is NON-PHYSICAL — the signature
// of a silently-dead WebGL context (WebKit returns zeros instead of throwing).
// Such a loss must never count toward convergence, and NONPHYSICAL_LIMIT of
// them in a row stops the run to reason "context" rather than looping to
// MAX_BATCHES producing a garbage policy.
const LOSS_FLOOR = 1e-4;
const NONPHYSICAL_LIMIT = 8;

// Main-optimizer LR schedule (see config.ts lrSchedule): a linear ramp
// start→peak over warmupBatches, then inverse-time decay toward floor.
const LR_START = CONFIG.trainer.lrSchedule.start;
const LR_PEAK = CONFIG.trainer.lrSchedule.peak;
const LR_WARMUP_BATCHES = CONFIG.trainer.lrSchedule.warmupBatches;
const LR_FLOOR = CONFIG.trainer.lrSchedule.floor;
const LR_DECAY_HALFLIFE = CONFIG.trainer.lrSchedule.decayHalfLife;

/** Main-optimizer Adam LR at this batch index: linear ramp LR_START→LR_PEAK
    over LR_WARMUP_BATCHES, then inverse-time decay toward LR_FLOOR. The
    collapse risk a flat high LR carries (config.ts's learningRate history)
    lives in the fragile OPENING phase — ramping past it (once mirror-balanced
    batches have de-risked that phase) reaches the faster regime without
    starting training on the cliff edge. */
function scheduledLr(batch: number): number {
  if (batch < LR_WARMUP_BATCHES)
    return LR_START + (LR_PEAK - LR_START) * (batch / LR_WARMUP_BATCHES);
  const t = batch - LR_WARMUP_BATCHES;
  return LR_FLOOR + (LR_PEAK - LR_FLOOR) / (1 + t / LR_DECAY_HALFLIFE);
}

/** Recover (θ1, θ2) from the model's 4 circular action outputs (see model.ts).
    atan2 reads only the DIRECTION, so the unconstrained radius is harmless. θ1
    is un-wrapped into solveIK's [-π/2, 3π/2) band (geometry.ts) so the rollout,
    which steps proportionally FROM the current pose, moves the short way round
    (a raw atan2 in (-π, π] could report a near-π target as its negative twin). */
function anglesFromCircular(
  cos1: number,
  sin1: number,
  cos2: number,
  sin2: number
): [number, number] {
  let t1 = Math.atan2(sin1, cos1);
  while (t1 < -Math.PI / 2) t1 += 2 * Math.PI;
  const t2 = Math.atan2(sin2, cos2);
  return [t1, t2];
}

/** Signed circular difference (pred − label) wrapped into (-π, π] — so the
    probe's angle-space Huber is correct across the wrap seam. */
function angleErr(pred: number, label: number): number {
  const m = 2 * Math.PI;
  // JS % can go negative; ((x % m) + m) % m reproduces Python's floor-mod.
  return (((pred - label + Math.PI) % m) + m) % m - Math.PI;
}

/**
 * WebKit workaround: turn off tfjs's fence-based GPU readback.
 *
 * backend_webgl resolves every `tensor.data()` by awaiting createAndWaitForFence(),
 * which creates a `fenceSync` and polls `clientWaitSync` until the GPU signals.
 * tfjs allocates one sync object per read and never calls `deleteSync`. On iOS
 * WebKit the fence stops signalling after a modest number of reads and the
 * promise simply never settles. `trainOnBatch` reads one loss per output head,
 * and this model has five, so the language warm-up (single-head) survives, the
 * first full step survives, and the SECOND full step wedges forever: thread
 * alive, status "training", `batches` frozen at 1. The page eventually loses its
 * web process to the leaked textures.
 *
 * With the flag off, createFence reports "passed" immediately and the texture is
 * downloaded synchronously. That read BLOCKS — which is precisely why the loop
 * belongs in the worker (trainer.worker.ts), where it costs the UI nothing.
 *
 * Gate notes: NOT tfjs's own `IS_SAFARI` flag. That tests `navigator.vendor`,
 * and WorkerNavigator does not expose it (`vendor`/`vendorSub`/`productSub` are
 * window-only), so it reads FALSE inside the worker on iOS — exactly where this
 * must fire. `userAgent` IS exposed there.
 *
 * The gate is "is this WebKit-on-iOS/iPadOS", NOT "is this Safari": the fence
 * bug is a WebKit-on-iOS defect, and EVERY iOS/iPadOS browser is WebKit
 * underneath whatever chrome it wears — Safari, plus the re-skins Chrome
 * (CriOS), Edge (EdgiOS) and Firefox (FxiOS). All share the bug, so all must
 * get the flag; a Safari-only gate would let e.g. iPad Chrome's CriOS build
 * wedge. Real DESKTOP Chrome/Edge must be EXCLUDED (the fence API works there,
 * and disabling it forces slow synchronous readbacks): those carry a
 * "Chrome/"/"Edg/" token with no iOS marker.
 *
 * LIMITATION — the iPad-masquerading-as-desktop case: iPadOS defaults to
 * desktop-class browsing, so iPad Safari sends a *macOS* UA with no iOS marker
 * at all. That still gets the flag here (via the AppleWebKit-and-not-Chromium
 * branch, harmless on real desktop Safari too). But an iPad *Chrome* in that
 * mode can send a UA indistinguishable from desktop Chrome ("Chrome/" +
 * "Safari/537.36", no CriOS) and slips through. The usual escape hatch
 * (navigator.platform==='iPad' || (platform==='MacIntel' && maxTouchPoints>1))
 * is out of reach: WorkerNavigator exposes neither maxTouchPoints nor a
 * trustworthy platform, and this runs in the worker — so we accept that
 * residual gap rather than risk disabling the fence on real desktop Chrome.
 *
 * Verified on iPhone (iOS 18.7 / Safari 26.5): converges in ~6s with this,
 * hangs at batch 1 without it. No effect on desktop Chrome, where the gate is
 * false.
 */
function maybeDisableWebGLFence(tf: typeof tfType): void {
  const ua = typeof navigator === "undefined" ? "" : (navigator.userAgent ?? "");
  if (!/AppleWebKit/.test(ua)) return; // Gecko / non-WebKit — no fence bug
  // Explicit iOS/iPadOS markers: the platform tokens (Safari + any mobile-mode
  // browser) OR one of the WebKit re-skins' tokens.
  const iosWebKit = /iPad|iPhone|iPod|CriOS|EdgiOS|FxiOS/.test(ua);
  // AppleWebKit with no iOS marker and no desktop-Chromium token ⇒ Safari:
  // desktop macOS Safari, or iPad Safari sending its default macOS UA. Firing
  // here is harmless on desktop Safari (readbacks just go synchronous, off the
  // main thread in the worker) and is what covers that iPad-as-macOS case.
  // Real desktop Chrome/Edge (and Android Chrome) carry Chrome//Edg/ and are
  // excluded; desktop Firefox has no AppleWebKit token and returned above.
  const safariLike = !/Chrome|Chromium|Edg\//.test(ua);
  if (iosWebKit || safariLike)
    tf.env().set("WEBGL_FENCE_API_ENABLED", false);
}

export type TrainerStatus =
  | "idle"
  | "loading"
  | "training"
  | "paused"
  | "converged"
  | "error";

/**
 * Why the trainer is in "error". Hosts key their recovery affordance off this,
 * and the difference matters:
 *
 *  - "assets" — the embedding fetch failed or the fetched assets didn't match
 *    this build. A RETRY CAN SUCCEED: loadEmbeddings un-caches a rejected
 *    promise, so calling start() again refetches. Offer "Try again".
 *  - "worker" — the worker script failed to load or evaluate (typically a
 *    content-hashed chunk that a redeploy deleted out from under an open tab).
 *    RETRYING IN-PAGE IS FUTILE: a fresh `new Worker(...)` resolves the same
 *    dead URL. Only a page reload can help. Offer "Reload".
 *  - "train" — a gradient step threw and even the cpu-backend fallback failed.
 *    Rare; treat like "assets" (a retry rebuilds the model from scratch).
 *  - "context" — the worker's WebGL context was lost (see installGLWatchdog).
 *    On iPadOS WebKit the OS can evict a worker's GL context — dead on arrival
 *    (tf.ready resolves against a dead context) or mid-run — and WebKit does
 *    NOT throw: readbacks silently return zeros, so a batch's loss reads as 0
 *    and, unguarded, a run of zeros would false-converge the streak. This is
 *    the SILENT-ZEROS path, distinct from "train" (a THROWN gradient error).
 *    RETRYING IN-PAGE IS FUTILE: a lost context never recovers without a new
 *    backend, and a fresh one is liable to be evicted again under the same
 *    pressure. Offer "Reload" (like "worker").
 */
export type TrainerError = "assets" | "worker" | "train" | "context";

/** One policy inference: the action plus the "where the model looks" viz,
    all read from a single forward pass of the viz twin (see model.ts). */
export interface PredictResult {
  /** Predicted ABSOLUTE target joint angles. */
  target: [number, number];
  /** Spatial attention over the ATTN_GRID×ATTN_GRID vision cells, row-major,
      normalized so the peak cell is 1 (ready to use as overlay alpha): the
      map tracking the commanded block, empty-handed or while carrying. */
  attn: number[];
  /** The map's soft-argmax — where the model looks, in [0,1] image
      coords of the silhouette view (x right, y down). */
  xy: [number, number];
  /** The gripper head's sigmoid output (0=open → 1=closed). The rollout
      turns its rising edge over a block into the physical grasp (Hero.tsx). */
  grip: number;
}

/** What the color head decodes from a token sequence (vision zeroed) —
    the "decoded target" readout and try-it routing. */
export interface DecodedCommand {
  color: number;
  colorProb: number;
}

/** One held-out evaluation snapshot, taken every `probeEveryN` batches when
    probing is on (the headless sweep harness turns it on; the demo leaves it
    off). The batch's single action-loss scalar hides WHERE a run fails —
    these per-phase buckets are the dials the experiment plan reads. */
export interface ProbeRow {
  batch: number;
  /** Mean Huber action loss per phase bucket ("reach" / "carry"), over
      `probeN` freshly synthesized held-out samples each. */
  buckets: Record<string, number>;
  /** Color-head accuracy over all probe samples (argmax vs. label). */
  colorAcc: number;
  /** Gripper-head accuracy over all probe samples ((sigmoid≥threshold) vs.
      the 0/1 close label) — the "did the gripper actually learn" dial: it
      should sit high AND the head's mean output should not collapse to a
      constant (measured separately in the sweep). */
  gripAcc: number;
}

/** 2D context in either environment (worker OffscreenCanvas / DOM canvas). */
type Ctx2D = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;

/** A square 2D canvas in whichever flavor this environment provides. */
function make2d(size: number): { canvas: CanvasImageSource; ctx: Ctx2D } {
  if (typeof OffscreenCanvas !== "undefined") {
    const canvas = new OffscreenCanvas(size, size);
    return {
      canvas,
      ctx: canvas.getContext("2d", {
        willReadFrequently: true,
      }) as OffscreenCanvasRenderingContext2D,
    };
  }
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  return {
    canvas,
    ctx: canvas.getContext("2d", { willReadFrequently: true })!,
  };
}

export class VLATrainerCore {
  status: TrainerStatus = "idle";
  /** Set whenever status is "error"; null otherwise. See TrainerError. */
  errorReason: TrainerError | null = null;
  /** Real action loss (Huber) from the latest trainOnBatch (NaN before the first). */
  loss = NaN;
  /** Mean action loss over the last CONVERGE_WINDOW batches — the low-lag
      signal convergence is judged on (NaN before the first batch). */
  smoothLoss = NaN;
  /** First recorded loss — the normalization anchor for the UI. */
  initialLoss = NaN;
  lossHistory: number[] = [];
  batches = 0;
  /** Probe cadence in batches; 0 (default) = off, so the demo path pays
      nothing. The vla-lab harness sets it (~25) for sweep runs. */
  probeEveryN = 0;
  /** Harness-only override of the converge.maxBatches fallback (null = use
      CONFIG). Lets a sweep run PAST the demo budget to gauge where a config
      genuinely converges — the number the demo budget is then set from. */
  maxBatchesOverride: number | null = null;
  /** Held-out samples per (task, phase) bucket per probe. */
  probeN = 24;
  probes: ProbeRow[] = [];

  private tf: TF | null = null;
  private models: VLAModels | null = null;
  /** A separate inference model holding a FROZEN copy of the policy weights,
      refreshed only at snapshotPolicy() calls (each demo-cycle boundary and on
      convergence). The live rollout attempt runs against this so it sees a
      fixed policy for its whole cycle while the main model keeps training. */
  private frozenModels: VLAModels | null = null;
  private running = false;
  private paused = false;
  private convergeStreak = 0;
  /** Consecutive NON-PHYSICAL batch losses (0 / non-finite). A real Huber
      action loss floors ~0.012 and is never 0, so a run of these means the GL
      context is silently dead (WebKit returns zeros, doesn't throw) — see the
      training loop's zero-loss guard. Reset per run. */
  private nonPhysicalStreak = 0;
  /** Latched true once the worker's WebGL context is lost — checked immediately
      after tf.ready() and via a webglcontextlost listener (installGLWatchdog).
      A lost context never recovers without a new backend, so this does NOT
      reset across runs: every subsequent start() fails fast to reason
      "context". */
  private glContextLost = false;
  /** True once the webglcontextlost listener is attached, so restarts don't
      stack duplicate listeners on the persistent backend canvas. */
  private glWatchdogInstalled = false;
  /** Guards against overlapping start() calls after a quick reset+restart. */
  private runId = 0;
  private sceneCanvas: CanvasImageSource | null = null;
  private sceneCtx: Ctx2D | null = null;
  private thumbCtx: Ctx2D | null = null;

  /** Total expert-labeled examples consumed so far. */
  get samples() {
    return this.batches * BATCH_SIZE;
  }

  get ready() {
    return (
      (this.status === "training" ||
        this.status === "paused" ||
        this.status === "converged") &&
      this.models !== null &&
      this.batches > 0
    );
  }

  private ensureCanvases() {
    if (this.sceneCtx && this.thumbCtx) return;
    const scene = make2d(RENDER_SIZE);
    this.sceneCanvas = scene.canvas;
    this.sceneCtx = scene.ctx;
    this.thumbCtx = make2d(IMG_SIZE).ctx;
  }

  /** Render a state through the silhouette pipeline; returns IMG_SIZE RGBA.
      `carry` draws that block at the effector instead of its rest spot —
      the carry-phase state cue. */
  private renderPose(
    a1: number,
    a2: number,
    layout: Layout,
    carry: number | null = null
  ): ImageData {
    this.ensureCanvases();
    paintSilhouette(this.sceneCtx!, RENDER_SIZE, a1, a2, layout, carry);
    const tctx = this.thumbCtx!;
    tctx.imageSmoothingEnabled = true;
    tctx.clearRect(0, 0, IMG_SIZE, IMG_SIZE);
    tctx.drawImage(
      this.sceneCanvas!,
      0,
      0,
      RENDER_SIZE,
      RENDER_SIZE,
      0,
      0,
      IMG_SIZE,
      IMG_SIZE
    );
    return tctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
  }

  /** Roughly-Gaussian noise (sum of two uniforms). */
  private gauss(std: number) {
    return (Math.random() + Math.random() - 1) * std * 2;
  }

  /** Write the attention-map label for one sample into ysM: BILINEAR weights
      over the (up to) 4 grid cells around a point given in the silhouette
      view's [0,1] (u, v) image coords. Soft on purpose — a hard one-hot label
      would train the map toward a one-hot, quantizing the soft-argmax readout
      to cell centers (measured: ~0.15 median reach error); the bilinear
      label's CE optimum is a distribution whose EXPECTATION is the point's
      continuous position, so sub-cell precision survives. */
  private writeMapLabel(ysM: Float32Array, base: number, u: number, v: number) {
    const G = ATTN_GRID;
    // continuous cell coords (cell centers sit at integer +0.5 / G)
    const cx = u * G - 0.5;
    const cy = v * G - 0.5;
    const j0 = Math.floor(cx);
    const i0 = Math.floor(cy);
    const fx = cx - j0;
    const fy = cy - i0;
    for (const [i, j, w] of [
      [i0, j0, (1 - fy) * (1 - fx)],
      [i0, j0 + 1, (1 - fy) * fx],
      [i0 + 1, j0, fy * (1 - fx)],
      [i0 + 1, j0 + 1, fy * fx],
    ]) {
      if (i < 0 || i >= G || j < 0 || j >= G || w === 0) continue;
      ysM[base + i * G + j] += w;
    }
  }

  /** A resting block's visual center in silhouette [0,1] (u, v) coords.
      Mirrors paintSilhouette's geometry: sceneMap placement, the block's rest
      height, plus the silBlockScale boost — so the label points at the pixels
      the model actually sees. */
  private blockUV(x: number, size: number, rest = 0): [number, number] {
    const u = 0.5 + (x - 0.5) * CONFIG.render.sceneScale;
    const yCenter = rest + (size * CONFIG.render.silBlockScale) / 2;
    const v = CONFIG.render.floorY - yCenter * CONFIG.render.sceneScale;
    return [u, v];
  }

  /** The effector position in silhouette [0,1] (u, v) coords — where a
      CARRIED block renders, so it's the attention anchor for lift's carry
      phase (the only pose-DEPENDENT label; the action label stays REST). */
  private effectorUV(a1: number, a2: number): [number, number] {
    const e = fk(a1, a2);
    const u = 0.5 + (e.ex - 0.5) * CONFIG.render.sceneScale;
    const v = CONFIG.render.floorY - e.ey * CONFIG.render.sceneScale;
    return [u, v];
  }

  /** Backing arrays of a synthesized batch — the tensors trainStep and
      runProbe build from. `force` pins every sample to one phase bucket;
      the training path leaves it unset and samples both freely.

      Every batch is MIRROR-PAIRED: it synthesizes ⌊n/2⌋ scenes and fills each
      odd slot as its EXACT horizontal mirror (n odd synthesizes one extra
      unmirrored). The task is exactly left-right symmetric about the arm base
      (config.arm.base x=0.5 — fk()'s effector mirrors as ex' = 1 − ex, and the
      silhouette's pixel base sits at the image's horizontal center regardless
      of pose), so a rendered scene's column-flip is pixel-equivalent to
      re-rendering the truly mirrored scene — a second training sample for ZERO
      extra renders. Every label transforms in closed form: (θ1,θ2)→(π−θ1,−θ2)
      [REST maps to itself], the attention map's columns mirror, and
      carry/color/gripper/language are invariant. Pairing this way SIDE-BALANCES
      every batch by construction — directly countering the side-binding
      collapse — at half the render cost. */
  private synthBatch(n: number, force?: { carry?: boolean }) {
    const px = IMG_SIZE * IMG_SIZE * 3;
    const cells = ATTN_GRID * ATTN_GRID;
    const G = ATTN_GRID;
    const vis = new Float32Array(n * px);
    const lang = new Int32Array(n * MAX_SEQ_LEN);
    /** Proprioceptive carry flag per sample (1 = block in the gripper). */
    const carryF = new Float32Array(n);
    // action label = circular coords (cosθ1, sinθ1, cosθ2, sinθ2), see model.ts
    const ysA = new Float32Array(n * 4);
    const ysC = new Float32Array(n * COLORS.length);
    const ysMPick = new Float32Array(n * cells);
    /** Gripper "close now" label per sample (1 = should be closed). */
    const ysG = new Float32Array(n);

    // Synthesize one FRESH sample into slot `idx` (render + IK + every label).
    const synthOne = (idx: number) => {
      const layout = randomLayout();
      // an executable command: the acted-on color is present in the scene
      const sentence = sampleCommand(layout);
      const target = blockOfColor(layout, sentence.color);

      // carry-conditioned label (see the file header): a CARRY_FRAC share of
      // samples render the commanded block in the gripper and label the
      // carry-phase target (REST — bring it home) instead of the grasp
      const midCarry = force?.carry ?? Math.random() < CARRY_FRAC;
      // among empty-handed samples, a GRASP_FRAC share are "grasp-now"
      // positives: posed tightly at the block's IK grasp pose so the effector
      // sits fully over the block, guaranteeing a dense supply of close-now
      // examples for the gripper head (see graspFrac in config.ts).
      const graspNow = !midCarry && Math.random() < GRASP_FRAC;
      carryF[idx] = midCarry ? 1 : 0;
      let t1: number;
      let t2: number;
      if (!midCarry) {
        // reach phase: the grasp point (the IK label depends on the block's
        // SIZE too — grasp height = size/2 — so a bigger commanded block
        // resolves to a higher reach)
        [t1, t2] = ikToX(target.x, target.size, target.y ?? 0);
      } else {
        // carrying → bring it home
        [t1, t2] = REST;
      }

      let a1: number;
      let a2: number;
      if (graspNow) {
        // tight jitter around the grasp pose so effectorOverBlock is reliably
        // true — the action label stays put (t1/t2 = the grasp pose)
        a1 = clamp(t1 + this.gauss(GRASP_JITTER_STD), THETA1_RANGE[0], THETA1_RANGE[1]);
        a2 = clamp(t2 + this.gauss(GRASP_JITTER_STD), THETA2_RANGE[0], THETA2_RANGE[1]);
      } else if (Math.random() < NEAR_TARGET_FRAC) {
        a1 = clamp(t1 + this.gauss(NEAR_TARGET_STD), THETA1_RANGE[0], THETA1_RANGE[1]);
        a2 = clamp(t2 + this.gauss(NEAR_TARGET_STD), THETA2_RANGE[0], THETA2_RANGE[1]);
      } else {
        a1 = THETA1_RANGE[0] + Math.random() * (THETA1_RANGE[1] - THETA1_RANGE[0]);
        a2 = THETA2_RANGE[0] + Math.random() * (THETA2_RANGE[1] - THETA2_RANGE[0]);
      }

      // THE gripper label (the shared invariant): close iff already carrying,
      // or the effector is fully over the commanded block. Identical predicate
      // to the rollout grasp gate (Hero.tsx) — that consistency is what stops
      // the "hold closed the whole time and snap on arrival" degenerate. Most
      // far/near poses give 0; grasp-now (and mid-carry) poses give 1.
      ysG[idx] =
        midCarry || effectorOverBlock(a1, a2, target, GRIP_RADIUS) ? 1 : 0;

      // ABSOLUTE target joint angles, not a delta from the sampled pose,
      // projected onto the unit circle (recovered via atan2 at readout — see
      // model.ts). The label doesn't depend on the (randomized, for robustness)
      // pose the arm is rendered at — only on the scene, the command and the
      // carry state; the rollout computes its own delta from the recovered
      // angles against its actual known current pose (see Hero.tsx).
      ysA[idx * 4] = Math.cos(t1);
      ysA[idx * 4 + 1] = Math.sin(t1);
      ysA[idx * 4 + 2] = Math.cos(t2);
      ysA[idx * 4 + 3] = Math.sin(t2);
      ysC[idx * COLORS.length + sentence.color] = 1;
      // attention supervision, one bilinear soft label, phase-INDEPENDENT
      // (see model.ts — the action loss alone cannot sharpen the map): the
      // commanded block wherever it renders (rest spot, or the effector while
      // carried).
      const [pu, pv] = midCarry
        ? this.effectorUV(a1, a2)
        : this.blockUV(target.x, target.size, target.y ?? 0);
      this.writeMapLabel(ysMPick, idx * cells, pu, pv);

      // INVERTED intensities (background 0, content sparse positive) — fed
      // raw, the near-all-white image saturates the conv branch and the
      // model collapses onto language-only predictions.
      const img = this.renderPose(
        a1,
        a2,
        layout,
        midCarry ? sentence.color : null
      ).data;
      const base = idx * px;
      for (let p = 0; p < IMG_SIZE * IMG_SIZE; p++) {
        vis[base + p * 3] = 1 - img[p * 4] / 255;
        vis[base + p * 3 + 1] = 1 - img[p * 4 + 1] / 255;
        vis[base + p * 3 + 2] = 1 - img[p * 4 + 2] / 255;
      }

      // word-dropout: tokens that don't carry label information (articles/
      // nouns/fillers — NOT color words or verbs, see LABEL_TOKEN_IDS)
      // occasionally become <unk> so the encoder learns to handle unknown
      // words in free user text
      for (let s = 0; s < MAX_SEQ_LEN; s++) {
        let id = sentence.tokens[s];
        if (id !== PAD && !LABEL_TOKEN_IDS.has(id) && Math.random() < WORD_DROPOUT)
          id = UNK;
        lang[idx * MAX_SEQ_LEN + s] = id;
      }
    };

    // Fill `dst` as the EXACT horizontal mirror of already-filled `src` — no
    // render, no IK, just the closed-form transform (see the method docstring).
    const mirrorInto = (src: number, dst: number) => {
      // vision: flip the [IMG,IMG,3] image left-right (mirror the columns)
      const sBase = src * px;
      const dBase = dst * px;
      for (let y = 0; y < IMG_SIZE; y++)
        for (let x = 0; x < IMG_SIZE; x++) {
          const so = sBase + (y * IMG_SIZE + (IMG_SIZE - 1 - x)) * 3;
          const doff = dBase + (y * IMG_SIZE + x) * 3;
          vis[doff] = vis[so];
          vis[doff + 1] = vis[so + 1];
          vis[doff + 2] = vis[so + 2];
        }
      // language / carry / color / gripper are invariant under the mirror
      for (let s = 0; s < MAX_SEQ_LEN; s++)
        lang[dst * MAX_SEQ_LEN + s] = lang[src * MAX_SEQ_LEN + s];
      carryF[dst] = carryF[src];
      for (let c = 0; c < COLORS.length; c++)
        ysC[dst * COLORS.length + c] = ysC[src * COLORS.length + c];
      ysG[dst] = ysG[src];
      // (θ1,θ2)→(π−θ1,−θ2) in circular coords: cos(π−θ1)=−cosθ1,
      // sin(π−θ1)=sinθ1, cos(−θ2)=cosθ2, sin(−θ2)=−sinθ2.
      ysA[dst * 4] = -ysA[src * 4];
      ysA[dst * 4 + 1] = ysA[src * 4 + 1];
      ysA[dst * 4 + 2] = ysA[src * 4 + 2];
      ysA[dst * 4 + 3] = -ysA[src * 4 + 3];
      // attention map: mirror the G×G columns (row-major i*G + j)
      for (let i = 0; i < G; i++)
        for (let j = 0; j < G; j++)
          ysMPick[dst * cells + i * G + j] =
            ysMPick[src * cells + i * G + (G - 1 - j)];
    };

    const half = Math.floor(n / 2);
    for (let k = 0; k < half; k++) {
      synthOne(2 * k);
      mirrorInto(2 * k, 2 * k + 1);
    }
    if (n % 2) synthOne(n - 1);

    return {
      n,
      vis,
      lang,
      carryF,
      ysA,
      ysC,
      ysMPick,
      ysG,
      cells,
    };
  }

  /**
   * A CHEAP language-only batch: sentences + their color label, with NO
   * vision render, NO IK and NO attention-map labels. Feeds languageWarmup —
   * the whole point is that it skips everything expensive in synthBatch, so a
   * warm-up step costs a small fraction of a full step. Same command
   * distribution as training (randomLayout → sampleCommand) and the same
   * word-dropout, so the warmed head sees the real token statistics.
   */
  private synthLangBatch(n: number) {
    const lang = new Int32Array(n * MAX_SEQ_LEN);
    const ysC = new Float32Array(n * COLORS.length);
    for (let i = 0; i < n; i++) {
      const layout = randomLayout();
      const sentence = sampleCommand(layout);
      ysC[i * COLORS.length + sentence.color] = 1;
      for (let s = 0; s < MAX_SEQ_LEN; s++) {
        let id = sentence.tokens[s];
        if (id !== PAD && !LABEL_TOKEN_IDS.has(id) && Math.random() < WORD_DROPOUT)
          id = UNK;
        lang[i * MAX_SEQ_LEN + s] = id;
      }
    }
    return { n, lang, ysC };
  }

  /**
   * Language warm-up: WARMUP_BATCHES text-only gradient steps on the language
   * twin (see model.ts `lang`), run during the Loading phase before the main
   * loop. Trains ONLY the color head and its conv scorer — the vision branch
   * is not in this graph — so the color decoding is already correct when the
   * coupled vision→action policy starts, and the attention query gets a clean
   * language slot from batch 0. Near-free vs a full step; interruptible by
   * reset (runId/running are re-checked each step). Runs inside start()'s
   * try/catch, so a WebGL failure here rides the same cpu-fallback path.
   */
  private async languageWarmup(myRun: number): Promise<void> {
    const tf = this.tf!;
    // early-stop: track the initial loss and a short trailing mean; the color
    // head has effectively converged once the mean has fallen to a small
    // fraction of where it started (a config-independent gate — the initial CE
    // is ~ln(#classes), which the ratio cancels out). WARMUP_BATCHES is only
    // the hard cap.
    let initial = NaN;
    const recent: number[] = [];
    for (let k = 0; k < WARMUP_BATCHES; k++) {
      if (!this.running || this.runId !== myRun) return;
      const b = this.synthLangBatch(WARMUP_BATCH_SIZE);
      const xsLang = tf.tensor2d(b.lang, [b.n, MAX_SEQ_LEN], "int32");
      const yColor = tf.tensor2d(b.ysC, [b.n, COLORS.length]);
      try {
        // trainOnBatch already syncs to return the loss, so reading it here is
        // free — no extra GPU readback.
        const h = await this.models!.lang.trainOnBatch([xsLang], [yColor]);
        const loss = Array.isArray(h) ? (h[0] as number) : (h as number);
        if (Number.isNaN(initial)) initial = loss;
        recent.push(loss);
        if (recent.length > 10) recent.shift();
      } finally {
        xsLang.dispose();
        yColor.dispose();
      }
      // eligible to stop after a small floor of steps, once the trailing mean
      // has dropped to <10% of the initial loss
      const mean = recent.reduce((a, c) => a + c, 0) / recent.length;
      if (k >= 30 && mean < 0.1 * initial) return;
    }
  }

  /**
   * One gradient step on a freshly synthesized micro-batch.
   * Returns the batch's action loss (Huber).
   */
  private async trainStep(): Promise<number> {
    const tf = this.tf!;
    // Override Adam's LR from the schedule BEFORE this step (mirrors trainer.py
    // fit()). this.batches is 0 on the shader-warmup step, 1..N in the loop, so
    // the b-th gradient step uses scheduledLr(b) exactly as in Python. Adam
    // reads its learningRate live each applyGradients, so the mutation takes
    // effect; the field is typed on the concrete optimizer, hence the cast.
    (this.models!.model.optimizer as unknown as { learningRate: number })
      .learningRate = scheduledLr(this.batches);
    const b = this.synthBatch(BATCH_SIZE);
    const xsVision = tf.tensor4d(b.vis, [b.n, IMG_SIZE, IMG_SIZE, 3]);
    const xsLang = tf.tensor2d(b.lang, [b.n, MAX_SEQ_LEN], "int32");
    const xsCarry = tf.tensor2d(b.carryF, [b.n, 1]);
    const yAction = tf.tensor2d(b.ysA, [b.n, 4]);
    const yColor = tf.tensor2d(b.ysC, [b.n, COLORS.length]);
    const yMapPick = tf.tensor2d(b.ysMPick, [b.n, b.cells]);
    const yGrip = tf.tensor2d(b.ysG, [b.n, 1]);

    try {
      const h = await this.models!.model.trainOnBatch(
        [xsVision, xsLang, xsCarry],
        [yAction, yColor, yMapPick, yGrip]
      );
      // multi-output: [total, action, color, map, grip] — index 1 stays the
      // Huber ACTION loss the convergence logic watches
      return Array.isArray(h) ? (h[1] as number) : (h as number);
    } finally {
      xsVision.dispose();
      xsLang.dispose();
      xsCarry.dispose();
      yAction.dispose();
      yColor.dispose();
      yMapPick.dispose();
      yGrip.dispose();
    }
  }

  /** Huber loss of one scalar error, matching tf.losses.huberLoss. */
  private huber(e: number): number {
    const d = ACTION_HUBER_DELTA;
    const a = Math.abs(e);
    return a <= d ? 0.5 * a * a : d * (a - 0.5 * d);
  }

  /** argmax of one row of a flat row-major [n, width] array. */
  private static argmaxRow(a: Float32Array, row: number, width: number): number {
    let best = 0;
    for (let k = 1; k < width; k++)
      if (a[row * width + k] > a[row * width + best]) best = k;
    return best;
  }

  /**
   * Held-out per-phase evaluation — forward passes only, no gradient step.
   * Synthesizes probeN fresh samples per phase bucket (reach / carry), reads
   * the mean Huber action loss per bucket plus color-head accuracy, and
   * appends one ProbeRow. Costs ~2 batch-equivalents of forward compute.
   */
  private async runProbe(): Promise<void> {
    const tf = this.tf!;
    const buckets: Record<string, number> = {};
    let headN = 0;
    let colorHits = 0;
    let gripHits = 0;

    for (const carry of [false, true]) {
      const b = this.synthBatch(this.probeN, { carry });
      const r = tf.tidy(() => {
        const v = tf.tensor4d(b.vis, [b.n, IMG_SIZE, IMG_SIZE, 3]);
        const l = tf.tensor2d(b.lang, [b.n, MAX_SEQ_LEN], "int32");
        const c = tf.tensor2d(b.carryF, [b.n, 1]);
        const [action, color, , grip] = this.models!.model.predict([
          v,
          l,
          c,
        ]) as tfType.Tensor[];
        return {
          action: action.dataSync() as Float32Array,
          color: color.dataSync() as Float32Array,
          grip: grip.dataSync() as Float32Array,
        };
      });
      // bucket metric stays ANGLE-space Huber (comparable across experiments):
      // recover both pred & label angles from the circular coords, then Huber
      // the wrapped per-joint error.
      let sum = 0;
      for (let i = 0; i < b.n; i++) {
        const [pt1, pt2] = anglesFromCircular(
          r.action[i * 4],
          r.action[i * 4 + 1],
          r.action[i * 4 + 2],
          r.action[i * 4 + 3]
        );
        const [lt1, lt2] = anglesFromCircular(
          b.ysA[i * 4],
          b.ysA[i * 4 + 1],
          b.ysA[i * 4 + 2],
          b.ysA[i * 4 + 3]
        );
        sum += this.huber(angleErr(pt1, lt1));
        sum += this.huber(angleErr(pt2, lt2));
      }
      buckets[carry ? "carry" : "reach"] = sum / (b.n * 2);
      for (let i = 0; i < b.n; i++) {
        const am = VLATrainerCore.argmaxRow;
        if (am(r.color, i, COLORS.length) === am(b.ysC, i, COLORS.length))
          colorHits++;
        // gripper accuracy: thresholded sigmoid vs. the 0/1 close label
        if ((r.grip[i] >= GRIP_THRESHOLD ? 1 : 0) === b.ysG[i]) gripHits++;
      }
      headN += b.n;
    }

    this.probes.push({
      batch: this.batches,
      buckets,
      colorAcc: colorHits / headN,
      gripAcc: gripHits / headN,
    });
  }

  /**
   * Wire up WebGL context-loss detection at the SOURCE, against the (persistent)
   * webgl backend. On iPadOS Safari/WebKit the OS enforces a process-wide cap on
   * live WebGL contexts, and bfcache keeps closed tabs' worker contexts alive —
   * so a fresh context can be evicted the instant it's created (dead on arrival)
   * or lost mid-run. WebKit does NOT throw when it happens: GL readbacks silently
   * return zeros, so trainOnBatch's loss reads back as 0. This detects both:
   *   - context already dead → latch glContextLost so start() fails fast;
   *   - else attach a one-time webglcontextlost listener that latches the flag if
   *     it dies mid-run (the training loop checks the flag each batch).
   * The zero-loss guard in the loop is the belt-and-braces backstop for the case
   * where zeros arrive without a fired event.
   *
   * No-op unless the active backend is webgl — the cpu fallback and non-webgl
   * envs (Node eval, some test harnesses) have no GL context.
   */
  private installGLWatchdog(): void {
    const tf = this.tf!;
    if (tf.getBackend() !== "webgl") return;
    // tfjs 4.x: the webgl backend exposes its GL context via
    // getGPGPUContext().gl. tf.backend() is typed as the abstract KernelBackend
    // in the umbrella package, so reach the concrete accessor through a narrow
    // cast (confirmed against @tensorflow/tfjs 4.22).
    const backend = tf.backend() as unknown as {
      getGPGPUContext?: () => { gl: WebGLRenderingContext };
    };
    const gl = backend.getGPGPUContext?.().gl;
    if (!gl) return; // backend shape unexpected — nothing to watch
    if (gl.isContextLost()) {
      this.glContextLost = true;
      return;
    }
    if (this.glWatchdogInstalled) return;
    // The context lives on the backend's canvas (an OffscreenCanvas in the
    // worker), which persists across runs — attach once. We deliberately do NOT
    // preventDefault(): that would ask WebKit to try restoring the context,
    // whereas we want to latch and bail to reason "context".
    const canvas = gl.canvas as unknown as EventTarget | null;
    canvas?.addEventListener("webglcontextlost", () => {
      this.glContextLost = true;
    });
    this.glWatchdogInstalled = true;
  }

  /** Bail the current run to the "context" error state — the WebGL context was
      lost (dead on arrival, event-signalled, or inferred from a run of silent-
      zero losses). Mirrors the load-failure error path but with reason
      "context". Deliberately does NOT dispose the models: their tensors live on
      a dead GL context, and the next start()/reset() disposes anyway. */
  private failContextLost(onUpdate?: () => void): void {
    this.running = false;
    this.status = "error";
    this.errorReason = "context";
    onUpdate?.();
  }

  /**
   * Load tfjs (first call only), build a fresh model and run batches until
   * pause/reset or convergence. onUpdate fires after every batch.
   *
   * `assetBase` locates the embedding assets (default `/vla`); it is passed
   * per-call rather than held on the instance because trainer.worker.ts builds
   * its core at module scope, before the host's base arrives with "start".
   */
  async start(onUpdate?: () => void, assetBase?: string): Promise<void> {
    if (this.running) return;
    const myRun = ++this.runId;
    this.running = true;
    this.paused = false;
    this.status = "loading";
    this.errorReason = null;
    onUpdate?.();

    // the pretrained GloVe assets (~1MB) load in parallel with tfjs; both
    // are cached after the first start, so a reset+restart resolves instantly
    let embed: Float32Array;
    try {
      const embedP = loadEmbeddings({ assetBase });
      if (!this.tf) {
        // Import the umbrella "@tensorflow/tfjs" package. A prior attempt to
        // import core+layers+webgl-backend separately (to shed the unused
        // converter/data/cpu-backend weight) measured ~0KB real savings in a
        // production build (core's op/gradient library dominates regardless)
        // AND broke at runtime: tfjs-layers imports its own copy of
        // tfjs-core internally, and the bundler didn't dedupe it against the
        // one imported here, so tensors crossing between "our" core and
        // layers' internal core lacked expected prototype methods (surfaced
        // as "rMat.flatten is not a function" deep inside GRU). The umbrella
        // package guarantees a single shared core instance — not worth
        // reintroducing that class of bug for zero measured benefit.
        const tf = await import("@tensorflow/tfjs");
        await tf.ready();
        maybeDisableWebGLFence(tf);
        this.tf = tf;
      }
      embed = await embedP;
    } catch (err) {
      // The embedding fetch failed, or the fetched assets didn't match this
      // build (see assertAssetShape). Surface it as a real error status the
      // host can render — a retry is worth offering, because loadEmbeddings
      // un-caches the rejection and start() refetches.
      console.error("VLA trainer failed to load", err);
      this.running = false;
      this.status = "error";
      this.errorReason = "assets";
      onUpdate?.();
      return;
    }
    if (!this.running || this.runId !== myRun) return; // reset while loading

    // With the (webgl) backend ready, wire up context-loss detection at the
    // source and fail fast if the context is already dead. On iPadOS a worker's
    // GL context can be evicted the instant it's created (process-wide live-
    // context cap + bfcache), and WebKit returns zeros rather than throwing —
    // this is the on-arrival guard; the zero-loss guard in the loop is the mid-
    // run backstop.
    this.installGLWatchdog();
    if (this.glContextLost) {
      this.failContextLost(onUpdate);
      return;
    }

    this.disposeModels();
    this.models = buildVLAModel(this.tf, embed);
    this.loss = NaN;
    this.smoothLoss = NaN;
    this.initialLoss = NaN;
    this.lossHistory = [];
    this.batches = 0;
    this.convergeStreak = 0;
    this.nonPhysicalStreak = 0;
    this.probes = [];

    try {
      // Language warm-up (text-only, cheap): train the color head to
      // convergence BEFORE the coupled loop, so the attention query starts
      // against a clean language slot. Runs while the UI still reads "Loading".
      await this.languageWarmup(myRun);
      if (!this.running || this.runId !== myRun) return;

      // WebGL compiles each distinct kernel shader (conv2d, pooling, embedding
      // gather, the losses, Adam's update ops) the first time it's used — a
      // one-time cost that would otherwise stall the FIRST visible batch right
      // after the status flips to "Training". Pay it here instead, while the UI
      // still reads "Loading" (a state the user already expects to wait
      // through), so training visibly moves at full speed from the first
      // rendered batch.
      const warmupLoss = await this.trainStep();
      if (!this.running || this.runId !== myRun) return;
      // A context lost during the shader warm-up would seed the whole run with
      // zeros — bail here rather than flicker into "training" first.
      if (this.glContextLost) {
        this.failContextLost(onUpdate);
        return;
      }
      this.loss = warmupLoss;
      this.smoothLoss = warmupLoss;
      this.initialLoss = warmupLoss;
      this.lossHistory.push(warmupLoss);
      this.batches = 1;

      this.status = "training";
      onUpdate?.();

      while (this.running && this.runId === myRun) {
        if (this.paused) {
          await new Promise((r) => setTimeout(r, 120));
          continue;
        }
        const t0 = performance.now();
        const loss = await this.trainStep();
        if (!this.running || this.runId !== myRun) break;

        // A webglcontextlost event during this batch (installGLWatchdog) means
        // every readback from here on is silently zeroed — bail before this
        // garbage batch can touch the convergence logic.
        if (this.glContextLost) {
          this.failContextLost(onUpdate);
          return;
        }

        // Zero-loss guard: a real Huber action loss is finite and > LOSS_FLOOR;
        // a 0 / non-finite loss is the silent-zeros signature of a dead GL
        // context (WebKit doesn't throw). Count consecutive non-physical batches
        // and, after NONPHYSICAL_LIMIT of them, stop to reason "context" instead
        // of looping to MAX_BATCHES on garbage. Distinct from the catch block's
        // cpu fallback, which handles THROWN gradient errors.
        const physical = Number.isFinite(loss) && loss > LOSS_FLOOR;
        if (physical) {
          this.nonPhysicalStreak = 0;
        } else if (++this.nonPhysicalStreak >= NONPHYSICAL_LIMIT) {
          this.glContextLost = true; // latch for any later start()
          this.failContextLost(onUpdate);
          return;
        }

        this.loss = loss;
        if (Number.isNaN(this.initialLoss)) this.initialLoss = loss;
        // keep the FULL curve from batch 0 → now (capped only by MAX_BATCHES),
        // so the plot shows the whole training development, not a trailing slice
        this.lossHistory.push(loss);
        this.batches++;
        // low-lag convergence signal: mean of the last CONVERGE_WINDOW raw
        // losses (read straight off the tail of lossHistory — no extra buffer)
        const window = this.lossHistory.slice(-CONVERGE_WINDOW);
        this.smoothLoss = window.reduce((a, b) => a + b, 0) / window.length;

        // held-out per-bucket telemetry (sweep harness only; probeEveryN
        // defaults to 0 so the demo path skips this entirely)
        if (this.probeEveryN > 0 && this.batches % this.probeEveryN === 0)
          await this.runProbe();

        // converged? training's job is done — keep the model, stop the loop.
        // Gate on `physical`: a non-physical (zeroed) loss must never advance
        // the streak, so a dead context can never satisfy convergence — even if
        // its zeros drag smoothLoss under CONVERGE_LOSS.
        if (physical && this.batches >= MIN_BATCHES && this.smoothLoss < CONVERGE_LOSS) {
          this.convergeStreak++;
        } else {
          this.convergeStreak = 0;
        }
        const maxBatches = this.maxBatchesOverride ?? MAX_BATCHES;
        if (this.convergeStreak >= CONVERGE_STREAK || this.batches >= maxBatches) {
          this.snapshotPolicy(); // freeze the final weights for "try it" mode
          this.status = "converged";
          this.running = false;
          onUpdate?.();
          return;
        }

        onUpdate?.();
        const gap = Math.max(8, BATCH_GAP_MS - (performance.now() - t0));
        await new Promise((r) => setTimeout(r, gap));
      }
    } catch (err) {
      // A batch threw mid-training (e.g. the WebGL backend lost its GPU context
      // and an op couldn't recover). Uncaught, this rejection has nobody
      // awaiting it — trainer.worker.ts calls start() fire-and-forget — so
      // status would sit stuck on "loading" forever with no visible signal.
      // Fall back to the cpu backend once (this small CNN trains fine without a
      // GPU, just slower) and retry from a clean model; give up to idle only if
      // cpu ALSO fails.
      if (!this.running || this.runId !== myRun) return;
      console.error("VLA trainer step failed", err);
      if (this.tf.getBackend() !== "cpu") {
        console.warn("VLA trainer falling back to the cpu backend");
        this.running = false;
        // A lost WebGL context is one reason a step throws here — but the cpu
        // backend doesn't touch GL, so it's a valid recovery. Clear the latch
        // so the cpu restart's on-arrival guard doesn't immediately fail fast to
        // "context" (installGLWatchdog is a no-op on cpu, so it won't re-latch).
        this.glContextLost = false;
        await this.tf.setBackend("cpu");
        return this.start(onUpdate, assetBase);
      }
      this.running = false;
      this.status = "error";
      this.errorReason = "train";
      this.disposeModels();
      onUpdate?.();
    }
  }

  /** Halt gradient steps without touching the model (Resume continues). */
  pause() {
    if (this.status !== "training") return;
    this.paused = true;
    this.status = "paused";
  }

  resume() {
    if (this.status !== "paused") return;
    this.paused = false;
    this.status = "training";
  }

  /** Stop training and discard the learned weights (fresh model next start). */
  reset() {
    this.running = false;
    this.paused = false;
    this.runId++;
    this.status = "idle";
    this.errorReason = null;
    this.disposeModels();
    this.loss = NaN;
    this.smoothLoss = NaN;
    this.initialLoss = NaN;
    this.lossHistory = [];
    this.batches = 0;
    this.convergeStreak = 0;
    this.nonPhysicalStreak = 0;
    // glContextLost is intentionally NOT cleared: a lost context stays lost, so
    // a reset+restart should keep failing fast to reason "context".
    this.probes = [];
  }

  private disposeModels() {
    // the viz/lang sub-models share layers with the main model — disposing the
    // main graph frees the shared weights; dispose() on the others only
    // drops their container objects. The lang twin owns a SEPARATE optimizer
    // (its Adam moment variables aren't part of any shared layer), so free
    // those explicitly or they'd leak on each reset+restart.
    this.models?.lang.optimizer?.dispose();
    this.models?.model.dispose();
    this.models = null;
    // the frozen snapshot is built via buildVLAModel too, so it carries an
    // (unused) lang optimizer of its own — free it alongside its graph.
    this.frozenModels?.lang.optimizer?.dispose();
    this.frozenModels?.model.dispose();
    this.frozenModels = null;
  }

  /** Preprocess an imgSize (48×48) RGBA thumb into the model's inverted input tensor. */
  private visionTensor(img: ImageData): tfType.Tensor4D {
    const tf = this.tf!;
    return tf.tidy(() =>
      tf.sub(1, tf.browser.fromPixels(img, 3).toFloat().div(255)).expandDims(0)
    ) as tfType.Tensor4D;
  }

  /**
   * Policy inference on the LIVE (still-training) model: render the given
   * state (pose + block layout + carried block) to the same model's-eye view
   * the training samples use, run the viz twin, return the predicted
   * ABSOLUTE target joint angles plus the spatial-
   * attention readout (not a delta — the caller subtracts its own known
   * current pose to get the step direction; see Hero.tsx's drawArm). Also
   * serves the Vision Encoder panel's live "where the model looks" heatmap
   * during training.
   */
  predictTarget(
    a1: number,
    a2: number,
    tokens: number[],
    layout: Layout,
    carry: number | null = null
  ): PredictResult | null {
    if (!this.ready) return null;
    return this.inferTarget(this.models!, a1, a2, tokens, layout, carry);
  }

  /**
   * Freeze the current policy weights into the separate inference model, so a
   * rollout attempt can run a FIXED policy for its whole cycle (matching how a
   * real rollout uses frozen weights) while background training keeps updating
   * the main model. Called at each demo-cycle boundary and on convergence.
   * No-op until the first batch has built the main model.
   */
  snapshotPolicy() {
    if (!this.tf || !this.models) return;
    // embeddingMatrix() is non-null whenever models exist (start() awaits it)
    if (!this.frozenModels)
      this.frozenModels = buildVLAModel(this.tf, embeddingMatrix()!);
    // getWeights() returns the live variables' current values; setWeights
    // copies them into the frozen model's own variables, so the snapshot holds
    // steady as the main model trains on. Same architecture → identical weight
    // ordering. The returned tensors are the main model's — do NOT dispose.
    this.frozenModels.model.setWeights(this.models.model.getWeights());
  }

  /**
   * Like predictTarget, but runs the FROZEN snapshot from the last
   * snapshotPolicy() call, so a rollout attempt sees one fixed policy for its
   * whole cycle. Falls back to the live model if no snapshot exists yet.
   */
  predictFrozenTarget(
    a1: number,
    a2: number,
    tokens: number[],
    layout: Layout,
    carry: number | null = null
  ): PredictResult | null {
    if (!this.ready) return null;
    return this.inferTarget(
      this.frozenModels ?? this.models!,
      a1,
      a2,
      tokens,
      layout,
      carry
    );
  }

  /** Render the state, run the given models' viz twin, return the predicted
      target angles + attention readout (one forward pass for both). */
  private inferTarget(
    models: VLAModels,
    a1: number,
    a2: number,
    tokens: number[],
    layout: Layout,
    carry: number | null = null
  ): PredictResult {
    const tf = this.tf!;
    const img = this.renderPose(a1, a2, layout, carry);
    const out = tf.tidy(() => {
      const v = this.visionTensor(img);
      const l = tf.tensor2d([tokens], [1, MAX_SEQ_LEN], "int32");
      // proprioceptive flag: the rollout KNOWS whether its gripper holds a
      // block (the snap-grasp set it) — same signal training synthesized
      const c = tf.tensor2d([carry !== null ? 1 : 0], [1, 1]);
      const [action, pick, grip] = models.viz.predict([
        v,
        l,
        c,
      ]) as tfType.Tensor[];
      return {
        action: action.dataSync(),
        pick: pick.dataSync() as Float32Array,
        grip: grip.dataSync() as Float32Array,
      };
    });
    // the attention map the UI shows — it tracks the commanded block whether
    // empty-handed (its floor spot) or carrying (the effector).
    const attnSel = out.pick;
    // the UI's gaze point: the map's expectation in plain [0,1] image coords,
    // computed here CPU-side (the in-graph attn_grid kernel feeds the action
    // head CENTERED+GAINED coords — see config.ts attnCoordGain — so it isn't
    // directly displayable). Must use the RAW softmax map, before the peak
    // normalization below.
    const G = ATTN_GRID;
    let ux = 0;
    let vy = 0;
    let peak = 0;
    for (let i = 0; i < attnSel.length; i++) {
      const w = attnSel[i];
      ux += w * ((i % G) + 0.5);
      vy += w * (Math.floor(i / G) + 0.5);
      if (w > peak) peak = w;
    }
    // normalize the map so the peak cell is 1 — the UI uses it as alpha
    const inv = peak > 0 ? 1 / peak : 0;
    // recover the target joint angles from the 4 circular action coords
    // (cosθ1,sinθ1,cosθ2,sinθ2) with atan2 + the θ1 unwrap — see model.ts. The
    // rollout consumes plain angles, so PredictResult.target stays [θ1, θ2].
    const [t1, t2] = anglesFromCircular(
      out.action[0],
      out.action[1],
      out.action[2],
      out.action[3]
    );
    return {
      target: [t1, t2],
      attn: Array.from(attnSel, (a) => a * inv),
      xy: [ux / G, vy / G],
      grip: out.grip[0],
    };
  }

  /**
   * Decode the acted-on color from a token sequence via the auxiliary color
   * head (which reads only the language branch — vision input is zeros).
   */
  decodeCommand(tokens: number[]): DecodedCommand | null {
    if (!this.ready) return null;
    const tf = this.tf!;
    const out = tf.tidy(() => {
      const v = tf.zeros([1, IMG_SIZE, IMG_SIZE, 3]);
      const l = tf.tensor2d([tokens], [1, MAX_SEQ_LEN], "int32");
      const c = tf.zeros([1, 1]); // decode = empty-handed reading of the text
      const [, color] = this.models!.model.predict([
        v,
        l,
        c,
      ]) as tfType.Tensor[];
      return { color: color.dataSync() };
    });
    const argmax = (a: Float32Array | Int32Array | Uint8Array) => {
      let best = 0;
      for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
      return best;
    };
    const c = argmax(out.color);
    return { color: c, colorProb: out.color[c] };
  }

  /**
   * The language encoder's per-token ATTENTION weights — the live per-chip
   * bars. The pooling scorer is a single LINEAR dense layer over the frozen
   * embeddings, so the exact masked-softmax weights the model computes
   * internally can be recomputed CPU-side here: score each token as
   * embedding[token] · scoreKernel + scoreBias, drop padding, softmax over
   * the rest. No forward pass — just the scorer's two small weight tensors
   * plus a dot product per token against the frozen table. Weights are
   * normalized so the most-attended token fills its bar.
   */
  attentionWeights(tokens: number[]): number[] | null {
    if (!this.ready) return null;
    // the embedding table is FROZEN pretrained GloVe — read it from the
    // CPU-side copy loadEmbeddings() kept (syncing the 20k x 50 table off
    // the GPU every refresh would dwarf the readout it powers); only the
    // small trainable conv-scorer kernel is pulled from the model. Bars
    // show the TARGET slot's attention ("attn_score" — the scorer whose
    // pooled vector the color head decodes).
    const embedTable = embeddingMatrix();
    if (!embedTable) return null;
    const w = this.models!.model.getLayer("attn_score").getWeights();
    const kernel = w[0].arraySync() as number[][][]; // [K][EMBED_DIM][1]
    const bias = (w[1].arraySync() as number[])[0];
    const K = kernel.length;
    const off = Math.floor(K / 2); // conv1d "same": window i-off .. i+off
    // raw scores; padding is excluded from the softmax entirely. The conv
    // window's out-of-range slots are implicit zeros in-graph, and PAD
    // tokens embed to the all-zero row — `continue` reproduces both.
    const scores = tokens.map((tok, i) => {
      if (tok === PAD) return -Infinity;
      let s = bias;
      for (let k = 0; k < K; k++) {
        const j = i + k - off;
        if (j < 0 || j >= tokens.length) continue;
        const t = tokens[j];
        if (t === PAD) continue;
        for (let d = 0; d < EMBED_DIM; d++)
          s += embedTable[t * EMBED_DIM + d] * kernel[k][d][0];
      }
      return s;
    });
    const finite = scores.filter((s) => Number.isFinite(s));
    const maxS = finite.length ? Math.max(...finite) : 0;
    let sum = 0;
    const exps = scores.map((s) => {
      if (!Number.isFinite(s)) return 0;
      const e = Math.exp(s - maxS);
      sum += e;
      return e;
    });
    const weights = exps.map((e) => (sum > 0 ? e / sum : 0));
    const maxW = Math.max(...weights, 1e-6);
    return weights.map((wt) => wt / maxW);
  }

  /** Loss normalized against the first batch, clamped to [0,1]. */
  lossNorm(): number {
    if (Number.isNaN(this.loss) || Number.isNaN(this.initialLoss)) return 1;
    if (this.initialLoss <= 0) return 0;
    return Math.max(0, Math.min(1, this.loss / this.initialLoss));
  }
}
