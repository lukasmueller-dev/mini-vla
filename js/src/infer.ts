// Shared policy INFERENCE + silhouette rendering — the render → forward →
// readout path, factored out of trainer.core.ts so BOTH the live trainer and
// the replay fallback (trainer.replay.ts) run the exact same code. Pure
// functions over (tf, models, canvases): no training, no instance state beyond
// the frozen embedding table these read from src/embeddings.
//
// The one bit of module state these depend on is the FROZEN GloVe table
// (embeddingMatrix()), which tokenAttention recomputes the language scorer
// against CPU-side — same rule as before: never sync the 20k×50 table off the
// GPU.

import type * as tfType from "@tensorflow/tfjs";
import { CONFIG } from "./config";
import { paintSilhouette } from "./scene";
import { MAX_SEQ_LEN, PAD, type Layout } from "./examples";
import { ATTN_GRID, IMG_SIZE, type TF, type VLAModels } from "./model";
import { EMBED_DIM } from "./vocab.gen";
import { embeddingMatrix } from "./embeddings";

// Silhouettes are drawn at this px then averaged down to IMG_SIZE (see config).
const RENDER_SIZE = CONFIG.trainer.renderSize;

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

/** The scene + thumbnail canvases the render pipeline draws into. Held by the
    caller (one per trainer/replay instance) and passed to renderPose. */
export interface RenderCanvases {
  sceneCanvas: CanvasImageSource;
  sceneCtx: Ctx2D;
  thumbCtx: Ctx2D;
}

export function makeRenderCanvases(): RenderCanvases {
  const scene = make2d(RENDER_SIZE);
  return {
    sceneCanvas: scene.canvas,
    sceneCtx: scene.ctx,
    thumbCtx: make2d(IMG_SIZE).ctx,
  };
}

/** Recover (θ1, θ2) from the model's 4 circular action outputs (see model.ts).
    atan2 reads only the DIRECTION, so the unconstrained radius is harmless. θ1
    is un-wrapped into solveIK's [-π/2, 3π/2) band (geometry.ts) so the rollout,
    which steps proportionally FROM the current pose, moves the short way round
    (a raw atan2 in (-π, π] could report a near-π target as its negative twin). */
export function anglesFromCircular(
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

/** Render a state through the silhouette pipeline; returns IMG_SIZE RGBA.
    `carry` draws that block at the effector instead of its rest spot —
    the carry-phase state cue. */
export function renderPose(
  cv: RenderCanvases,
  a1: number,
  a2: number,
  layout: Layout,
  carry: number | null = null
): ImageData {
  paintSilhouette(cv.sceneCtx, RENDER_SIZE, a1, a2, layout, carry);
  const tctx = cv.thumbCtx;
  tctx.imageSmoothingEnabled = true;
  tctx.clearRect(0, 0, IMG_SIZE, IMG_SIZE);
  tctx.drawImage(
    cv.sceneCanvas,
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

/** Preprocess an IMG_SIZE (48×48) RGBA thumb into the model's inverted input. */
function visionTensor(tf: TF, img: ImageData): tfType.Tensor4D {
  return tf.tidy(() =>
    tf.sub(1, tf.browser.fromPixels(img, 3).toFloat().div(255)).expandDims(0)
  ) as tfType.Tensor4D;
}

/** Render the state, run the models' viz twin, return the predicted target
    angles + attention readout (one forward pass for both). */
export function inferTarget(
  tf: TF,
  models: VLAModels,
  cv: RenderCanvases,
  a1: number,
  a2: number,
  tokens: number[],
  layout: Layout,
  carry: number | null = null
): PredictResult {
  const img = renderPose(cv, a1, a2, layout, carry);
  const out = tf.tidy(() => {
    const v = visionTensor(tf, img);
    const l = tf.tensor2d([tokens], [1, MAX_SEQ_LEN], "int32");
    // proprioceptive flag: the rollout KNOWS whether its gripper holds a
    // block (the snap-grasp set it) — same signal training synthesized
    const c = tf.tensor2d([carry !== null ? 1 : 0], [1, 1]);
    const [action, pick, grip] = models.viz.predict([v, l, c]) as tfType.Tensor[];
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
export function decodeColor(
  tf: TF,
  models: VLAModels,
  tokens: number[]
): DecodedCommand {
  const out = tf.tidy(() => {
    const v = tf.zeros([1, IMG_SIZE, IMG_SIZE, 3]);
    const l = tf.tensor2d([tokens], [1, MAX_SEQ_LEN], "int32");
    const c = tf.zeros([1, 1]); // decode = empty-handed reading of the text
    const [, color] = models.model.predict([v, l, c]) as tfType.Tensor[];
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
 * The language encoder's per-token ATTENTION weights — the live per-chip bars.
 * The pooling scorer is a single LINEAR dense layer over the frozen embeddings,
 * so the exact masked-softmax weights the model computes internally can be
 * recomputed CPU-side here: score each token as embedding[token] · scoreKernel
 * + scoreBias, drop padding, softmax over the rest. No forward pass — just the
 * scorer's two small weight tensors plus a dot product per token against the
 * frozen table. Weights are normalized so the most-attended token fills its bar.
 */
export function tokenAttention(
  models: VLAModels,
  tokens: number[]
): number[] | null {
  // the embedding table is FROZEN pretrained GloVe — read it from the CPU-side
  // copy loadEmbeddings() kept (syncing the 20k × 50 table off the GPU every
  // refresh would dwarf the readout it powers); only the small trainable
  // conv-scorer kernel is pulled from the model. Bars show the TARGET slot's
  // attention ("attn_score" — the scorer whose pooled vector the color head
  // decodes).
  const embedTable = embeddingMatrix();
  if (!embedTable) return null;
  const w = models.model.getLayer("attn_score").getWeights();
  const kernel = w[0].arraySync() as number[][][]; // [K][EMBED_DIM][1]
  const bias = (w[1].arraySync() as number[])[0];
  const K = kernel.length;
  const off = Math.floor(K / 2); // conv1d "same": window i-off .. i+off
  // raw scores; padding is excluded from the softmax entirely. The conv
  // window's out-of-range slots are implicit zeros in-graph, and PAD tokens
  // embed to the all-zero row — `continue` reproduces both.
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
