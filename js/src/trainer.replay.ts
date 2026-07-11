// The REPLAY fallback. When live training can't run (iOS/iPadOS WebKit's dead
// WebGL context — see the trainer.ts watchdog), VLATrainer swaps this in behind
// the identical surface, so Hero renders it without knowing the difference.
//
// What's REAL here vs SCRIPTED:
//   - Demonstrations: real — Hero samples them from the analytical-IK expert
//     (pure geometry, no tfjs), exactly as in a live run.
//   - Rollouts: real — genuine forward passes of a PRETRAINED policy on the
//     CPU backend (no GL context, cheap: inference is a few passes/sec, not the
//     hundreds of gradient batches training needs). The bad→good progression is
//     real too: a LADDER of checkpoints captured along one real training run
//     (assets/replay/, see js/capture) is selected by training progress, so
//     early cycles roll out a genuinely clumsy policy and later ones a good one.
//   - The loss curve: the only scripted element — drawn through the checkpoints'
//     real (samples, loss) anchors with correlated noise (anchored-procedural),
//     over a per-visit-randomized ~25s, so no two runs look identical.
//
// Because WASM can't train this model (missing Conv2DBackpropFilter) and CPU
// training is ~5min, a live-training experience is impossible on these devices;
// this replays a real one instead. Runs INLINE on the main thread (inference is
// cheap and the rollout already tolerates predict latency).

import { CONFIG } from "./config";
import { loadEmbeddings } from "./embeddings";
import type { Layout } from "./examples";
import { buildVLAModel, type TF, type VLAModels } from "./model";
import {
  decodeColor,
  inferTarget,
  makeRenderCanvases,
  tokenAttention,
  type DecodedCommand,
  type PredictResult,
  type RenderCanvases,
} from "./infer";
import {
  applyPolicyWeights,
  type PolicyCheckpoint,
  type PolicyWeightSpec,
} from "./policy-weights";
import type { TrainerError, TrainerStatus } from "./trainer.core";

const CONVERGE_WINDOW = CONFIG.trainer.converge.window;
const R = CONFIG.replay;

interface ReplayManifest {
  batchSize: number;
  cadencePerSec: number;
  floorLoss: number;
  weightSpecs: PolicyWeightSpec[];
  checkpoints: { samples: number; loss: number; file: string }[];
}

/** One rung of the captured ladder: a policy checkpoint + its anchor (the
    displayed sample count it sits at, and the smoothed loss there). */
interface Anchor {
  samples: number;
  loss: number;
  ckpt: PolicyCheckpoint;
}

/** Roughly-Gaussian noise (sum of two uniforms). */
function gauss(std: number): number {
  return (Math.random() + Math.random() - 1) * std;
}

function randInt(lo: number, hi: number): number {
  return lo + Math.floor(Math.random() * (hi - lo + 1));
}

/** log(loss) at a displayed sample count, piecewise-linear in log-space through
    the anchors (so segments are exponential decays — the natural loss shape). */
function interpLogLoss(anchors: Anchor[], s: number): number {
  if (s <= anchors[0].samples) return Math.log(anchors[0].loss);
  const last = anchors[anchors.length - 1];
  if (s >= last.samples) return Math.log(last.loss);
  for (let i = 1; i < anchors.length; i++) {
    if (s <= anchors[i].samples) {
      const a = anchors[i - 1];
      const b = anchors[i];
      const t = (s - a.samples) / (b.samples - a.samples);
      return Math.log(a.loss) + t * (Math.log(b.loss) - Math.log(a.loss));
    }
  }
  return Math.log(last.loss);
}

export class ReplayTrainer {
  status: TrainerStatus = "idle";
  errorReason: TrainerError | null = null;
  loss = NaN;
  smoothLoss = NaN;
  initialLoss = NaN;
  lossHistory: number[] = [];
  batches = 0;

  private tf: TF | null = null;
  private models: VLAModels | null = null;
  private cv: RenderCanvases | null = null;
  /** The raw checkpoints + their real (samples, loss), loaded once and cached. */
  private raw: { samples: number; loss: number; ckpt: PolicyCheckpoint }[] = [];
  private batchSize = 32;
  private cadenceMs = 100;
  private floor = 0.008;

  private onUpdate?: () => void;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private running = false;
  private paused = false;
  private runId = 0;

  // ── per-visit randomization (fresh each start → no two runs identical) ──
  private anchors: Anchor[] = [];
  private totalBatches = 250;
  private thresholdOffset = 0;
  private walk = 0;
  private selectedIdx = -1;

  get samples(): number {
    return this.batches * this.batchSize;
  }

  get ready(): boolean {
    return (
      this.models !== null &&
      (this.status === "training" ||
        this.status === "paused" ||
        this.status === "converged") &&
      this.batches > 0
    );
  }

  /** Loss normalized against the first batch, clamped to [0,1] (matches the
      live trainer's contract — Hero reads it for the arm's exhaustion sway). */
  lossNorm(): number {
    if (Number.isNaN(this.loss) || Number.isNaN(this.initialLoss)) return 1;
    if (this.initialLoss <= 0) return 0;
    return Math.max(0, Math.min(1, this.loss / this.initialLoss));
  }

  /** Begin a replayed run. Loads (once, cached) tfjs on the CPU backend, the
      embeddings, and the checkpoint ladder; then plays the scripted loss +
      real rollouts. onUpdate fires per scripted batch + on status changes —
      same cadence Hero expects from the live trainer. */
  async start(onUpdate?: () => void, assetBase?: string): Promise<void> {
    if (this.running) return;
    const myRun = ++this.runId;
    this.running = true;
    this.paused = false;
    this.onUpdate = onUpdate;
    this.status = "loading";
    this.errorReason = null;
    this.loss = NaN;
    this.smoothLoss = NaN;
    this.initialLoss = NaN;
    this.lossHistory = [];
    this.batches = 0;
    this.selectedIdx = -1;
    onUpdate?.();

    try {
      await this.load(assetBase);
    } catch (err) {
      if (!this.running || this.runId !== myRun) return;
      // The replay is the safety net; if its OWN assets 404 / fail, surface a
      // retryable "assets" error and let the host's outer watchdog show a
      // static card (v0.4.1 behavior — never a new hang).
      console.error("VLA replay failed to load", err);
      this.running = false;
      this.status = "error";
      this.errorReason = "assets";
      onUpdate?.();
      return;
    }
    if (!this.running || this.runId !== myRun) return;

    // per-visit params: a randomized run length (→ ~R.minBatches..maxBatches /
    // cadence seconds), the converged rung repositioned to that end, and a
    // threshold jitter that shifts the bad→good timing between visits.
    this.totalBatches = Math.max(
      randInt(R.minBatches, R.maxBatches),
      // always leave room past the last INTERMEDIATE anchor for a visible
      // final-convergence segment, regardless of how long the capture ran
      Math.ceil(this.raw[this.raw.length - 2].samples / this.batchSize) + 20
    );
    const finalSamples = this.totalBatches * this.batchSize;
    this.anchors = this.raw.map((a, i) => ({
      // intermediate rungs keep their REAL sample position (so the early
      // timing matches a live run — 5s in ≈ the clumsy rung); the converged
      // rung is moved to the replay's end so it converges in a realistic ~25s
      // no matter how many batches this capture happened to take.
      samples: i === this.raw.length - 1 ? finalSamples : a.samples,
      loss: a.loss,
      ckpt: a.ckpt,
    }));
    this.thresholdOffset = (Math.random() * 2 - 1) * R.thresholdJitter;
    this.walk = 0;
    this.applyForSamples(0); // seed the initial (clumsy) rung

    this.status = "training";
    this.tick(myRun);
  }

  /** Idempotent load: tfjs (cpu), embeddings, manifest + checkpoint bins. */
  private async load(assetBase?: string): Promise<void> {
    if (!this.tf) {
      const tf = await import("@tensorflow/tfjs");
      // CPU only — the whole point on iOS is to never touch a WebGL context.
      await tf.setBackend("cpu");
      await tf.ready();
      this.tf = tf;
    }
    if (this.models && this.raw.length) return; // already loaded — reuse
    const tf = this.tf;
    const base = assetBase ?? "/vla";
    const [embed, manifest] = await Promise.all([
      loadEmbeddings({ assetBase }),
      fetch(`${base}/replay/manifest.json`).then((r) => {
        if (!r.ok) throw new Error("replay manifest failed to load");
        return r.json() as Promise<ReplayManifest>;
      }),
    ]);
    if (!manifest.checkpoints?.length || manifest.checkpoints.length < 2)
      throw new Error("replay manifest has too few checkpoints");
    const bins = await Promise.all(
      manifest.checkpoints.map(async (c) => {
        const res = await fetch(`${base}/replay/${c.file}`);
        if (!res.ok) throw new Error(`replay checkpoint ${c.file} failed to load`);
        return new Float32Array(await res.arrayBuffer());
      })
    );
    this.batchSize = manifest.batchSize;
    this.cadenceMs = 1000 / (manifest.cadencePerSec || 10);
    this.floor = Math.min(...manifest.checkpoints.map((c) => c.loss)) * 0.5;
    this.raw = manifest.checkpoints.map((c, i) => ({
      samples: c.samples,
      loss: c.loss,
      ckpt: { specs: manifest.weightSpecs, data: bins[i] },
    }));
    this.models = buildVLAModel(tf, embed); // seeds the frozen embedding + grid
    this.cv = makeRenderCanvases();
  }

  private tick(myRun: number): void {
    if (!this.running || this.runId !== myRun) return;
    if (this.paused) {
      this.timer = setTimeout(() => this.tick(myRun), this.cadenceMs);
      return;
    }
    this.batches++;
    // anchored-procedural loss: the anchor curve in log-space + an AR(1) walk
    // (correlated batch-to-batch noise — reads as a real loss, not white jitter)
    const logMean = interpLogLoss(this.anchors, this.samples);
    this.walk = R.noiseRho * this.walk + gauss(R.noiseSigma);
    const loss = Math.max(this.floor, Math.exp(logMean + this.walk));
    this.loss = loss;
    if (Number.isNaN(this.initialLoss)) this.initialLoss = loss;
    this.lossHistory.push(loss);
    const w = this.lossHistory.slice(-CONVERGE_WINDOW);
    this.smoothLoss = w.reduce((a, b) => a + b, 0) / w.length;

    if (this.batches >= this.totalBatches) {
      this.applyForSamples(Infinity); // pin the converged rung for "try it"
      this.status = "converged";
      this.running = false;
      this.onUpdate?.();
      return;
    }
    this.onUpdate?.();
    this.timer = setTimeout(() => this.tick(myRun), this.cadenceMs);
  }

  /** Freeze the policy shown for THIS demo cycle (Hero calls it at each cycle
      boundary): pick the rung for the current progress and load it. Once
      converged, always the final (best) rung — what "try it" runs against. */
  snapshotPolicy(): void {
    this.applyForSamples(this.status === "converged" ? Infinity : this.samples);
  }

  /** Select + load the ladder rung for a displayed sample count. Idempotent
      (no-op if the rung is already loaded). The per-visit thresholdOffset
      shifts WHICH rung a given progress maps to, so the bad→good timing varies
      between visits (the 5s→clumsy start is robust to it, though). */
  private applyForSamples(s: number): void {
    if (!this.tf || !this.models || this.anchors.length === 0) return;
    let idx = 0;
    if (s === Infinity) {
      idx = this.anchors.length - 1;
    } else {
      const target = s * (1 + this.thresholdOffset);
      for (let i = 0; i < this.anchors.length; i++)
        if (this.anchors[i].samples <= target) idx = i;
    }
    if (idx === this.selectedIdx) return;
    this.selectedIdx = idx;
    applyPolicyWeights(this.tf, this.models.model, this.anchors[idx].ckpt);
  }

  // ── inference: the async surface Hero calls; real sync CPU forward inside ──
  predictFrozenTarget(
    a1: number,
    a2: number,
    tokens: number[],
    layout: Layout,
    carry: number | null = null
  ): Promise<PredictResult | null> {
    if (!this.ready || !this.tf || !this.models || !this.cv)
      return Promise.resolve(null);
    return Promise.resolve(
      inferTarget(this.tf, this.models, this.cv, a1, a2, tokens, layout, carry)
    );
  }

  /** In a live run this is the still-training model; in replay it's the same
      per-cycle rung as the frozen snapshot (the gaze panel refines in cycle
      steps rather than continuously — the one visibly-quantized difference). */
  predictLive(
    a1: number,
    a2: number,
    tokens: number[],
    layout: Layout,
    carry: number | null = null
  ): Promise<PredictResult | null> {
    return this.predictFrozenTarget(a1, a2, tokens, layout, carry);
  }

  decodeCommand(tokens: number[]): Promise<DecodedCommand | null> {
    if (!this.ready || !this.tf || !this.models) return Promise.resolve(null);
    return Promise.resolve(decodeColor(this.tf, this.models, tokens));
  }

  attentionWeights(tokens: number[]): Promise<number[] | null> {
    if (!this.ready || !this.models) return Promise.resolve(null);
    return Promise.resolve(tokenAttention(this.models, tokens));
  }

  pause(): void {
    if (this.status !== "training") return;
    this.paused = true;
    this.status = "paused";
  }

  resume(): void {
    if (this.status !== "paused") return;
    this.paused = false;
    this.status = "training";
  }

  reset(): void {
    this.running = false;
    this.paused = false;
    this.runId++;
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    this.status = "idle";
    this.errorReason = null;
    this.loss = NaN;
    this.smoothLoss = NaN;
    this.initialLoss = NaN;
    this.lossHistory = [];
    this.batches = 0;
    this.selectedIdx = -1;
    // keep tf + models + checkpoints cached so a restart replays instantly
  }

  destroy(): void {
    this.reset();
    this.models?.model.dispose();
    this.models = null;
    this.raw = [];
    this.anchors = [];
  }
}
