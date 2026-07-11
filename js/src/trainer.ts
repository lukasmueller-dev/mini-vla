// Main-thread proxy for the VLA trainer. The actual training loop lives in
// src/trainer.core.ts and normally runs inside src/trainer.worker.ts
// (a module Web Worker with OffscreenCanvas + its own WebGL context), so
// gradient steps never contend with the hero's 60fps rAF loop. This class is
// what Hero.tsx talks to; it keeps the pre-worker VLATrainer surface:
//
//  - control (start/pause/resume/reset/snapshotPolicy) is fire-and-forget:
//    the proxy applies the same optimistic state transition the old in-thread
//    class did synchronously, posts the command, and re-syncs from the
//    worker's authoritative {t:"state"} echo.
//  - telemetry (status/errorReason/loss/lossHistory/batches/ready/lossNorm)
//    stays a plain synchronous read, served from a mirror updated per batch.
//    lossHistory is reconstructed by appending each batch's loss — only a
//    few numbers cross the thread boundary per step, never the full curve.
//  - inference (predictFrozenTarget/decodeCommand/attentionWeights) is now
//    ASYNC (Promise), matched to worker replies by request id. The rollout
//    already re-predicts on a throttle and steps toward its LAST target every
//    frame, so the added round-trip latency is invisible (see Hero.tsx).
//
// A `gen` counter increments on every reset and is echoed on each state post,
// so batch updates still in flight when the mirror was cleared are dropped
// instead of repopulating it.
//
// On browsers without module-worker/OffscreenCanvas support the proxy falls
// back to running VLATrainerCore inline on the main thread (the pre-worker
// behavior, jank and all) behind the same async API.

import { CONFIG } from "./config";
import { registerFullVocab, type Layout } from "./examples";
import {
  DEFAULT_RUN_CONFIG,
  setRunConfig,
  type RunConfig,
} from "./run-config";
import {
  VLATrainerCore,
  type DecodedCommand,
  type PredictResult,
  type TrainerError,
  type TrainerStatus,
} from "./trainer.core";
import { ReplayTrainer } from "./trainer.replay";
import type { WorkerRequest, WorkerResponse } from "./trainer.worker";

export type { DecodedCommand, PredictResult, TrainerError, TrainerStatus };

/** Host wiring for the trainer. */
export interface VLATrainerOptions {
  /** Directory the host serves the package's `assets/` from — the embedding
      `.bin` + `vocab.txt` are fetched from here. Defaults to `/vla`, which is
      what the package's own demo/eval/test pages serve. A host that redeploys
      often should version-stamp it (`/vla/0.4.0`) so a stale tab 404s instead
      of loading assets from a different generation than its JS. */
  assetBase?: string;
  /** Opt in to the REPLAY fallback (see trainer.replay.ts). When on, a run that
      never reaches its first training batch within `CONFIG.replay.watchdogMs`
      (the iOS/iPadOS dead-WebGL-context wedge, where tf.ready hangs and the
      worker posts nothing) OR that errors out is transparently replaced by a
      scripted-loss + real-CPU-rollout replay of a pretrained policy — behind
      this SAME surface, so the host UI just keeps rendering. It abstracts away
      WHY the real path failed. Default off (existing behavior). The host still
      keeps its own outer watchdogs as a last-ditch net. */
  replayFallback?: boolean;
  /** How long the real path may sit in Loading (Language Warmup) without a
      first training batch before the replay fallback takes over, in ms.
      Overrides `CONFIG.replay.watchdogMs` (7500) for THIS trainer only; ignored
      unless `replayFallback` is on. Lower it to get stalled devices onto the
      replay sooner — but keep clear headroom above a HEALTHY device's warm-up
      time, or a merely-slow (cold tfjs, thermal-throttled, low-end GPU) real
      run gets swapped to the replay before its own first batch lands. Default
      `CONFIG.replay.watchdogMs`. */
  replayWatchdogMs?: number;
}

const BATCH_SIZE = CONFIG.trainer.batchSize;

/** Omit that distributes over a union (plain Omit collapses it to the
    members' common properties, losing the per-variant fields). */
type DistributiveOmit<T, K extends keyof T> = T extends unknown
  ? Omit<T, K>
  : never;

/** True when the trainer can host itself off the main thread. */
function workerSupported() {
  return (
    typeof Worker !== "undefined" && typeof OffscreenCanvas !== "undefined"
  );
}

export class VLATrainer {
  /** Inline fallback core (only when workerSupported() is false). */
  private core: VLATrainerCore | null = null;
  private worker: Worker | null = null;
  private onUpdate: (() => void) | undefined;
  private assetBase: string | undefined;
  /** Reset generation — state posts from an older gen are stale, drop them. */
  private gen = 0;
  private nextId = 1;
  private pending = new Map<number, (result: never) => void>();
  /** Replay fallback (trainer.replay.ts). Non-null ONCE the real path has been
      given up on — from then every accessor + control delegates here, and the
      real worker/core is gone. */
  private replay: ReplayTrainer | null = null;
  private replayFallback: boolean;
  /** Load-watchdog interval (ms) — the host-tunable override of
      CONFIG.replay.watchdogMs; see VLATrainerOptions.replayWatchdogMs. */
  private replayWatchdogMs: number;
  /** Fires if the real path hasn't produced a first training batch in time. */
  private loadWatchdog: ReturnType<typeof setTimeout> | null = null;
  /** True once the real path reached "training" with ≥1 batch — disarms the
      load watchdog (the run is progressing on its own). */
  private reachedTraining = false;

  // Mirror of the worker-side core's telemetry (worker mode only; in inline
  // mode every accessor delegates to the core directly).
  private m = {
    status: "idle" as TrainerStatus,
    errorReason: null as TrainerError | null,
    loss: NaN,
    smoothLoss: NaN,
    initialLoss: NaN,
    lossHistory: [] as number[],
    batches: 0,
  };

  constructor(options: VLATrainerOptions = {}) {
    this.assetBase = options.assetBase;
    this.replayFallback = options.replayFallback ?? false;
    this.replayWatchdogMs = options.replayWatchdogMs ?? CONFIG.replay.watchdogMs;
  }

  // Every accessor checks the replay fallback FIRST: once it's live the real
  // path is gone and the replay is the single source of truth (its surface
  // matches the core's, so the host reads it identically).
  get status() {
    if (this.replay) return this.replay.status;
    return this.core ? this.core.status : this.m.status;
  }
  /** Why status is "error" (null otherwise) — decides whether the host offers
      a retry or a reload. See TrainerError. */
  get errorReason() {
    if (this.replay) return this.replay.errorReason;
    return this.core ? this.core.errorReason : this.m.errorReason;
  }
  /** Real action loss (Huber) from the latest batch (NaN before the first). */
  get loss() {
    if (this.replay) return this.replay.loss;
    return this.core ? this.core.loss : this.m.loss;
  }
  /** Trailing-window mean action loss (the convergence signal). */
  get smoothLoss() {
    if (this.replay) return this.replay.smoothLoss;
    return this.core ? this.core.smoothLoss : this.m.smoothLoss;
  }
  /** First recorded loss — the normalization anchor for the UI. */
  get initialLoss() {
    if (this.replay) return this.replay.initialLoss;
    return this.core ? this.core.initialLoss : this.m.initialLoss;
  }
  get lossHistory() {
    if (this.replay) return this.replay.lossHistory;
    return this.core ? this.core.lossHistory : this.m.lossHistory;
  }
  get batches() {
    if (this.replay) return this.replay.batches;
    return this.core ? this.core.batches : this.m.batches;
  }
  /** Total expert-labeled examples consumed so far. */
  get samples() {
    if (this.replay) return this.replay.samples;
    return this.core ? this.core.samples : this.m.batches * BATCH_SIZE;
  }
  get ready() {
    if (this.replay) return this.replay.ready;
    if (this.core) return this.core.ready;
    return (
      (this.m.status === "training" ||
        this.m.status === "paused" ||
        this.m.status === "converged") &&
      this.m.batches > 0
    );
  }

  /** Whether the replay fallback is currently standing in for the real path
      (it was given up on this session). The host UI is meant to be oblivious,
      but this stays exposed for diagnostics/logging — the reason is abstracted
      away, the fact isn't. */
  get usingReplay(): boolean {
    return this.replay !== null;
  }

  /** Loss normalized against the first batch, clamped to [0,1]. */
  lossNorm(): number {
    if (this.replay) return this.replay.lossNorm();
    if (this.core) return this.core.lossNorm();
    const { loss, initialLoss } = this.m;
    if (Number.isNaN(loss) || Number.isNaN(initialLoss)) return 1;
    if (initialLoss <= 0) return 0;
    return Math.max(0, Math.min(1, loss / initialLoss));
  }

  private ensureBackend() {
    if (this.core || this.worker) return;
    if (!workerSupported()) {
      this.core = new VLATrainerCore();
      // with a replay standing by, a thrown step should hand off to it, not
      // grind on the cpu backend (see VLATrainerCore.skipCpuFallback)
      this.core.skipCpuFallback = this.replayFallback;
      return;
    }
    this.worker = new Worker(
      new URL("./trainer.worker.ts", import.meta.url),
      { type: "module" }
    );
    // If the worker script fails to load/evaluate (stale chunk, unsupported
    // browser, etc.) or posts something structured-clone can't handle, there
    // is otherwise no signal at all — status would sit at "loading" forever.
    // Surface it as status "error" / reason "worker": unlike an "assets"
    // failure, retrying in-page is futile here, because a fresh Worker would
    // resolve the same dead URL. Only a page reload can recover.
    this.worker.onerror = (ev) => {
      console.error("VLA trainer worker failed:", ev.message || ev);
      this.handleWorkerFailure();
    };
    this.worker.onmessageerror = () => {
      console.error("VLA trainer worker sent an unclonable message");
      this.handleWorkerFailure();
    };
    this.worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      if (this.replay) return; // real path abandoned — ignore any straggler
      const msg = e.data;
      switch (msg.t) {
        case "state": {
          if (msg.gen !== this.gen) return; // pre-reset message in flight
          if (msg.batches > this.m.batches) this.m.lossHistory.push(msg.loss);
          this.m.status = msg.status;
          this.m.errorReason = msg.errorReason;
          this.m.loss = msg.loss;
          this.m.smoothLoss = msg.smoothLoss;
          this.m.initialLoss = msg.initialLoss;
          this.m.batches = msg.batches;
          // disarm the load watchdog on first real batch, or fall over to the
          // replay if this update is a terminal error (when replayFallback)
          this.afterRealUpdate();
          if (!this.replay) this.onUpdate?.();
          break;
        }
        case "vocab":
          // the worker's embedding load registered the 20k-word list with ITS
          // examples.ts; mirror that into this thread's tokenizer so typed
          // near-synonyms resolve to real token ids in "try it" mode
          registerFullVocab(msg.words);
          break;
        case "predict":
        case "predictLive":
          this.take(msg.id)?.(msg.result as never);
          break;
        case "decode":
          this.take(msg.id)?.(msg.result as never);
          break;
        case "attention":
          this.take(msg.id)?.(msg.result as never);
          break;
      }
    };
  }

  private take(id: number) {
    const resolve = this.pending.get(id);
    this.pending.delete(id);
    return resolve;
  }

  /** Worker died (load error or unclonable message) — drop it, resolve any
      in-flight requests with null so callers don't hang, and land on "error"
      so the UI shows a reload affordance instead of sitting on "loading"
      forever. */
  private handleWorkerFailure() {
    // with the replay fallback on, a dead worker is just another reason to fall
    // over — do that instead of surfacing a terminal "worker" error.
    if (this.replayFallback && !this.replay) {
      this.triggerReplay();
      return;
    }
    this.worker?.terminate();
    this.worker = null;
    this.gen++; // orphan any reply that somehow still lands
    this.pending.forEach((resolve) => resolve(null as never));
    this.pending.clear();
    this.m.status = "error";
    this.m.errorReason = "worker";
    this.m.loss = NaN;
    this.m.smoothLoss = NaN;
    this.m.initialLoss = NaN;
    this.m.lossHistory = [];
    this.m.batches = 0;
    this.onUpdate?.();
  }

  // ── replay fallback: watchdog + one-way swap ────────────────────────────
  /** After every real-path state update: disarm the load watchdog once a first
      training batch has landed, or — if the update is a terminal error — hand
      off to the replay (when replayFallback is on). */
  private afterRealUpdate() {
    if (!this.replayFallback || this.replay) return;
    if (!this.reachedTraining && this.status === "training" && this.batches > 0) {
      this.reachedTraining = true;
      this.clearLoadWatchdog();
    }
    if (this.status === "error") this.triggerReplay();
  }

  /** Arm (on start) the timer that fires iff the real path never reaches a
      first training batch — the dead-on-arrival wedge, where tf.ready hangs and
      the worker posts nothing, so no state update ever arrives to react to. */
  private armLoadWatchdog() {
    if (!this.replayFallback) return;
    this.clearLoadWatchdog();
    this.reachedTraining = false;
    this.loadWatchdog = setTimeout(() => {
      this.loadWatchdog = null;
      if (!this.reachedTraining && !this.replay) this.triggerReplay();
    }, this.replayWatchdogMs);
  }

  private clearLoadWatchdog() {
    if (this.loadWatchdog !== null) {
      clearTimeout(this.loadWatchdog);
      this.loadWatchdog = null;
    }
  }

  /** Give up on the real path and hand off to the replay behind the SAME
      surface — the host just keeps rendering (status runs loading→training→
      converged as usual; it never learns WHY the live run failed). One-way for
      the session: reset()/start() thereafter replay again, not retry the real
      path (which would only re-wedge for another watchdog interval). */
  private triggerReplay() {
    this.clearLoadWatchdog();
    // tear the real path down completely — terminate() releases the worker's
    // dead GL context instead of leaving it pinned for the tab's life
    this.worker?.terminate();
    this.worker = null;
    this.core?.reset();
    this.core = null;
    this.gen++;
    this.pending.forEach((resolve) => resolve(null as never));
    this.pending.clear();
    this.m.status = "loading";
    this.m.errorReason = null;
    this.replay = new ReplayTrainer();
    void this.replay.start(this.onUpdate, this.assetBase);
  }

  private send(msg: WorkerRequest) {
    this.worker!.postMessage(msg);
  }

  /** Ask the worker and get a typed reply matched by request id. */
  private request<T>(
    msg: DistributiveOmit<Extract<WorkerRequest, { id: number }>, "id" | "gen">
  ): Promise<T> {
    return new Promise<T>((resolve) => {
      const id = this.nextId++;
      this.pending.set(id, resolve as (result: never) => void);
      this.send({ ...msg, id, gen: this.gen } as WorkerRequest);
    });
  }

  /**
   * Kick off training (loads tfjs + embeddings in the worker on first call).
   * onUpdate fires after every batch and on status changes, same as before.
   * `cfg` is the user's ⚙ run config; start() installs it on BOTH sides —
   * this thread (Hero's demo-cycle layout/sentence sampling reads the same
   * module state) and the training thread (worker via the message, or the
   * shared module instance in inline-fallback mode).
   */
  start(onUpdate?: () => void, cfg: RunConfig = DEFAULT_RUN_CONFIG) {
    this.onUpdate = onUpdate;
    setRunConfig(cfg);
    // Already fell over to the replay this session → restart just replays again
    // (a fresh randomized run), never re-attempts the real path (it would only
    // re-wedge for another watchdog interval).
    if (this.replay) {
      void this.replay.start(onUpdate, this.assetBase);
      return;
    }
    this.ensureBackend();
    // arm the "never reached a first training batch" watchdog for this attempt
    // (no-op unless replayFallback is on)
    this.armLoadWatchdog();
    if (this.core) {
      // wrap onUpdate so the inline path feeds the watchdog / error swap too
      void this.core.start(() => {
        this.afterRealUpdate();
        if (!this.replay) onUpdate?.();
      }, this.assetBase);
      return;
    }
    // "error" restarts too: an "assets" failure un-cached its promise, so a
    // retry refetches. (A "worker" failure already dropped the worker, so
    // ensureBackend just made a fresh one — futile against a dead chunk URL,
    // but harmless, and the host is told to reload rather than retry.)
    if (this.m.status !== "idle" && this.m.status !== "error") return;
    this.m.status = "loading";
    this.m.errorReason = null;
    this.send({
      t: "start",
      gen: this.gen,
      cfg,
      assetBase: this.assetBase,
      replayFallback: this.replayFallback,
    });
    onUpdate?.();
  }

  /** Halt gradient steps without touching the model (Resume continues). */
  pause() {
    if (this.replay) return this.replay.pause();
    if (this.core) return this.core.pause();
    if (this.m.status !== "training") return;
    this.m.status = "paused";
    this.send({ t: "pause", gen: this.gen });
  }

  resume() {
    if (this.replay) return this.replay.resume();
    if (this.core) return this.core.resume();
    if (this.m.status !== "paused") return;
    this.m.status = "training";
    this.send({ t: "resume", gen: this.gen });
  }

  /** Stop training and discard the learned weights (fresh model next start).
      The worker itself stays warm — tfjs + embeddings stay cached, so a
      restart skips straight to model build. */
  reset() {
    this.clearLoadWatchdog();
    if (this.replay) return this.replay.reset();
    if (this.core) return this.core.reset();
    this.gen++;
    this.m.status = "idle";
    this.m.errorReason = null;
    this.m.loss = NaN;
    this.m.smoothLoss = NaN;
    this.m.initialLoss = NaN;
    this.m.lossHistory = [];
    this.m.batches = 0;
    // in-flight prediction replies will still arrive and resolve; anything
    // queued after this reset computes against a disposed model → null
    if (this.worker) this.send({ t: "reset", gen: this.gen });
  }

  /** Tear the worker down entirely (component unmount). */
  destroy() {
    this.reset();
    this.replay?.destroy();
    this.replay = null;
    this.worker?.terminate();
    this.worker = null;
    this.pending.clear();
    this.core = null;
  }

  /** Freeze the current policy weights for the per-cycle rollout snapshot.
      Fire-and-forget; postMessage FIFO guarantees it lands before any predict
      request posted after it. */
  snapshotPolicy() {
    if (this.replay) return this.replay.snapshotPolicy();
    if (this.core) return this.core.snapshotPolicy();
    if (!this.ready) return;
    this.send({ t: "snapshot", gen: this.gen });
  }

  /**
   * Policy inference against the frozen per-cycle snapshot — async (the
   * render + forward pass happen in the worker). Replies with the target
   * angles PLUS the spatial-attention readout from
   * the same pass. `carry` is the rollout's currently-held block (rendered
   * at the effector in the model's-eye view — the carry-phase state cue).
   * The caller keeps stepping toward its previous target until the reply
   * lands; see Hero.tsx.
   */
  predictFrozenTarget(
    a1: number,
    a2: number,
    tokens: number[],
    layout: Layout,
    carry: number | null = null
  ): Promise<PredictResult | null> {
    if (this.replay)
      return this.replay.predictFrozenTarget(a1, a2, tokens, layout, carry);
    if (this.core)
      return Promise.resolve(
        this.core.predictFrozenTarget(a1, a2, tokens, layout, carry)
      );
    if (!this.ready) return Promise.resolve(null);
    return this.request({ t: "predict", a1, a2, tokens, layout, carry });
  }

  /**
   * Same inference, but on the LIVE (still-training) model — powers the
   * Vision Encoder panel's "where the model looks" heatmap during training,
   * queried against the demonstration's current state.
   */
  predictLive(
    a1: number,
    a2: number,
    tokens: number[],
    layout: Layout,
    carry: number | null = null
  ): Promise<PredictResult | null> {
    if (this.replay)
      return this.replay.predictLive(a1, a2, tokens, layout, carry);
    if (this.core)
      return Promise.resolve(
        this.core.predictTarget(a1, a2, tokens, layout, carry)
      );
    if (!this.ready) return Promise.resolve(null);
    return this.request({ t: "predictLive", a1, a2, tokens, layout, carry });
  }

  /** Decode the acted-on color from a token sequence via the color head. */
  decodeCommand(tokens: number[]): Promise<DecodedCommand | null> {
    if (this.replay) return this.replay.decodeCommand(tokens);
    if (this.core) return Promise.resolve(this.core.decodeCommand(tokens));
    if (!this.ready) return Promise.resolve(null);
    return this.request({ t: "decode", tokens });
  }

  /** The language encoder's per-token attention weights (live token bars). */
  attentionWeights(tokens: number[]): Promise<number[] | null> {
    if (this.replay) return this.replay.attentionWeights(tokens);
    if (this.core) return Promise.resolve(this.core.attentionWeights(tokens));
    if (!this.ready) return Promise.resolve(null);
    return this.request({ t: "attention", tokens });
  }
}
