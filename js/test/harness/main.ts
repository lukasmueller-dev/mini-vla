// E2E smoke-test harness — boots the REAL pipeline exactly the way the
// portfolio host does (worker-backed VLATrainer proxy, RolloutEngine, the two
// canvas painters), instrumented for a Playwright spec to poll and drive via
// window.__smoke. NO shortcuts into trainer.core: everything crosses the same
// public surface the website uses, so an engine-specific worker/WebGL failure
// (see the WebKit fence note in trainer.core.ts) fails HERE, not in
// production.
//
// Query params:
//   ?preset=desktop|mobile   which run-config profile to install (default: the
//                            package's own DEFAULT_RUN_CONFIG). The CALLER picks
//                            it — mapping a viewport to a profile is a host
//                            decision, so no breakpoint lives in this package.
//   ?max=N                   auto-pause after N batches (0 = train freely)
//   ?forceInline=1           blank out OffscreenCanvas BEFORE the trainer
//                            boots, forcing the proxy's inline main-thread
//                            fallback (the no-worker legacy path)
//   ?autostart=0             don't start training on load (controls spec
//                            drives start/pause/resume/reset itself)
//   ?assetBase=/custom/base  serve the embeddings from a non-default URL
//                            (omitted → loadEmbeddings' "/vla" default, which
//                            is what the demo/eval pages and the host rely on)

import { VLATrainer } from "../../src/trainer";
import { ReplayTrainer } from "../../src/trainer.replay";
import { embeddingMatrix, loadEmbeddings } from "../../src/embeddings";
import { RolloutEngine } from "../../src/rollout";
import { paintScene, paintSilhouette, DEFAULT_PALETTE } from "../../src/scene";
import {
  PRESETS,
  DEFAULT_RUN_CONFIG,
  runConfig,
  type RunConfig,
} from "../../src/run-config";
import {
  activePalette,
  randomLayout,
  sampleCommand,
  tokenize,
  DEFAULT_LAYOUT,
} from "../../src/examples";
import { REST, THETA1_RANGE, THETA2_RANGE } from "../../src/geometry";
import { ATTN_GRID } from "../../src/model";
import { CONFIG } from "../../src/config";

const q = new URLSearchParams(location.search);

// window/promise errors — the spec asserts this stays empty
const errors: string[] = [];
window.addEventListener("error", (e) => errors.push(String(e.message)));
window.addEventListener("unhandledrejection", (e) =>
  errors.push(String(e.reason))
);

// ?forceInline: kill OffscreenCanvas before the trainer's workerSupported()
// probe runs. trainer.core's make2d() falls back to DOM canvases, so the
// inline path stays fully functional — this is exactly the old-Safari host
// environment.
const forceInline = q.get("forceInline") === "1";
if (forceInline) (globalThis as { OffscreenCanvas?: unknown }).OffscreenCanvas = undefined;

// The package installs whatever RunConfig it is handed; which one a given
// viewport deserves is the host's call, not ours. Specs name the preset they
// want so the matrix still trains both profiles.
const presetName = q.get("preset") ?? "desktop";
const cfg: RunConfig = PRESETS[presetName] ?? DEFAULT_RUN_CONFIG;

const maxBatches = Number(q.get("max")) || 0;
// omitted ⇒ pass nothing, exercising the zero-wiring "/vla" default path
const assetBase = q.get("assetBase") ?? undefined;
// ?replayFallback=1 arms the fallback on the real VLATrainer (a stall/error
// then swaps in the replay). ?forceReplay=1 drives a ReplayTrainer DIRECTLY —
// the deterministic way to exercise the replayed run (scripted loss + real CPU
// rollout) without simulating a 7.5s wedge. Both share the surface the helpers
// below use.
const forceReplay = q.get("forceReplay") === "1";
const replayFallback = q.get("replayFallback") === "1";
// ?watchdogMs=N shrinks the load watchdog so a spec can force the stall→replay
// swap deterministically: a tiny value fires before the real path's first batch
// (which takes seconds), so the fallback engages on a HEALTHY run.
const watchdogMs = Number(q.get("watchdogMs"));
if (watchdogMs > 0) CONFIG.replay.watchdogMs = watchdogMs;
const trainer: VLATrainer | ReplayTrainer = forceReplay
  ? new ReplayTrainer()
  : new VLATrainer({ assetBase, replayFallback });
const doStart = forceReplay
  ? () => (trainer as ReplayTrainer).start(onUpdate, assetBase)
  : () => (trainer as VLATrainer).start(onUpdate, cfg);
const mode = forceReplay ? "replay" : forceInline ? "inline" : "worker";
const engine = new RolloutEngine();

const hud = document.getElementById("hud")!;
const onUpdate = () => {
  if (
    maxBatches &&
    trainer.status === "training" &&
    trainer.batches >= maxBatches
  )
    trainer.pause(); // soft cap — weights stay inspectable, same as the eval page
  hud.textContent =
    `preset=${presetName} mode=${mode}\n` +
    `status=${trainer.status} batches=${trainer.batches} ` +
    `smooth=${trainer.smoothLoss.toFixed(4)} errors=${errors.length}`;
};

/** The full loss curve — a spec compares two runs' curves to assert the replay
    varies between visits (never byte-identical). */
function lossCurve() {
  return trainer.lossHistory.slice();
}

/** Serializable trainer telemetry snapshot. */
function state() {
  return {
    status: trainer.status,
    errorReason: trainer.errorReason,
    batches: trainer.batches,
    loss: trainer.loss,
    smoothLoss: trainer.smoothLoss,
    initialLoss: trainer.initialLoss,
    ready: trainer.ready,
    samples: trainer.samples,
    lossNorm: trainer.lossNorm(),
    // true once the replay fallback is standing in (forced, or swapped in after
    // a stall/error) — the spec asserts the swap actually happened
    usingReplay: forceReplay ? true : (trainer as VLATrainer).usingReplay,
    numColors: runConfig().numColors,
    maxBlocks: runConfig().maxBlocks,
    errors: [...errors],
  };
}

/** Color-head decode of a free-text command (round-trips the worker). */
async function decode(text: string) {
  return trainer.decodeCommand(tokenize(text));
}

/** One live-model inference from REST over a random layout — returns the
    PredictResult contract facts the spec asserts (plus the theta ranges so
    the spec needn't duplicate config). */
async function predict(text: string) {
  const layout = randomLayout();
  const r = await trainer.predictLive(
    REST[0],
    REST[1],
    tokenize(text),
    layout,
    null
  );
  if (!r) return null;
  return {
    target: r.target,
    xy: r.xy,
    grip: r.grip,
    attnLen: r.attn.length,
    attnMax: Math.max(...r.attn),
    attnMin: Math.min(...r.attn),
    attnFinite: r.attn.every(Number.isFinite),
    expectedAttnLen: ATTN_GRID * ATTN_GRID,
    theta1Range: THETA1_RANGE,
    theta2Range: THETA2_RANGE,
  };
}

/** Drive one full RolloutEngine episode against the frozen policy, real-time
    paced (setTimeout(0) per frame ≈ rAF cadence for the async predict
    round-trips). Returns a serializable episode summary. */
async function rollout(maxFrames = 1600) {
  const layout = randomLayout();
  const cmd = sampleCommand(layout);
  trainer.snapshotPolicy(); // FIFO: lands before the first predict request
  engine.begin(cmd.color, cmd.tokens);
  const phases: string[] = [];
  let frames = 0;
  let nonFinite = 0;
  let sawTarget = false;
  let grasped = false;
  while (engine.hasEpisode && frames < maxFrames) {
    const f = engine.step(performance.now(), layout, (a1, a2, t, l, c) =>
      trainer.predictFrozenTarget(a1, a2, t, l, c)
    );
    if (phases[phases.length - 1] !== f.phase) phases.push(f.phase);
    if (!Number.isFinite(f.a1) || !Number.isFinite(f.a2)) nonFinite++;
    if (f.target) sawTarget = true;
    if (f.carry !== null) grasped = true;
    frames++;
    await new Promise((r) => setTimeout(r));
  }
  return {
    frames,
    ended: !engine.hasEpisode,
    phases,
    nonFinite,
    sawTarget,
    grasped,
    command: cmd.text,
  };
}

/** Closed-loop GRASP RATE over `episodes` fresh episodes against the current
    (post-convergence) policy — the perf suite's soft quality floor. Snapshots
    the policy ONCE, then drives each episode through the same RolloutEngine the
    portfolio uses, counting the fraction that actually grasp (carry begins).
    Mirrors js/eval/main.ts's closedLoopEval, but through the public worker
    surface so it scores the same path the site ships. */
async function graspRate(episodes = 12, maxFrames = 1600) {
  trainer.snapshotPolicy(); // freeze the converged weights for every episode
  let grasps = 0;
  for (let e = 0; e < episodes; e++) {
    const layout = randomLayout();
    const cmd = sampleCommand(layout);
    engine.begin(cmd.color, cmd.tokens);
    let frames = 0;
    let grasped = false;
    while (engine.hasEpisode && frames < maxFrames) {
      const f = engine.step(performance.now(), layout, (a1, a2, t, l, c) =>
        trainer.predictFrozenTarget(a1, a2, t, l, c)
      );
      if (f.carry !== null) grasped = true;
      frames++;
      await new Promise((r) => setTimeout(r));
    }
    if (grasped) grasps++;
  }
  return { episodes, grasps, graspRate: grasps / episodes };
}

/** Paint both renderers onto fresh DOM canvases and pixel-check them:
    the display scene must draw SOMETHING (alpha coverage) and the model's-eye
    silhouette must contain non-white pixels including genuinely colored
    (non-gray) block pixels. */
function paintCheck() {
  const W = 260;
  const H = 240;
  const scene = document.createElement("canvas");
  scene.width = W;
  scene.height = H;
  const sctx = scene.getContext("2d")!;
  paintScene(sctx, W, H, {
    a1: REST[0],
    a2: REST[1],
    layout: DEFAULT_LAYOUT,
    accent: "#e12d1a",
    grip: 0,
    palette: DEFAULT_PALETTE,
  });
  const sd = sctx.getImageData(0, 0, W, H).data;
  let scenePainted = 0;
  for (let i = 3; i < sd.length; i += 4) if (sd[i] > 0) scenePainted++;

  const size = CONFIG.trainer.renderSize;
  const sil = document.createElement("canvas");
  sil.width = size;
  sil.height = size;
  const lctx = sil.getContext("2d")!;
  paintSilhouette(lctx, size, REST[0], REST[1], DEFAULT_LAYOUT, null);
  const ld = lctx.getImageData(0, 0, size, size).data;
  let nonWhite = 0;
  let colored = 0;
  for (let i = 0; i < ld.length; i += 4) {
    const [r, g, b] = [ld[i], ld[i + 1], ld[i + 2]];
    if (r < 250 || g < 250 || b < 250) nonWhite++;
    if (Math.abs(r - g) > 24 || Math.abs(g - b) > 24) colored++;
  }
  const px = size * size;
  return {
    sceneCoverage: scenePainted / (W * H),
    silNonWhite: nonWhite / px,
    silColored: colored / px,
  };
}

/** Call loadEmbeddings directly (this thread's module instance — untouched by
    the worker's) and report how it went, plus whether a table actually landed.
    Used with ?autostart=0 so nothing else has loaded the embeddings first:
    lets a spec assert the shape check rejects, that no NaN table is published,
    and that a retry after the rejection refetches instead of returning the
    cached failure. */
async function probeEmbeddings(base?: string) {
  try {
    const m = await loadEmbeddings({ assetBase: base });
    return {
      ok: true as const,
      message: "",
      rows: m.length,
      anyNaN: m.some(Number.isNaN),
      published: embeddingMatrix() !== null,
    };
  } catch (err) {
    return {
      ok: false as const,
      message: err instanceof Error ? err.message : String(err),
      rows: 0,
      anyNaN: false,
      published: embeddingMatrix() !== null,
    };
  }
}

/** The active palette (name + first synonym per color) — the spec derives
    expected decode indices from this instead of hard-coding color order. */
function palette() {
  return activePalette(runConfig()).map((c) => ({
    name: c.name,
    synonym: c.synonyms[0],
  }));
}

declare global {
  interface Window {
    __smoke?: {
      preset: string;
      mode: "worker" | "inline" | "replay";
      assetBase: string | undefined;
      state: typeof state;
      decode: typeof decode;
      predict: typeof predict;
      rollout: typeof rollout;
      graspRate: typeof graspRate;
      lossCurve: typeof lossCurve;
      paintCheck: typeof paintCheck;
      palette: typeof palette;
      probeEmbeddings: typeof probeEmbeddings;
      start: () => void;
      pause: () => void;
      resume: () => void;
      reset: () => void;
      snapshot: () => void;
    };
  }
}

window.__smoke = {
  preset: presetName,
  mode,
  assetBase,
  state,
  decode,
  predict,
  rollout,
  graspRate,
  lossCurve,
  paintCheck,
  palette,
  probeEmbeddings,
  start: doStart,
  pause: () => trainer.pause(),
  resume: () => trainer.resume(),
  reset: () => trainer.reset(),
  snapshot: () => trainer.snapshotPolicy(),
};

onUpdate();
if (q.get("autostart") !== "0") doStart();
