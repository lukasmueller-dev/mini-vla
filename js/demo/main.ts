// Minimal standalone demo: two canvases (the IK-expert DEMONSTRATION and the
// live POLICY ROLLOUT), a Start/Pause + Reset control, and a loss readout. No
// encoder panels — those are the host's job (they live portfolio-side). This
// file proves the package trains and rolls out with zero host wiring: the
// trainer's module Worker, the rollout engine, the task generator and the scene
// renderer are all imported straight from the package source.

import { VLATrainer } from "../src/trainer";
import { RolloutEngine, type RolloutFrame } from "../src/rollout";
import { paintScene, sceneMap, type ScenePalette } from "../src/scene";
import { DEMO_PERIOD_MS, demoPose, makeDemoPlan, type DemoPlan } from "../src/demo";
import {
  DEFAULT_LAYOUT,
  DEFAULT_SENTENCE,
  randomLayout,
  sampleCommand,
  type Layout,
  type Sentence,
} from "../src/examples";
const ACCENT = "#e12d1a";
const PALETTE: ScenePalette = {
  floor: "#e6e6e6",
  pedestal: "#2b2b2b",
  link: "#8a8a8a",
  joint: "#fff",
  effectorOpen: "#fff",
  effectorOpenEdge: "#6f6f6f",
  effectorClosed: "#6f6f6f",
};

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const demoCanvas = $<HTMLCanvasElement>("demo");
const rolloutCanvas = $<HTMLCanvasElement>("rollout");
const primaryBtn = $<HTMLButtonElement>("primary");
const resetBtn = $<HTMLButtonElement>("reset");
const promptEl = $("prompt");
const statusEl = $("status");
const lossEl = $("loss");
const batchesEl = $("batches");

const trainer = new VLATrainer();
const engine = new RolloutEngine();

// ── demo / rollout state (a stripped copy of the hero's loop bookkeeping) ──
let demoLayout: Layout = DEFAULT_LAYOUT.map((b) => ({ ...b }));
let demoSentence: Sentence = DEFAULT_SENTENCE;
let demoPlan: DemoPlan = makeDemoPlan(demoLayout, demoSentence);
let rolloutLayout: Layout = DEFAULT_LAYOUT.map((b) => ({ ...b }));
let lastCycle = -1;
let demoT = 0;
// clock anchored to the click (+500ms hold), pause-compensated
let trainStart = 0;
let pausedAccum = 0;
let pauseStart: number | null = null;

function fit(c: HTMLCanvasElement) {
  const dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth || 260;
  const H = c.clientHeight || 240;
  if (c.width !== Math.round(W * dpr)) c.width = Math.round(W * dpr);
  if (c.height !== Math.round(H * dpr)) c.height = Math.round(H * dpr);
  const ctx = c.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  return { ctx, W, H };
}

const trailToPx = (trail: { x: number; y: number }[], W: number, H: number) => {
  const m = sceneMap(W, H);
  return trail.map((p) => ({ x: m.X(p.x), y: m.Y(p.y) }));
};

const predictFrozen = (
  a1: number,
  a2: number,
  tokens: number[],
  layout: Layout,
  carry: number | null
) => trainer.predictFrozenTarget(a1, a2, tokens, layout, carry);

// Advance the shared demo cycle: a fresh scene + command each period; the
// rollout re-syncs (clone the scene, freeze the policy, begin an attempt).
function advanceCycle(now: number) {
  if (!trainer.ready || trainer.status === "paused") return;
  const eff = now - trainStart - pausedAccum;
  if (eff < 0) return;
  const cycle = Math.floor(eff / DEMO_PERIOD_MS);
  if (cycle !== lastCycle) {
    lastCycle = cycle;
    demoLayout = randomLayout();
    demoSentence = sampleCommand(demoLayout);
    demoPlan = makeDemoPlan(demoLayout, demoSentence);
    rolloutLayout = demoLayout.map((b) => ({ ...b }));
    trainer.snapshotPolicy();
    engine.begin(demoSentence.color, demoSentence.tokens);
    promptEl.textContent = `“${demoSentence.text}”`;
  }
  demoT = (eff % DEMO_PERIOD_MS) / DEMO_PERIOD_MS;
}

function drawDemo() {
  const { ctx, W, H } = fit(demoCanvas);
  const pose = demoPose(demoPlan, trainer.ready ? demoT : 0);
  paintScene(ctx, W, H, {
    a1: pose.a1,
    a2: pose.a2,
    layout: demoLayout,
    accent: ACCENT,
    carry: pose.carry,
    grip: pose.grip,
    palette: PALETTE,
  });
}

function drawRollout(now: number) {
  const { ctx, W, H } = fit(rolloutCanvas);
  const f: RolloutFrame =
    trainer.ready && trainer.status !== "paused" && engine.hasEpisode
      ? engine.step(now, rolloutLayout, predictFrozen)
      : engine.frame();
  paintScene(ctx, W, H, {
    a1: f.a1,
    a2: f.a2,
    layout: rolloutLayout,
    accent: ACCENT,
    trail:
      f.phase === "reach" || f.phase === "carry"
        ? trailToPx(f.trail, W, H)
        : null,
    lossNorm: trainer.lossNorm(),
    carry: f.carry,
    grip: f.grip,
    palette: PALETTE,
  });
}

function loop(now: number) {
  advanceCycle(now);
  drawDemo();
  drawRollout(now);
  requestAnimationFrame(loop);
}

// ── controls ──
function refreshHud() {
  statusEl.textContent = trainer.status;
  lossEl.textContent = Number.isNaN(trainer.loss) ? "—" : trainer.loss.toFixed(3);
  batchesEl.textContent = String(trainer.batches);
  primaryBtn.textContent =
    trainer.status === "training"
      ? "Pause"
      : trainer.status === "paused"
        ? "Resume"
        : trainer.status === "converged"
          ? "Trained ✓"
          : trainer.status === "loading"
            ? "Loading…"
            : // a dead worker chunk can't be retried in-page, only reloaded;
              // a failed asset load refetches on the next start()
              trainer.status === "error"
              ? trainer.errorReason === "worker"
                ? "Reload"
                : "Retry"
              : "Start Training";
  primaryBtn.disabled =
    trainer.status === "loading" || trainer.status === "converged";
}

const onUpdate = () => refreshHud();

function resetState() {
  engine.reset();
  lastCycle = -1;
  demoLayout = DEFAULT_LAYOUT.map((b) => ({ ...b }));
  demoSentence = DEFAULT_SENTENCE;
  demoPlan = makeDemoPlan(demoLayout, demoSentence);
  rolloutLayout = DEFAULT_LAYOUT.map((b) => ({ ...b }));
  demoT = 0;
  pausedAccum = 0;
  pauseStart = null;
  promptEl.textContent = "press Start Training";
}

primaryBtn.onclick = () => {
  if (trainer.status === "training") {
    trainer.pause();
    pauseStart = performance.now();
  } else if (trainer.status === "paused") {
    if (pauseStart !== null) pausedAccum += performance.now() - pauseStart;
    pauseStart = null;
    trainer.resume();
  } else if (trainer.status === "error" && trainer.errorReason === "worker") {
    location.reload();
  } else if (trainer.status === "idle" || trainer.status === "error") {
    resetState();
    trainStart = performance.now() + 500; // brief hold on the resting scene
    trainer.start(onUpdate);
  }
  refreshHud();
};

resetBtn.onclick = () => {
  trainer.reset();
  resetState();
  refreshHud();
};

// idle rest pose is just REST over the default scene
engine.reset();
refreshHud();
requestAnimationFrame(loop);
