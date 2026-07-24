// Headless sweep harness — the portfolio's /vla-lab page ported to a
// framework-free Vite page driven by eval/run.mjs (Playwright). Usage (query
// string, all optional):
//
//   /?colors=4&blocks=3&probe=25&max=600&eval=24&set=model.mapLossWeight:2.5
//
// It runs VLATrainerCore INLINE on the main thread (the core is environment-
// agnostic; the worker indirection buys nothing headless), installs the run
// config + CONFIG knob overrides from the query string, turns on per-bucket
// probe telemetry, and mirrors the full trainer state onto window.__vlaLab
// after every batch for the runner to poll. `max` soft-caps the run.
//
// IMPORT ORDER IS LOAD-BEARING (documented in the README): trainer.core (and
// model.ts underneath) snapshot CONFIG values into module constants when they
// first evaluate, so this file applies the ?set= overrides to CONFIG BEFORE
// dynamic-importing trainer.core — and geometry/examples are imported late,
// inside closedLoopEval. Only CONFIG + run-config are imported statically.

import { CONFIG } from "../src/config";
import { setRunConfig, type RunConfig } from "../src/run-config";
import { scoreGraspRate, type RolloutEval } from "./closed-loop";

declare global {
  interface Window {
    __vlaLab?: {
      status: string;
      batches: number;
      loss: number;
      smoothLoss: number;
      lossHistory: number[];
      colorLoss: number;
      smoothColorLoss: number;
      probes: unknown[];
      rollout?: RolloutEval | null;
      done: boolean;
    };
  }
}

/** Score the converged core closed-loop via THE shared rollout integrator
    (js/eval/closed-loop.ts), logging each episode. A dead context (null) maps
    to a zeroed RolloutEval, exactly as before. */
async function closedLoopEval(
  core: import("../src/trainer.core").VLATrainerCore,
  episodes: number
): Promise<RolloutEval> {
  const r = await scoreGraspRate(core, episodes, (e, total) =>
    console.log(`[eval] episode ${e + 1}/${total}`)
  );
  return (
    r ?? { episodes: 0, graspRate: 0, meanGraspFrames: null, reachJitter: null }
  );
}

const statusEl = document.getElementById("status")!;
const setLine = (s: string) => (statusEl.textContent = s);

const q = new URLSearchParams(window.location.search);
const rc: RunConfig = {
  numColors: (Number(q.get("colors")) || 8) as RunConfig["numColors"],
  maxBlocks: (Number(q.get("blocks")) || 4) as RunConfig["maxBlocks"],
};
setRunConfig(rc);

// CONFIG knob overrides, applied BEFORE trainer.core/model evaluate (see the
// import note above). Numeric values only.
const set = q.get("set");
if (set)
  for (const kv of set.split(",")) {
    const [path, val] = kv.split(":");
    const keys = path.split(".");
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let o: any = CONFIG;
    for (const k of keys.slice(0, -1)) o = o[k];
    o[keys[keys.length - 1]] = Number(val);
  }

(async () => {
  const { VLATrainerCore } = await import("../src/trainer.core");
  const core = new VLATrainerCore();
  core.probeEveryN = Number(q.get("probe")) || 25;
  const maxBatches = Number(q.get("max")) || 0;
  // `max` both soft-caps short runs AND extends past the demo budget — the
  // core's own fallback would otherwise end the run at CONFIG's maxBatches
  if (maxBatches) core.maxBatchesOverride = maxBatches;

  const evalEpisodes = Number(q.get("eval")) || 24;
  let evalStarted = false;
  const publish = () => {
    const done = core.status === "converged" || core.status === "paused";
    window.__vlaLab = {
      status: core.status,
      batches: core.batches,
      loss: core.loss,
      smoothLoss: core.smoothLoss,
      lossHistory: core.lossHistory,
      colorLoss: core.colorLoss,
      smoothColorLoss: core.smoothColorLoss,
      probes: core.probes,
      rollout: window.__vlaLab?.rollout ?? null,
      done,
    };
    setLine(
      `eval: ${core.status} b=${core.batches} smooth=${core.smoothLoss.toFixed(4)} ` +
        `colorSmooth=${core.smoothColorLoss.toFixed(4)}`
    );
    if (maxBatches && core.batches >= maxBatches && core.status === "training")
      core.pause(); // soft cap — keeps the trained weights inspectable
    // once training ends, score the policy CLOSED-LOOP (grasp/seat/jitter)
    if (done && !evalStarted && core.ready) {
      evalStarted = true;
      closedLoopEval(core, evalEpisodes)
        .then((r) => {
          if (window.__vlaLab) window.__vlaLab.rollout = r;
          setLine(`${statusEl.textContent} | rollout ${JSON.stringify(r)}`);
        })
        .catch((err) => {
          console.error("[eval] closed-loop eval failed", err);
          if (window.__vlaLab)
            window.__vlaLab.rollout = {
              episodes: -1,
              graspRate: -1,
              meanGraspFrames: null,
              reachJitter: null,
            };
        });
    }
  };

  core.start(publish);
})();
