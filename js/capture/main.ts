// DEV-ONLY: capture the replay fallback's policy checkpoints. NOT part of the
// published package (package.json "files"). Trains a REAL VLATrainerCore
// (WebGL) to convergence on the DEFAULT run config and, at a ladder of sample
// milestones, snapshots the trainable policy weights (compact format —
// embedding/grid excluded, policy-weights.ts) plus the SMOOTHED loss at that
// point. run.mjs writes them to assets/replay/. The replay (trainer.replay.ts)
// then re-plays this exact bad→good trajectory on the CPU backend when live
// training can't run (iOS/iPadOS).

import { CONFIG } from "../src/config";
import { DEFAULT_RUN_CONFIG, setRunConfig } from "../src/run-config";

interface CaptureCkpt {
  samples: number;
  /** Smoothed action loss at capture — the anchor the replay draws through. */
  loss: number;
  /** Flat trainable weights (Float32 values as a plain array for the runner). */
  weights: number[];
}

declare global {
  interface Window {
    __vlaCapture?: {
      done: boolean;
      status: string;
      batches: number;
      batchSize: number;
      cadencePerSec: number;
      floorLoss: number;
      weightSpecs: { name: string; shape: number[] }[];
      checkpoints: CaptureCkpt[];
    };
  }
}

// The demo's default profile — the policy the replay stands in for.
setRunConfig(DEFAULT_RUN_CONFIG);

// Sample milestones (batch counts) → the bad→good ladder the replay shows.
// Batch 1 is the first main step (barely-trained, clumsy). The final converged
// state is always captured too, on top of these.
const MILESTONE_BATCHES = [1, 40, 90, 150, 220];

const statusEl = document.getElementById("status")!;

(async () => {
  const { VLATrainerCore } = await import("../src/trainer.core");
  const core = new VLATrainerCore();

  const cap: NonNullable<Window["__vlaCapture"]> = {
    done: false,
    status: "idle",
    batches: 0,
    batchSize: CONFIG.trainer.batchSize,
    cadencePerSec: 10, // browser steady-state ≈ 10 batches/s (see config.ts)
    floorLoss: CONFIG.trainer.converge.loss,
    weightSpecs: [],
    checkpoints: [],
  };
  window.__vlaCapture = cap;

  let nextMilestone = 0;

  const grab = () => {
    const ck = core.exportTrainableWeights();
    if (!ck) return;
    if (cap.weightSpecs.length === 0) cap.weightSpecs = ck.specs;
    cap.checkpoints.push({
      samples: core.samples,
      loss: Number.isFinite(core.smoothLoss) ? core.smoothLoss : core.loss,
      weights: Array.from(ck.data),
    });
  };

  const publish = () => {
    cap.status = core.status;
    cap.batches = core.batches;
    statusEl.textContent = `[capture] ${core.status} b=${core.batches} smooth=${core.smoothLoss.toFixed(
      4
    )} ckpts=${cap.checkpoints.length}`;
    // capture each milestone the moment training reaches it
    while (
      nextMilestone < MILESTONE_BATCHES.length &&
      core.batches >= MILESTONE_BATCHES[nextMilestone]
    ) {
      grab();
      nextMilestone++;
    }
    if (core.status === "converged") {
      grab(); // always capture the final policy (powers "try it" mode)
      cap.done = true;
    } else if (core.status === "error") {
      cap.done = true; // runner reports the failure
    }
  };

  core.start(publish);
})();
