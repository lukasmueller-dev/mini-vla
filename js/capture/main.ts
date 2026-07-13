// DEV-ONLY: capture the replay fallback's policy checkpoints, GRASP-GATED. NOT
// part of the published package (package.json "files"). For each ATTEMPT it
// trains a REAL VLATrainerCore (WebGL) to convergence on the DEFAULT run config,
// snapshotting the trainable policy weights at a ladder of sample milestones
// (compact format — embedding/grid excluded, policy-weights.ts) plus the final
// converged policy, then scores that final policy CLOSED-LOOP (same rollout
// integrator + grasp gate the replay itself runs). It only ships an attempt that
// GRASPS reliably: fresh random inits vary (≈1/8 collapse to an always-one-side
// policy, plus seed noise), so this is best-of-N — stop early once an attempt
// clears EARLY_ACCEPT, else keep the highest-grasp attempt after MAX_ATTEMPTS.
// run.mjs writes the winner's ladder + its grasp rate to assets/replay/. The
// replay (trainer.replay.ts) then re-plays this exact bad→good trajectory on the
// CPU backend when live training can't run (iOS/iPadOS).

import { CONFIG } from "../src/config";
import { DEFAULT_RUN_CONFIG, setRunConfig } from "../src/run-config";
import type { VLATrainerCore } from "../src/trainer.core";

interface CaptureCkpt {
  samples: number;
  /** Smoothed action loss at capture — the anchor the replay draws through. */
  loss: number;
  /** Flat trainable weights (Float32 values as a plain array for the runner). */
  weights: number[];
}

interface AttemptLog {
  /** Closed-loop grasp rate of this attempt's converged policy, or −1 if it
      never converged (asset/context error). */
  graspRate: number;
  batches: number;
  converged: boolean;
}

declare global {
  interface Window {
    __vlaCapture?: {
      done: boolean;
      status: string;
      batches: number;
      /** Which attempt is training right now (1-based); 0 before the first. */
      attempt: number;
      batchSize: number;
      cadencePerSec: number;
      floorLoss: number;
      /** Gate size + the SHIPPED policy's grasp rate over that many episodes. */
      graspEpisodes: number;
      graspRate: number;
      /** One entry per training attempt — the audit trail run.mjs logs. */
      attempts: AttemptLog[];
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

// ── Grasp gate ──────────────────────────────────────────────────────────────
// Only ship a converged policy that actually grasps. GATE_EPISODES closed-loop
// rollouts score each attempt's final policy; the search stops on the first that
// clears EARLY_ACCEPT, else keeps the best of up to MAX_ATTEMPTS (each a fresh
// random init). A healthy seed grasps ~0.9+, a collapsed one ~0, so this
// reliably rejects the ≈1/8 collapse and seed-unlucky policies that made the old
// single-shot capture "okay-ish, missing grasps".
const GATE_EPISODES = 20;
const EARLY_ACCEPT = 0.9;
const MAX_ATTEMPTS = 8;

/** Closed-loop rollout grasp rate for a converged core — a faithful mirror of
    js/eval/main.ts closedLoopEval and the live RolloutEngine (src/rollout.ts):
    proportional step toward the latest prediction, re-predict every few frames,
    learned gripper grasp on the rising edge while the effector is over the
    commanded block, via THE shared effectorOverBlock predicate. Kept local:
    capture is dev-only tooling, and this scores exactly the weights that ship. */
async function graspRateOf(
  core: VLATrainerCore,
  episodes: number
): Promise<number> {
  const { randomLayout, sampleCommand, blockOfColor } = await import(
    "../src/examples"
  );
  const { REST, effectorOverBlock } = await import("../src/geometry");
  const R = CONFIG.rollout;
  const GRIP_RADIUS = CONFIG.gripper.radius;
  const GRIP_THRESHOLD = CONFIG.gripper.threshold;
  const PRED_EVERY = 6; // frames between re-predictions ≈ the engine's predictMs
  const MAX_FRAMES = 300; // bound the worst-case predict count (SwiftShader)

  let grasps = 0;
  for (let e = 0; e < episodes; e++) {
    await new Promise((r) => setTimeout(r, 0)); // yield so status polling isn't starved
    const layout = randomLayout();
    const cmd = sampleCommand(layout);
    const target = blockOfColor(layout, cmd.color);

    let a1 = REST[0];
    let a2 = REST[1];
    let carry: number | null = null;
    let pred: [number, number] = [a1, a2];
    let near = 0;
    let settle = 0;
    let predGrip = 0;
    let sawOpen = false;

    for (let f = 0; f < MAX_FRAMES; f++) {
      if (f % PRED_EVERY === 0) {
        const p = core.predictTarget(a1, a2, cmd.tokens, layout, carry);
        if (!p) return 0; // model gone — treat the whole attempt as failed
        pred = p.target;
        predGrip = p.grip;
      }
      a1 += (pred[0] - a1) * R.stepGain;
      a2 += (pred[1] - a2) * R.stepGain;

      if (carry === null) {
        // learned grasp gate — SAME predicate as training + the RolloutEngine:
        // effector fully over the commanded block AND the predicted gripper
        // closing on the rising edge (a prior open frame required)
        const closing = predGrip >= GRIP_THRESHOLD;
        const over = effectorOverBlock(a1, a2, target, GRIP_RADIUS);
        if (over && closing && sawOpen) {
          if (++near >= R.nearFrames) {
            carry = cmd.color;
            grasps++;
          }
        } else near = 0;
        if (!closing) sawOpen = true;
      } else if (
        Math.abs(pred[0] - a1) < R.settleEps &&
        Math.abs(pred[1] - a2) < R.settleEps
      ) {
        // carry settled — the arm holds the block aloft; episode done
        if (++settle >= 4) break;
      } else settle = 0;
    }
  }
  return grasps / episodes;
}

const statusEl = document.getElementById("status")!;

(async () => {
  const { VLATrainerCore } = await import("../src/trainer.core");

  const cap: NonNullable<Window["__vlaCapture"]> = {
    done: false,
    status: "idle",
    batches: 0,
    attempt: 0,
    batchSize: CONFIG.trainer.batchSize,
    cadencePerSec: 10, // browser steady-state ≈ 10 batches/s (see config.ts)
    floorLoss: CONFIG.trainer.converge.loss,
    graspEpisodes: GATE_EPISODES,
    graspRate: 0,
    attempts: [],
    weightSpecs: [],
    checkpoints: [],
  };
  window.__vlaCapture = cap;

  let best: {
    specs: { name: string; shape: number[] }[];
    ckpts: CaptureCkpt[];
    graspRate: number;
    batches: number;
  } | null = null;

  for (
    let attempt = 1;
    attempt <= MAX_ATTEMPTS && !(best && best.graspRate >= EARLY_ACCEPT);
    attempt++
  ) {
    cap.attempt = attempt;
    const core = new VLATrainerCore();
    const ckpts: CaptureCkpt[] = [];
    let specs: { name: string; shape: number[] }[] = [];
    let nextMilestone = 0;

    const grab = () => {
      const ck = core.exportTrainableWeights();
      if (!ck) return;
      if (specs.length === 0) specs = ck.specs;
      ckpts.push({
        samples: core.samples,
        loss: Number.isFinite(core.smoothLoss) ? core.smoothLoss : core.loss,
        weights: Array.from(ck.data),
      });
    };

    // start() resolves at "converged" (after snapshotPolicy) or an error status.
    await core.start(() => {
      cap.status = core.status;
      cap.batches = core.batches;
      statusEl.textContent =
        `[capture] attempt ${attempt}/${MAX_ATTEMPTS} ${core.status} ` +
        `b=${core.batches} smooth=${core.smoothLoss.toFixed(4)} ` +
        `ckpts=${ckpts.length}` +
        (best ? ` best=${best.graspRate.toFixed(2)}` : "");
      // capture each milestone the moment training reaches it
      while (
        nextMilestone < MILESTONE_BATCHES.length &&
        core.batches >= MILESTONE_BATCHES[nextMilestone]
      ) {
        grab();
        nextMilestone++;
      }
      if (core.status === "converged") grab(); // the final policy
    });

    if (core.status !== "converged") {
      // errored (dead context / asset load / cpu-fallback exhausted) — log & retry
      cap.attempts.push({ graspRate: -1, batches: core.batches, converged: false });
      core.reset();
      continue;
    }

    // Score the just-converged policy closed-loop, then free it before the next
    // attempt (best.ckpts already holds plain-number weights, so disposal is safe).
    const gr = await graspRateOf(core, GATE_EPISODES);
    cap.attempts.push({ graspRate: gr, batches: core.batches, converged: true });
    if (!best || gr > best.graspRate)
      best = { specs, ckpts, graspRate: gr, batches: core.batches };
    core.reset();
  }

  if (!best) {
    cap.status = "error";
    cap.done = true;
    statusEl.textContent = `[capture] FAILED — no attempt converged in ${MAX_ATTEMPTS} tries`;
    return;
  }

  // Publish the winning attempt's ladder + its grasp rate for run.mjs to write.
  cap.weightSpecs = best.specs;
  cap.checkpoints = best.ckpts;
  cap.graspRate = best.graspRate;
  cap.batches = best.batches;
  cap.status = "converged";
  cap.done = true;
  statusEl.textContent =
    `[capture] DONE — shipped grasp=${best.graspRate.toFixed(2)} ` +
    `over ${GATE_EPISODES} eps in ${cap.attempts.length} attempt(s), ` +
    `${best.ckpts.length} ckpts`;
})();
