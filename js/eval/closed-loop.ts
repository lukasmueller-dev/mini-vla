// DEV-ONLY shared closed-loop scorer — the ONE headless rollout integrator both
// the replay-capture grasp gate (js/capture/main.ts) and the sweep harness
// (js/eval/main.ts) score policies with. Keeping it single-sourced is the whole
// point of the gate: the capture step only ships a policy that grasps under the
// SAME rollout the replay actually runs, so any drift between "how we score" and
// "how we play back" would silently invalidate it.
//
// This is a faithful mirror of the live RolloutEngine (src/rollout.ts):
// proportional step toward the latest prediction, re-predict every PRED_EVERY
// frames, learned-gripper grasp on the rising edge while the effector is over
// the commanded block (via THE shared effectorOverBlock predicate), then
// settle-then-done. NOT part of the published package. (The runtime
// RolloutEngine stays separate on purpose — it's stateful/rAF-driven and emits
// RolloutFrames for rendering; unifying with it is a riskier, separate change.)

import { CONFIG } from "../src/config";
import type { VLATrainerCore } from "../src/trainer.core";

/** Closed-loop rollout metrics over simulated episodes with the converged
    policy — the dial the open-loop probes lack: probes score held-out Huber on
    random states, but "does the arm actually reach, grasp and seat, and how
    much do its predictions wobble en route" is a property of the CLOSED loop. */
export interface RolloutEval {
  episodes: number;
  graspRate: number;
  /** mean frames from episode start to grasp (successful grasps only). */
  meanGraspFrames: number | null;
  /** mean |Δ target| (rad) between consecutive reach-phase predictions. */
  reachJitter: number | null;
}

/** Score a converged core CLOSED-LOOP over `episodes` fresh episodes. Returns
    the full rollout metrics, or `null` iff the WebGL context died mid-scoring
    (predictTarget returned null) — callers map that dead-context case to their
    own sentinel (the capture gate to −1 so it can't win best-of-N; the eval
    harness to a zeroed RolloutEval). `onEpisode` is invoked at the start of each
    episode (after the yield) so callers can log progress without baking their
    logging into the shared loop. */
export async function scoreGraspRate(
  core: VLATrainerCore,
  episodes: number,
  onEpisode?: (index: number, total: number) => void
): Promise<RolloutEval | null> {
  const { randomLayout, sampleCommand, blockOfColor } = await import(
    "../src/examples"
  );
  const { REST, effectorOverBlock } = await import("../src/geometry");
  const R = CONFIG.rollout;
  const GRIP_RADIUS = CONFIG.gripper.radius;
  const GRIP_THRESHOLD = CONFIG.gripper.threshold;
  const PRED_EVERY = 6; // frames between re-predictions ≈ the engine's predictMs
  const MAX_FRAMES = 300; // SwiftShader predicts are ~50-250ms each — keep
  // the worst-case predict count bounded (episodes × MAX_FRAMES/PRED_EVERY)

  let grasps = 0;
  let graspFramesSum = 0;
  let jitterSum = 0;
  let jitterN = 0;

  for (let e = 0; e < episodes; e++) {
    // yield between episodes so the runner's polling isn't starved for the
    // whole eval (the frame loop is sync)
    await new Promise((r) => setTimeout(r, 0));
    onEpisode?.(e, episodes);
    const layout = randomLayout();
    const cmd = sampleCommand(layout);
    const target = blockOfColor(layout, cmd.color);

    let a1 = REST[0];
    let a2 = REST[1];
    let carry: number | null = null;
    let pred: [number, number] = [a1, a2];
    let prevPred: [number, number] | null = null;
    let near = 0;
    let settle = 0;
    let predGrip = 0;
    let sawOpen = false;

    for (let f = 0; f < MAX_FRAMES; f++) {
      if (f % PRED_EVERY === 0) {
        const p = core.predictTarget(a1, a2, cmd.tokens, layout, carry);
        if (!p) return null; // context died mid-scoring — sentinel, NOT a real 0% grasp
        if (prevPred && carry === null && f > PRED_EVERY)
          // reach-phase prediction wobble, skipping the first transient
          (jitterSum += Math.hypot(
            p.target[0] - prevPred[0],
            p.target[1] - prevPred[1]
          )),
            jitterN++;
        prevPred = pred = p.target;
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
            graspFramesSum += f;
            prevPred = null; // carry phase — jitter metric stays reach-only
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
  return {
    episodes,
    graspRate: grasps / episodes,
    meanGraspFrames: grasps ? graspFramesSum / grasps : null,
    reachJitter: jitterN ? jitterSum / jitterN : null,
  };
}
