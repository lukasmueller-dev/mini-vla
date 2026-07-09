// The run configuration of the VLA hero — the scene DIFFICULTY a run trains on
// (palette size, scene density), as opposed to CONFIG (src/config.ts), the
// developer knob sheet. CONFIG tunes HOW it trains; a RunConfig picks WHAT.
//
// There is no user picker anymore: instead of choosing numColors/maxBlocks by
// hand, the host ships two FIXED named profiles (DESKTOP / MOBILE below) and
// serves one by device class — DESKTOP (the hardest setting) on ≥1100px,
// MOBILE (a lighter, faster-converging task) on phone-class viewports, where
// every gradient step costs battery and heat. The host resolves the profile and
// passes it into trainer.start(cfg); the package never detects the device
// itself (the trainer runs in a Worker with no window/matchMedia).
//
// The active RunConfig is plain module state, and — like examples.ts's
// registerFullVocab — it must be installed on BOTH threads: the main thread
// (Hero's demo-cycle layout/sentence sampling) and the trainer worker (batch
// synthesis) each hold their own copy of this module. Hero calls setRunConfig
// before trainer.start(); the proxy ships the config inside the {t:"start"}
// message and the worker installs it before building the model. Latching the
// profile at Start (not tracking a live media query) is deliberate: a viewer
// resizing across the breakpoint mid-run must not leave the main thread's
// randomLayout() sampling scenes the worker is not training on.
//
// It deliberately does NOT change any model shape: the color head stays
// 8-wide regardless of the profile — numColors only restricts what the
// samplers draw, so every RunConfig trains the same architecture and the
// calibrated CONFIG numbers stay comparable. (The host relies on this: an
// untrained color word yields a confident WRONG answer, not an error — its
// "never learned that color" guard is built on the head staying 8-wide.)

import { CONFIG } from "./config";

export interface RunConfig {
  /** Palette size — scenes draw colors from the FIRST N entries of COLORS
      (see activePalette in examples.ts, the single source of that rule). */
  numColors: 2 | 4 | 8;
  /** Scene density cap — a scene holds 2..min(maxBlocks, numColors) blocks
      (colors are unique per scene, so the palette also caps the count). */
  maxBlocks: 2 | 3 | 4;
}

// ── Named profiles the host ships (mirrors mini_vla/run_config.py) ──────────
/** DESKTOP — the HARDEST setting: the full 8-color palette, up to 4
    blocks/scene. The default when no profile is passed. */
export const DESKTOP_RUN_CONFIG: RunConfig = { numColors: 8, maxBlocks: 4 };
/** MOBILE — the lighter budget profile: a 4-color palette, up to 3
    blocks/scene. An easier task that converges in fewer batches, to fit a
    phone's train budget. Its four colors are the palette's first four
    (red, black, blue, yellow); activePalette(MOBILE_RUN_CONFIG) is the exact
    set its scenes contain. */
export const MOBILE_RUN_CONFIG: RunConfig = { numColors: 4, maxBlocks: 3 };

/** Selectable by name — the host's device-class pick (and train.py's --preset
    on the Python side). */
export const PRESETS: Record<string, RunConfig> = {
  desktop: DESKTOP_RUN_CONFIG,
  mobile: MOBILE_RUN_CONFIG,
};

/** Default landing-page behavior — the hardest (desktop) profile. Also the
    fallback cfg for trainer.start() when the host passes none. */
export const DEFAULT_RUN_CONFIG: RunConfig = DESKTOP_RUN_CONFIG;

let current: RunConfig = { ...DEFAULT_RUN_CONFIG };

/** Install the active run config on THIS thread (defensive copy). */
export function setRunConfig(rc: RunConfig) {
  current = { ...rc };
}

export function runConfig(): RunConfig {
  return current;
}

/** Training-time estimate the host prints in the control bar's idle status
    column ("est. ~51s on a laptop GPU"), from the factor table in CONFIG.eta
    (gauged 2026-07 against the carry-flag pick-up arch — see the eta comment
    there for the measurement basis). Desktop ⇒ ~51s, mobile ⇒ ~42s. */
export function estimateTrainingSeconds(rc: RunConfig): number {
  const e = CONFIG.eta;
  return Math.round(
    e.baseSeconds * e.colorFactor[rc.numColors] * e.blockFactor[rc.maxBlocks]
  );
}
