// Public `mini-vla/config` entrypoint: the model/task knobs (CONFIG, incl.
// CONFIG.gripper) AND the run-config — the palette/density profile a host
// installs before training (RunConfig, the fixed DESKTOP_RUN_CONFIG /
// MOBILE_RUN_CONFIG profiles + PRESETS, DEFAULT_RUN_CONFIG, setRunConfig,
// estimateTrainingSeconds). setRunConfig installs per-thread module state, so
// the host and the training thread must both import THIS one package instance
// (see the ESM-singleton note in the README).
export * from "./config";
export * from "./run-config";
