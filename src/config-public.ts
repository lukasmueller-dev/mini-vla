// Public `mini-vla/config` entrypoint: the model/task knobs (CONFIG, incl.
// CONFIG.gripper) AND the run-config — the ⚙ palette/density/task-set a host
// picks before training (RunConfig, setRunConfig, DEFAULT_RUN_CONFIG,
// estimateTrainingSeconds). setRunConfig installs per-thread module state, so
// the host and the training thread must both import THIS one package instance
// (see the ESM-singleton note in the README).
export * from "./config";
export * from "./run-config";
