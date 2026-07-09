// Public `mini-vla/task` entrypoint: scene + command generation (layouts,
// colors, the slot-grammar tokenizer, registerFullVocab) AND the scripted
// demonstration (demoPose / makeDemoPlan / DEMO_PERIOD_MS, DemoPose incl. its
// gripper flag). Grouped because a host renders both the demonstration and the
// commands from the same task definition.
export * from "./examples";
export * from "./demo";
