# mini-vla

A **standalone, in-browser vision-language-action demo**. A TensorFlow.js
behavior-cloning policy trains *live in the browser* against an analytical
inverse-kinematics expert on a 2-link arm, then executes pick-up commands
("grab the blue cube") in a closed loop — reaching with an open gripper,
closing on the block, and lifting it home. No server, no tfjs-node: the
training runs in a real (or headless) browser, always.

This is the model + task + trainer + rollout engine + renderers, packaged as an
**ESM-only TypeScript source package**. It ships the framework-free `src/` and
the pretrained embedding assets; a host (e.g. a Next.js site) transpiles the
source and renders whatever UI it likes on top of the data the package exposes.

```
npm install
npm run demo     # Vite page: watch it train + roll out (open → close → lift)
npm run eval     # headless sweep: train to convergence, print grasp metrics
npm run typecheck
```

`npm run eval` needs a Chromium — `npx playwright install chromium` in a fresh
clone (or set `MINIVLA_CHROMIUM=/path/to/chromium` to reuse an existing binary).

## What the model is

One forward pass of a language-conditioned **spatial-attention** policy produces
**four heads** — this is the contract a host renders from:

| head | output | drives |
|------|--------|--------|
| **action** | target joint angles `[shoulder, elbow]` | the arm's reach/carry |
| **color** | which block the command names | the "decoded target" readout, try-it routing |
| **gripper** | a sigmoid `0 (open) → 1 (closed)` | the grasp (its rising edge over a block) |
| **attention** | a spatial map over the vision grid (+ its soft-argmax) | the "where the model looks" heatmap |

A conv stack turns the 32×32 silhouette into a `G×G` feature map; a single
language query (an attention-pooled, GloVe-embedded sentence) dot-product-scores
every cell; a spatial softmax makes the attention map; the readout is the map's
soft-argmax plus its attention-weighted features. A small dense head regresses
the joint angles from that readout. The embedding table is a frozen ~20k-word
GloVe slice, so unseen near-synonyms ("gold", "violet") still resolve.

## Architecture

```
              ┌───────────── training thread (Web Worker) ─────────────┐
  task.ts ───▶│  trainer.core.ts  synth batch → model.ts → grad step   │
 (scenes,     │        ▲  paintSilhouette(scene) = the model's input    │
  commands,   │        │  loadEmbeddings(assets/) = frozen GloVe table   │
  demoPose)   └────────┼────────────────────────────────────────────────┘
                       │  trainer.ts (proxy): async predict / telemetry
                       ▼
  rollout.ts  RolloutEngine.step() ── reach → grasp → carry → lift ──▶ RolloutFrame
                       │
  scene.ts   paintScene(frame, palette)   ·   host renders panels/overlays
```

- **`geometry.ts`** — the 2-link arm, IK expert, and **`effectorOverBlock`**, THE
  grasp predicate (shared by the training label, the `RolloutEngine` gate, and
  the headless eval — never copied).
- **`rollout.ts`** — `RolloutEngine`: the closed-loop episode machine (phases,
  the learned-grasp rising-edge gate, carry attachment, stepping toward the last
  async-predicted target while replies are in flight). Emits a serializable
  `RolloutFrame` per step.
- **`scene.ts`** — `paintScene` (the styled display; takes a host `palette`) and
  `paintSilhouette` (the model's-eye view — **no palette**, its tones are tuned
  so the color head keys on blocks, not the robot; it versions with the model).

## Package surface

Import subpaths (TS source, consumed via a bundler that transpiles it, e.g.
`transpilePackages` in Next):

| subpath | exposes |
|---------|---------|
| `mini-vla/trainer` | `VLATrainer`, `TrainerStatus`, `PredictResult`, `DecodedCommand` |
| `mini-vla/rollout` | `RolloutEngine`, `RolloutFrame` |
| `mini-vla/scene` | `paintScene` (palette + grip), `paintSilhouette`, `sceneMap`, `effectorPx` |
| `mini-vla/task` | tokenizer/examples (`registerFullVocab`, `sampleCommand`, layouts, colors) + `demoPose`, `makeDemoPlan`, `DEMO_PERIOD_MS` |
| `mini-vla/geometry` | `fk`, `REST`, `effectorOverBlock`, … |
| `mini-vla/config` | `CONFIG` (incl. `CONFIG.gripper`), `RunConfig`, `setRunConfig`, `DEFAULT_RUN_CONFIG` |
| `mini-vla/model` | `IMG_SIZE`, `ATTN_GRID`, … |

`@tensorflow/tfjs` is a **peer dependency** — the host provides exactly one copy
(it's only ever loaded via a dynamic import inside `trainer.ts`, so importing
`mini-vla/model` for `IMG_SIZE` stays tfjs-free at runtime).

## Gotchas (read before refactoring)

- **ESM-only, single instance.** The tokenizer vocab (`examples.ts`), the
  installed run-config (`run-config.ts`), and the `CONFIG` knobs (`config.ts`)
  are **per-thread module-state singletons**. The training Worker posts its
  loaded 20k-word vocab back to the main thread (`{t:"vocab"}` →
  `registerFullVocab`) precisely because module state is per-thread. Both sides
  MUST import the *same* package instance — a dual CJS/ESM build (two instances)
  silently breaks the vocab post-back and typed near-synonyms. Do not add a CJS
  build.
- **Import order in `eval/`.** `trainer.core.ts` (and `model.ts` under it)
  snapshot `CONFIG` values into module constants on first evaluation. Any
  `?set=` knob override MUST be applied to `CONFIG` *before* `trainer.core` is
  imported — `eval/main.ts` therefore statically imports only `config` +
  `run-config` and **dynamic-imports** `trainer.core` (and `geometry`/`examples`)
  afterward. This is the first thing a future refactor will break.
- **The gripper head needs its dedicated samples.** It only learns because of
  the grasp-class training samples (`CONFIG.trainer.graspFrac`) and a non-zero
  `CONFIG.model.gripperLossWeight` — remove either and the head collapses to a
  constant. And don't lower the block-size floor (`CONFIG.block.min = 0.12`):
  below it, closed-loop grasps miss small blocks (measured graspRate ≈ 0.13 at
  0.08).
- **The rollout keeps stepping toward its last target while a predict is in
  flight** — this is intentional (hides async latency); don't "fix" it into
  awaiting each prediction.
- Embedding-matrix rows 0/1 (`<pad>`/`<unk>`) are zero on purpose.

## Assets

`assets/embeddings-50d.bin` + `assets/vocab.txt` are int8-quantized GloVe 50d
vectors + the word list, generated by `npm run gen:embeddings` (rerun only when
the grammar, vocab size, or quantization changes; they're committed). A host
serves them wherever `loadEmbeddings({ assetBase })` points — default `/vla`.
