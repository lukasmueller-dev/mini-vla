---
name: port-to-js
description: Port model / task / config / asset changes from the Python source of truth (mini_vla/) to the in-browser TensorFlow.js artifact (js/src/), verify both sides, and produce the downstream go-live + portfolio-site hand-off. Use after changing the Python model, task, config, or embedding assets when you want the change live in the browser demo / portfolio.
user-invocable: true
allowed-tools:
  - Read
  - Write
  - Edit
  - Grep
  - Glob
  - Bash
---

# /port-to-js — Python (`mini_vla/`) → browser TF.js (`js/src/`)

Port a change made in the **Python source of truth** (`mini_vla/`, Keras) into the
**in-browser artifact** (`js/src/`, TensorFlow.js) that the portfolio embeds, then
hand off everything downstream needs.

## The parity contract (read first, never violate)

- **Python is the source of truth.** `js/src/*` is *regenerated from* `mini_vla/*`.
  Never port JS → Python here.
- **Architecture + task + config + silhouette-render parity — NOT weights.** The
  browser retrains from scratch live; you never move a trained checkpoint. So you
  mirror the *network shape, the task, the knob values, and the model-input
  render* — not learned weights, not exact pixels (functional render parity only).
- **Mirror, don't invent.** Keep the two implementations line-for-line
  recognizable. Preserve the JS-side load-bearing rationale comments (they carry
  sweep data the Python doesn't); only change what the Python change forces.

## Module map

| Python (source) | JS target | port rule |
|---|---|---|
| `mini_vla/config.py` | `js/src/config.ts` | port **shared knob values only** (see Phase 1) |
| `mini_vla/run_config.py` | `js/src/run-config.ts` | mirror `RunConfig` + defaults |
| `mini_vla/geometry.py` | `js/src/geometry.ts` | mechanical (FK/IK/predicates) |
| `mini_vla/task.py` | `js/src/examples.ts` | grammar/tokenizer/layouts/colors |
| `mini_vla/render.py` | `js/src/scene.ts` → **`paintSilhouette` only** | model-input render; functional parity |
| `mini_vla/embeddings.py` | `js/src/embeddings.ts` | logic parity, **keep JS I/O** (fetch, not file read) |
| `mini_vla/model.py` | `js/src/model.ts` | Keras→tfjs (see Phase 2) |
| `mini_vla/trainer.py` | `js/src/trainer.core.ts` | batch synth + loop |
| `mini_vla/eval.py` | `js/eval/main.ts` (`closedLoopEval`) | keep the grasp-rate gate in sync |
| `mini_vla/vocab_gen.py` | `js/src/vocab.gen.ts` | **generated — never hand-edit**; both are emitted by `gen:embeddings` |

**JS-only — NO Python counterpart. Never overwrite these from Python:**
`scene.ts` `paintScene` (styled display), `rollout.ts` (animation engine),
`demo.ts` (scripted demonstration), `trainer.ts` + `trainer.worker.ts` (main-thread
worker proxy), `config-public.ts` (re-export barrel), the `js/demo/` UI,
`js/eval/run.mjs` (Playwright harness). If your change needs one of these to move
too (e.g. a new head the rollout must gate on), that's a **task change** — see
Phase 3 — and often a **portfolio change** — see Phase 7.

## Phase 0 — Scope the change

1. Figure out **what changed in Python** since the last port. Ask the user if
   unclear, or diff against the last shipped tag:
   `git diff <last-vX.Y.Z-tag>..HEAD -- mini_vla/ assets/grammar.json`
2. Classify into the phases that apply. A change is rarely one bucket:
   - **Config only** (knob values) → Phase 1.
   - **Architecture** (layers, heads, inputs, grid size, losses) → Phase 2.
   - **Task** (grammar, geometry predicate, new head/label/input, eval gate) →
     Phase 3 (+ usually 1 & 2).
   - **Assets** (grammar words / vocab size / quantization) → Phase 4.
3. Note up front whether the change touches the **model output contract** (the
   exported `PredictResult` / `DecodedCommand` / `RolloutFrame` shapes, the head
   set, `ATTN_GRID`/`G`, model inputs, run-config options, color count). If so,
   flag Phase 7 (portfolio visualization) as **required**, not optional.

Then work the phases in order, verifying (Phase 5) before shipping (Phase 6).

## Phase 1 — Config & params (`config.py` → `config.ts`)

`config.py` is intentionally a **subset** of `config.ts`: it holds only what the
model/training/eval read. `config.ts` additionally owns **demo/display-only knobs
that must be preserved**:

- `CONFIG.demo` (scripted trajectory), `CONFIG.eta` (⚙ ETA estimate).
- `CONFIG.rollout`: Python has only `stepGain`, `nearFrames`, `settleEps`. **Leave
  the JS-only rollout knobs untouched** (`predictMs`, `graspEps`, `reachTimeout`,
  `topHold`, `returnFrames`, `trailLen`, `langMs`, `silRender`).
- `CONFIG.render`: `sceneScale`, `floorY`, `silBlockScale` exist in both — port
  those; nothing else.

Rule: for every knob present in **both**, copy the Python value into `config.ts`.
Never delete a JS-only knob. Field names are camelCase on both sides, so this is
near-mechanical. If a knob was **added** in Python and is model/task-relevant, add
it to `config.ts` (and, if a host tunes it, to the `config-public.ts` surface).
Then mirror `run_config.py` → `run-config.ts` (palette/density + `estimateTrainingSeconds`
lives JS-side via `CONFIG.eta`).

## Phase 2 — Architecture (`model.py` → `model.ts`)

Keras → tfjs layer translation:

| Keras (`tf.keras.layers`) | tfjs (`tf.layers`) |
|---|---|
| `Input` | `tf.input({shape, name, dtype})` — tokens `dtype:"int32"` |
| `Embedding(trainable=False)` | `embedding({inputDim,outputDim,trainable:false})` |
| `Conv1D(1,7,"same")` linear | `conv1d({filters:1,kernelSize:7,padding:"same"})` |
| `Conv2D(...,relu)` / `MaxPooling2D(2)` | `conv2d({...,activation:"relu"})` / `maxPooling2d({poolSize:2})` |
| `Reshape((G*G,C))` | `reshape({targetShape:[G*G,C]})` |
| `Dense` | `dense({units,activation,useBias,trainable})` |
| `Concatenate` | `concatenate()` |
| `Dot(axes=(2,1))` / `(1,1)` | `dot({axes:[2,1]})` / `dot({axes:[1,1]})` |
| `Activation("softmax")` | `activation({activation:"softmax"})` |
| custom `AttentionPooling` | `makeAttentionPooling(tf)` factory (subclass the **dynamically-imported** tf's `Layer`) |

Load-bearing translation rules:

1. **Loss weighting is the #1 divergence.** Keras `compile(loss=[...],
   loss_weights=[1, cw, mw, gw])` — tfjs-layers has **no `lossWeights`**. Translate
   each weighted output to a custom loss function and make the loss array
   **all-functions** (so the action loss is a function too):
   - action → `tf.losses.huberLoss(yTrue,yPred,undefined,ACTION_HUBER_DELTA)`
   - color → `tf.metrics.categoricalCrossentropy(...).mul(COLOR_LOSS_WEIGHT)`
   - map → `tf.metrics.categoricalCrossentropy(...).mul(MAP_LOSS_WEIGHT)`
   - gripper → `tf.metrics.binaryCrossentropy(...).mul(GRIPPER_LOSS_WEIGHT)`
   - the `lang` warm-up twin compiles plain (unweighted) `"categoricalCrossentropy"`.
2. **Keep `action` as output index 0.** `trainer.core` reads `h[1]` (the action
   Huber) as the convergence signal — mirror `trainer.py`'s `_action_loss`.
3. **Post-build weight seeding must be reproduced exactly:** frozen `text_embedding`
   via `setWeights([embed])`; frozen `pick_grid` soft-argmax kernel seeded with the
   same centered+gained `((c+0.5)/G-0.5)*attnCoordGain` coords; `pick_query` kernel
   scaled by `1/√C`.
4. **Weight ORDER must stay identical across the live model and the frozen snapshot**
   (`snapshotPolicy` copies weights by position). Don't reorder layer creation.
5. Keep the three models — `model`, `viz`, `lang` — sharing layers.

**If Phase 2 adds/removes/renames/resizes a head, changes `ATTN_GRID` (G), or adds
a model input → the output contract changed → Phase 7 is required.**

## Phase 3 — Task changes (e.g. "also let it just *touch* the box")

A task change is the highest-touch port. Walk every surface it hits and port each:

| Task-change surface | Python | JS to update |
|---|---|---|
| new verb / task type / words | `assets/grammar.json` (**shared — edit once**) | (same file) → **Phase 4** if new words |
| new/changed geometry predicate | `geometry.py` (e.g. `effector_over_block`) | `geometry.ts` |
| new/changed head or input | `model.py` | `model.ts` (Phase 2 rules) |
| new/changed training label | `trainer.py` `synth_batch` | `trainer.core.ts` `synthBatch` |
| inference/readout shape | `trainer.py` `PredictResult` | `trainer.core.ts` `PredictResult` **+** `trainer.ts` proxy plumbing |
| closed-loop gate | `eval.py` `closed_loop_eval` | `js/eval/main.ts` `closedLoopEval` **+** `rollout.ts` `RolloutEngine` |

**Worked example — add a "touch" capability** (arm may just tap the box instead of
lifting it). Typical Python change → what to port:

- **Grammar:** a `"touch"`/`"tap"` verb or a second task in `grammar.json`. → shared
  file already updated; if `touch`/`tap` aren't in the current vocab, run **Phase 4**.
- **Geometry:** maybe a looser predicate `effector_touches_block` (center-in vs
  fully-contained). → port to `geometry.ts` beside `effectorOverBlock`.
- **Model/labels:** commonly a **task-type input** (one-hot/flag) or a **new head**
  (e.g. a "touch vs lift" intent), plus its label in the batch synth and its loss.
  → `model.ts` (new `tf.input` / new `dense` head + its weighted loss, Phase 2),
  `trainer.core.ts` (label synthesis), and extend `PredictResult`.
- **Rollout/eval:** the closed-loop gate now succeeds on *touch* (no carry/lift). →
  update `closedLoopEval` **and** the `RolloutEngine` phase machine in `rollout.ts`
  (a JS-only file that nonetheless must follow the task).
- **Run-config:** if "touch mode" is user-selectable, extend `RunConfig` +
  `run-config.ts` + the ⚙ menu (**portfolio**, Phase 7).

Always record, in your final summary and the Phase 7 hand-off: *"the task changed —
the demo's behavior and/or contract changed"*, with the specific new
verbs/heads/phases, because that is exactly what the portfolio renders.

## Phase 4 — Assets (grammar words / vocab size / quantization changed)

Only when the **word inventory, vocab size, or quantization** changed (a pure
knob/architecture change does NOT need this):

```bash
npm run gen:embeddings   # needs network — fetches GloVe, ~10MB
```

This rewrites, in lockstep: `assets/embeddings-50d.bin`, `assets/vocab.txt`,
`js/src/vocab.gen.ts`, **and** `mini_vla/vocab_gen.py`. Commit all four. Never
hand-edit the two `vocab_gen`/`vocab.gen` files. Then re-verify **both** sides
resolve the new words (Phase 5). **Downstream:** the portfolio serves these from
`public/vla/` — changed assets must be copied there (Phase 6 / Phase 7).

## Phase 5 — Verify

```bash
# JS
npm run typecheck
npm run demo    # watch it train → roll out (open → close → lift / touch)
npm run eval    # headless grasp/touch metric (needs a Chromium)

# Python still describes the same task (sanity)
python train.py --max-batches 40 --colors 4 --blocks 3
```

Confirm: typecheck clean; the demo trains and the rollout performs the new
behavior; `eval` grasp/touch rate is sane; the JS `PredictResult`/heads match what
Python produces. If a head or `ATTN_GRID` changed, confirm the live "where the
model looks" heatmap and per-token bars still render in `npm run demo`.

**Budget gate (hard product requirement, see `CLAUDE.md`):** the in-browser demo
must train-to-converge **+ roll out in < 60 s**. Check `train.py`'s `[budget]`
line (and, in `npm run demo`, that convergence lands well under 60 s of wall
clock). If the port pushed it over, it is **not shippable** — reduce
batches-to-converge or per-batch compute (`imgSize` / conv / `batchSize`) before
Phase 6.

## Phase 6 — Ship (the Development → live steps)

1. Bump the version in **`package.json`** and **`pyproject.toml`** together.
2. Commit and merge via PR into `main`; then, from an up-to-date `main`, run
   `npm run release -- vX.Y.Z` (the only sanctioned way to tag — see
   `scripts/release.mjs`'s header) and push the tag it creates. Do not
   hand-tag with a raw `git tag` + push — that skips every check the script
   runs.
3. **Portfolio repo (downstream — this is what makes it live):**
   - Bump the dependency to the new tag: `github:lukasmueller-dev/mini-vla#vX.Y.Z`, then
     `npm install`.
   - **If assets changed (Phase 4):** copy `assets/embeddings-50d.bin` +
     `assets/vocab.txt` into the portfolio's `public/vla/`.
   - **If the visualization needs changes (Phase 7):** apply them (hand-off below).
   - Rebuild / redeploy.

The portfolio consumes only the `exports` subpaths (`mini-vla/trainer`,
`mini-vla/rollout`, `mini-vla/scene`, `mini-vla/task`, `mini-vla/geometry`,
`mini-vla/config`, `mini-vla/model`) via `transpilePackages` — internal file moves
are invisible to it, but **contract shape changes are not.**

## Phase 7 — Portfolio visualization: does it need changes?

The portfolio renders directly from the package's contract. **A portfolio change is
required if your port did any of:**

- [ ] added / removed / renamed a **head**, or changed what one outputs
- [ ] changed the **`PredictResult`** fields (`target`, `attn`, `xy`, `grip`, …),
      **`DecodedCommand`** (`color`, `colorProb`), or **`RolloutFrame`** shape
- [ ] changed **`ATTN_GRID` / G** (the attention-map heatmap is `G×G`)
- [ ] added a **model input** (the proxy `trainer.ts` + `Hero` must feed it)
- [ ] added/changed a **`RunConfig`** option (the ⚙ menu) or the **ETA** table
- [ ] changed the **color palette / count** (chips, legend, decoded-target readout)
- [ ] changed a **task phase / gate** the rollout panels animate (e.g. touch vs lift)
- [ ] regenerated **assets** (Phase 4 → copy into `public/vla/`)

If **none** are checked, the portfolio just needs the git-ref bump + redeploy
(Phase 6) — say so explicitly. If **any** are checked, produce the hand-off prompt
below, filled in, and give it to the user to run in a session opened in the
portfolio repo. Do not edit the portfolio repo from here.

### Hand-off prompt (fill the `{{…}}` and hand to a portfolio-site agent)

```
You are working in the portfolio-site repo. The `mini-vla` package was updated
and released as {{vX.Y.Z}}. It is consumed as a git-ref dependency
(github:lukasmueller-dev/mini-vla#<tag>) with transpilePackages and served assets under
public/vla/. Apply the update end to end:

1. Bump the mini-vla dependency to `github:lukasmueller-dev/mini-vla#{{vX.Y.Z}}`, run npm install.
{{IF ASSETS CHANGED}}2. Copy the new embedding assets into public/vla/
   (embeddings-50d.bin + vocab.txt) from the mini-vla repo's assets/.{{/IF}}
3. The model contract changed as follows — update the Hero component and its
   panels accordingly:
   - Heads: {{e.g. added a "touch" intent head (sigmoid, 0..1); PredictResult now
     also carries `touch: number`}}
   - PredictResult / DecodedCommand / RolloutFrame: {{exact field changes}}
   - Attention grid G: {{old}} → {{new}} (the "where the model looks" heatmap is GxG)
   - New model input(s): {{e.g. a task-type flag the rollout must pass to predict}}
   - Run config (⚙ menu): {{e.g. new "touch mode" toggle}} ; ETA: {{changed?}}
   - Colors / palette: {{changed? count?}}
   - Task/behavior: {{e.g. rollout can now succeed on a touch without lifting —
     the rollout panel's phase labels / success copy need updating}}
4. Find every `mini-vla/*` import (grep `from "mini-vla`) and the Hero/rollout/
   language panels; update readouts, overlays, and any hard-coded head list or grid
   size to match the new contract.
5. Verify: typecheck/build, run the page, confirm training → rollout shows the new
   behavior and every panel (attention heatmap, per-token bars, decoded-target,
   gripper/touch state) renders. Then redeploy.
```

## Gotchas (carry over from the JS package — do not break)

- **ESM-only, single instance. Do NOT add a CJS build.** Tokenizer vocab
  (`examples.ts`), run-config, and `CONFIG` are per-thread module singletons; a
  dual build silently breaks the worker→main `registerFullVocab` vocab post-back.
- **Import order in `js/eval/`**: `trainer.core`/`model` snapshot `CONFIG` into
  module constants on first eval, so any `?set=` override must be applied to
  `CONFIG` **before** `trainer.core` is imported (it's dynamic-imported after).
  Mirror of Python's override-before-import contract in `train.py`.
- **Per-thread state**: a new user-selectable knob must be installed on **both**
  threads (main + worker) like `setRunConfig`.
- Embedding rows 0/1 (`<pad>`/`<unk>`) are zero on purpose.
- `paintSilhouette` (model input) only — never regenerate `paintScene` from Python.
- **Silhouette parity is functional, not byte-exact** (the browser retrains).
- Don't "fix" the rollout stepping toward its last target while a predict is in
  flight — it's intentional latency-hiding.
- Keep the gripper's dedicated grasp-class samples + non-zero gripper loss weight,
  and the `block.min` floor — removing either collapses the head (see config notes).
```
