# mini-vla — agent working notes

Two-sided repo: **`mini_vla/`** (Python/Keras) is the model **source of truth**;
**`js/`** (TensorFlow.js) is the **ported output artifact** that trains *live in the
browser* and is embedded on the portfolio. `assets/` is shared. See
[README.md](README.md) for the full layout and the Python→JS workflow.

## ⛔ Standing constraints for any architecture / model change

These bound **every** decision when editing `mini_vla/model.py`, `config.py`,
`trainer.py`, `task.py`, or `geometry.py`. Read them before proposing a change.

### 1. The real deliverable is the ported **browser (TF.js)** model — not the Python

You develop in Python, but the change only counts once it ships to `js/src/` via
the **`/port-to-js`** skill and runs in the browser. So the Python must stay
**portable to `tf.layers` (tfjs-layers)**:

- Use **only layers/ops with a tfjs-layers equivalent** and a WebGL-friendly
  shape. The allowed set is the Keras→tfjs table in `.claude/skills/port-to-js/SKILL.md`:
  `conv2d, conv1d, dense, embedding, maxPooling2d, reshape, dot, activation,
  concatenate` + the one custom `AttentionPooling` layer. **No** exotic Keras
  layers, custom gradients, `tf.function`-only tricks, or ops that exist only in
  Python TF — they will not port.
- Keep the model **small and single-instance** (ESM singleton; frozen 20k×50
  embedding is the only big tensor — don't add more that would blow WebGL memory).
- Any new **input / head / knob** must be portable, and if user-visible must work
  on both the worker and main threads (per-thread module state). If it changes the
  exported contract (`PredictResult` / `DecodedCommand` / `RolloutFrame`, the head
  set, `ATTN_GRID`/G, model inputs), it also needs **portfolio** changes — the
  `/port-to-js` skill's Phase 7 handles that.

### 2. It must train-to-converge **+ roll out in < 30 s, live in the browser**

This is a hard **product** requirement — it's a portfolio demo a visitor watches.
It bounds `imgSize`, the conv stack, `batchSize`, warm-up, and above all
**batches-to-convergence**.

- Calibration: the browser does **~10 gradient batches/s** (WebGL, mid laptop GPU;
  see the `converge`/`eta` notes in `js/src/config.ts`). Budget ≈
  `~2 s load + batches_to_converge / 10`. To stay under 30 s, keep typical
  convergence **under ~250–280 batches** at the current per-batch cost.
- `python train.py` **prints the projected browser time** every run
  (`[budget] … est. browser train ≈ Ns`) — watch it; treat "OVER BUDGET" as a
  regression to fix, not a warning to ignore.
- **Pay-as-you-go:** if you make the net heavier (bigger `imgSize`, deeper conv,
  larger `batchSize`), you must *buy it back* — faster/fewer batches to converge —
  or you break the budget. Note the default `converge.maxBatches` fallback (800 ≈
  80 s) already exceeds budget; healthy runs converge well before it, so don't let
  the *typical* path drift toward the fallback.

## After Python architecture work

Run **`/port-to-js`** to re-express the change in `js/src/`, verify
(`npm run typecheck` / `demo` / `eval`), and get the go-live + portfolio hand-off.
Do not hand-edit `js/src/*` as the primary path — it's regenerated from Python.
