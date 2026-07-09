# mini-vla

A tiny **vision-language-action** policy: a language-conditioned spatial-attention
network learns, by behavioral cloning against an analytical-IK expert, to pick up
a named block on a 2-link arm ("grab the blue cube") — reaching with an open
gripper, closing over the block, and lifting it home.

This repo has two sides:

| | | |
|---|---|---|
| **`mini_vla/`** (+ `train.py`) | **the model — source of truth** | Python / Keras. Where the architecture, config, and wandb experiments are actively developed. |
| **`js/`** | **the output artifact** | A TensorFlow.js re-expression of the same architecture + task that trains *live in the browser*, embedded on the portfolio. Regenerated **from** the Python. |
| **`assets/`** | **shared, framework-neutral** | Pretrained int8 GloVe table (`embeddings-50d.bin`, `vocab.txt`) + the slot-grammar word inventory (`grammar.json`). Read by both sides. |

The two sides are kept in **architecture + task parity, not weight parity**: the
browser demo retrains from scratch (that's its whole point), so `js/` mirrors the
network shape, the task, the config, and the silhouette render — never a trained
checkpoint.

## Python (develop here)

```bash
uv venv --python 3.12 && uv pip install -e .   # or: pip install -e .
python train.py                                # 8 colors, 4 blocks → converge → grasp-rate
python train.py --colors 4 --blocks 3 --wandb  # track a run
python train.py --set model.mapLossWeight:2.5 model.learningRate:0.003 --probe 25
```

`train.py` builds the model, warms up the language head, trains to convergence
(or `--max-batches`), scores the policy closed-loop (grasp rate / jitter), and
saves weights. `--wandb` logs the loss curve, per-phase probe buckets, and the
final grasp rate. See `mini_vla/config.py` — the single knob sheet — for every
tunable and its rationale.

### Module map (`mini_vla/` ↔ `js/src/`)

| Python | mirrors | what it is |
|--------|---------|-----------|
| `config.py` / `run_config.py` | `config.ts` / `run-config.ts` | the knob sheet + user palette/density |
| `geometry.py` | `geometry.ts` | 2-link arm FK/IK, the shared grasp predicate |
| `task.py` | `examples.ts` | slot grammar, tokenizer, scene layouts, colors |
| `render.py` | `scene.ts` (`paintSilhouette`) | the model's-eye input rasterizer |
| `embeddings.py` | `embeddings.ts` | frozen GloVe table loader |
| `model.py` | `model.ts` | the Keras policy (+ viz / lang twins) |
| `trainer.py` | `trainer.core.ts` | batch synthesis + training loop |
| `eval.py` | `eval/main.ts` (`closedLoopEval`) | closed-loop grasp-rate metric |

The JS-only pieces have **no Python counterpart** — they are portfolio-render
concerns, not model logic: `scene.ts` `paintScene` (styled display), `rollout.ts`
(animation engine), `demo.ts` (scripted demonstration), `trainer.ts` /
`trainer.worker.ts` (main-thread worker proxy), and the demo page UI.

## JS (the in-browser artifact)

Runs from the repo root against the root `package.json` (the paths point into
`js/`). See [`js/README.md`](js/README.md) for the full package/architecture doc.

```bash
npm install
npm run demo        # Vite page: watch it train + roll out
npm run eval        # headless grasp-rate sweep (needs a Chromium)
npm run typecheck
```

The portfolio consumes this as a git-ref npm dependency
(`github:…/mini-vla#<tag>`) through the `exports` map in the root `package.json`
— it only ever sees the `mini-vla/trainer`, `mini-vla/rollout`, … subpaths, never
the internal file layout.

## Development → live workflow

1. **Develop in Python.** Iterate architecture/config in `mini_vla/`, run
   `python train.py` (+`--wandb`), read the grasp rate from `eval.py`.
2. **Port to JS** with the Python→JS port skill: it re-expresses `mini_vla/*` →
   `js/src/*` (architecture + task + config + silhouette render; **no weights**).
3. **Verify JS:** `npm run typecheck`, `npm run demo` (watch train→rollout),
   `npm run eval` (headless grasp metric).
4. **Regenerate assets** only if the grammar/vocab/quantization changed:
   `npm run gen:embeddings` → rewrites `assets/` and, in lockstep,
   `js/src/vocab.gen.ts` + `mini_vla/vocab_gen.py`.
5. **Commit + tag** a new version here (e.g. `v0.2.0`).
6. **Go live:** in the portfolio repo bump the `mini-vla` dependency to the new
   tag, reinstall, redeploy. That single step ships it.
