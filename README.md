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

### Serving the assets (`assetBase`)

The browser side `fetch()`es `embeddings-50d.bin` + `vocab.txt` at runtime from
`assetBase`, which defaults to `/vla` — where this repo's own demo, eval and test
pages serve `assets/`, so they need no wiring at all. A host that redeploys often
should serve them from a **version-stamped** directory and say so:

```ts
import { version } from "mini-vla/package.json";
new VLATrainer({ assetBase: `/vla/${version}` });
```

Copy `assets/` into that same `public/vla/<version>/`, deriving both strings from
that one `version` field. A tab left open across a deploy then gets a clean 404 —
surfacing as `status: "error"` with `errorReason: "assets"` — instead of quietly
loading embeddings from a different generation than its JS. `loadEmbeddings()`
also validates the fetched bytes against the constants compiled in from
`vocab.gen.ts`, throwing with expected vs. actual sizes, so a mismatch can never
reach the model as a silently NaN-poisoned table.

(`grammar.json` is a bundle-time import, not a fetch; `assetBase` doesn't apply.)

## Architecture

The policy is a **language-conditioned spatial-attention** network, not a
flatten→dense fusion — trained by **behavioral cloning against an analytical IK
expert**, with no reward or simulator rollout collection. See
[`mini_vla/model.py`](mini_vla/model.py) for the layer-by-layer build.

Three inputs drive one forward pass, re-run closed-loop every few frames during
rollout:

| input | shape | what it's for |
|---|---|---|
| `vision_pixels` | `imgSize × imgSize × 3` | model's-eye silhouette render of the scene |
| `language_tokens` | `MAX_SEQ_LEN` | tokenized command, ids into a frozen pretrained GloVe table |
| `carry_flag` | `1` | is the gripper already holding a block? — disambiguates "go pick" vs. "come home" for the identical (scene, sentence) |

**Language branch.** Token ids look up a **frozen** 50-d GloVe embedding; a
linear `Conv1D` (kernel 7 — each token scored together with its ±3-token
neighborhood) scores every token, and a masked softmax attention-pools the
sequence into a single vector — the language slot the rest of the network
reads. Only the scorer (and the heads downstream) fine-tune; the embedding
table itself never moves, so unseen near-synonyms ride along on the frozen
GloVe geometry. A text-only warm-up phase pretrains this branch (+ an
auxiliary color-classification head) before the vision branch joins the loss.

**Vision branch → spatial attention, not concat-and-flatten.** A small
config-driven conv stack (`mini_vla/config.py`'s `ModelConfig.conv` — 3 conv
layers, relu, 2 max-pools by default) turns the silhouette into a `G×G`
feature map. Vision and language meet through an explicit attention readout:

1. the pooled language vector + `carry_flag` form a query;
2. the query is dot-producted against every cell of the `G×G` map and
   softmaxed into an attention map — "where the model is looking";
3. a **frozen soft-argmax kernel** turns that map into an expected `(x̂, ŷ)`
   image coordinate (gradients still flow through it — only the kernel is
   fixed, seeded post-build from each cell's grid center);
4. the same attention map reads out an attention-weighted feature vector —
   what the map "sees" at that spot (block size, local shape).

**Fusion + heads.** `(x̂, ŷ)` + the attended feature vector + the language
vector + `carry_flag` concatenate into one small dense fusion layer, which
feeds four heads: the target joint angles (Huber regression), a
color-classification head (auxiliary, off the language vector alone), the
attention map itself (auxiliary supervision, categorical cross-entropy against
the commanded block's known grid cell — without it the action loss alone
under-trains the attention), and a sigmoid "close gripper now" command (binary
cross-entropy).

**Training labels come from geometry, not demonstrations.** Each batch
synthesizes random scenes, arm poses, and commands, renders the silhouette
through the same pipeline the live rollout uses, and labels it directly: the
target joint angles come from closed-form 2-link inverse kinematics toward the
commanded block (or home, if `carry_flag` is set), the attention-map label
comes from that block's known grid cell, and the gripper label comes from a
shared geometric "is the effector fully over the block" predicate — the same
predicate the closed-loop rollout and eval use to score a grasp. At rollout
time there's no ground truth: the model predicts, the arm steps proportionally
toward the target, the gripper closes on the learned head's rising edge, and
the loop repeats until it settles.

## Python (develop here)

```bash
uv venv --python 3.12 && uv pip install -e .   # or: pip install -e .
python train.py                                # desktop preset: 8 colors, 4 blocks → converge → grasp-rate
python train.py --preset mobile --wandb        # mobile preset: 4 colors, 3 blocks, track a run
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
npm run test:e2e    # pipeline smoke across the browser×device matrix
npm run perf        # convergence-perf gate (real GPU; batches-to-converge vs budget)
npm run typecheck
```

`npm run perf` (`js/test/perf.spec.ts`) trains each profile to full convergence
and asserts it stays within the batch budget in `js/test/perf-budgets.json` — a
regression guard on the number the < 30 s in-browser budget is built on. It needs
a real GPU and runs off-CI (a full run is ~500 software-WebGL batches otherwise);
tuning a budget or adding a task is a JSON edit.

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
   `npm run eval` (headless grasp metric), `npm run test:e2e` (pipeline smoke);
   after an architecture/config change, `npm run perf` (convergence-perf gate —
   update `js/test/perf-budgets.json` if the batch budget genuinely moved).
4. **Regenerate assets** only if the grammar/vocab/quantization changed:
   `npm run gen:embeddings` → rewrites `assets/` and, in lockstep,
   `js/src/vocab.gen.ts` + `mini_vla/vocab_gen.py`.
5. **Commit + tag** a new version here (e.g. `v0.2.0`).
6. **Go live:** in the portfolio repo bump the `mini-vla` dependency to the new
   tag, reinstall, redeploy. That single step ships it.
