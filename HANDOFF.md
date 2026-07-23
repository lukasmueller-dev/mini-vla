# Handoff — mini-vla / color-convergence-check

> The baton. Present tense only: where the work stands *now* and what the
> next session should do. Overwrite it each session — never append. History
> lives in git and rationale in commit bodies, not here. Before the task
> ends, promote anything durable (project status, repo instructions, commit
> body) and delete this file from the branch: a finished task hands
> nothing off, and merged, a leftover baton strays onto the default branch.

- **Repo:** mini-vla
- **Branch:** `color-convergence-check`
- **Worktree:** /root/git/worktrees/mini-vla/color-convergence-check
- **Last updated:** 2026-07-23 23:43 UTC · server (srv1841294)

## State

- portfolio-site's `bump-mini-vla.yml` (hardened today in review-round-1, PR
  #57) ran `hero-full.spec.ts`'s train-to-Ready-then-decode-command test for
  the first time ever, against the already-shipped v0.7.1. It fails
  consistently: the model reaches "Ready" but decodes the wrong color (asked
  "red", got "black"/"yellow") — across desktop/mobile/webkit-mobile, on
  both the attempt and the retry.
- Root cause identified: `js/src/trainer.core.ts`'s convergence gate (~line
  1049, `this.smoothLoss < CONVERGE_LOSS`) only tracks the **action (motor)
  loss** — output index 1 of the model's multi-output loss array. The
  color/language loss (index 2, weighted `colorLossWeight = 0.6` in both
  `js/src/config.ts` and `mini_vla/config.py`) is never part of the "Ready"
  decision. Training can hit `MAX_BATCHES` (800) or the action-loss
  threshold while the color head is still undertrained, and "Ready" fires
  anyway.
- This gap predates v0.7.1 — the v0.7.0→v0.7.1 diff was checked and is
  behavior-preserving refactors/comment fixes only. It's surfacing now only
  because this is the first time `hero-full.spec.ts` (needs `VLA_FULL=1`)
  has ever run in any CI gate, for any mini-vla version.
- Separate, unrelated bug flagged in the same run: v0.7.1's `package.json`
  version still reads "0.7.0" (a mini-vla release-script bug). Already
  tracked and patched going forward (`scripts/release.mjs` + `e2e.yml`'s
  tag-check in this repo) — only affects the served asset path label, not
  decode correctness. Don't conflate it with this task.
- `mini_vla/` (Python) is the source of truth; `js/src/` is the ported
  artifact — any convergence-gate change must land in
  `mini_vla/trainer.py`/`config.py` first, then re-port via `/port-to-js`.

## Next action

Design and implement a convergence check that also requires the
color/language loss to be low before declaring "Ready" — not just the
action loss. Concretely: add a color-loss threshold (mirroring
`ConvergeConfig`'s loss/window/streak shape) in `mini_vla/config.py`, apply
it in `mini_vla/trainer.py`'s train loop, then port to `js/src/config.ts` +
`js/src/trainer.core.ts`'s convergence block (~line 1049) so
`convergeStreak` only increments when both action and color smoothed losses
are under threshold. Must stay within the <60s browser budget (CLAUDE.md) —
check `python train.py`'s `[budget]` output, retune `colorLossWeight` or the
threshold if it blows the budget. Verify against portfolio-site's
`tests/e2e/hero-full.spec.ts` (`VLA_FULL=1`) — that's the reproduction and
the pass bar (8/8 colors decode correctly).

## Blockers

None known yet.

## Root-cause evidence (from CI artifacts)

Pulled `playwright-report` from run
https://github.com/lukasmueller-dev/portfolio-site/actions/runs/30037663156
and read the four failures' `error-context.md` snapshots. Each shows the HUD
batch count at "Ready":

| Browser | Batches at Ready | Decoded (wrong) |
|---|---|---|
| desktop | 269 | black |
| desktop retry | 273 | black |
| mobile | 243 | yellow |
| mobile retry | 239 | yellow |

All four cluster at 239-273 batches — well below BOTH calibrations on
record: `mini_vla/config.py`'s `ConvergeConfig` comment (desktop ~415-618,
mobile ~283-404, by-loss, "9 seeds, 0 hit the fallback") and
`hero-full.spec.ts`'s own comment ("converges around 370-560 batches" on
this exact headless-SwiftShader setup). Consistent, not noisy — action-loss
convergence is firing well before the color head has had the batch count it
was calibrated against. This is the actual mechanism behind the wrong
decodes, on top of the structural gap (color loss never gates "Ready").

Note: this repo has a recurring pattern of stale calibration comments (see
`73cecfd`, `24fd44e`) — the "370-560" figure may simply never have been
re-measured after the Huber→circular-MSE action-loss switch, meaning
~250-batch convergence could be the current *actual* behavior, not a fresh
regression. Either way: **size the color-convergence threshold/streak
against a ~250-batch action-convergence point**, not the ~400-600 range the
old comments assume — re-measure with `python train.py` / a fresh
`perf-nightly`-style run rather than trusting the existing comments.

## Gotchas (unpromoted)

- `MAX_BATCHES` (800) is a hard fallback that calls "Ready" regardless of
  loss quality — decide whether color-convergence needs its own
  fallback/cap or rides the same ceiling.
- `hero-full.spec.ts` lives only in portfolio-site (behind `VLA_FULL=1`) —
  no equivalent test exists in mini-vla's own `js/test/` suite, so this repo
  currently can't catch this regression on its own. Worth adding one here
  too.
