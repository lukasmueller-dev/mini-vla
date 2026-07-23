# Handoff — mini-vla / fresh-review-3

> The baton. Present tense only: where the work stands *now* and what the
> next session should do. Overwrite it each session — never append. History
> lives in git and rationale in commit bodies, not here. Before the task
> ends, promote anything durable (project status, repo instructions, commit
> body) and delete this file from the branch: a finished task hands
> nothing off, and merged, a leftover baton strays onto the default branch.

- **Repo:** mini-vla
- **Branch:** `fresh-review-3`
- **Worktree:** /root/git/worktrees/mini-vla/fresh-review-3
- **Last updated:** 2026-07-23 · server (srv1841294)

## State

Not started. This is review round 3 — round 1 (PR #22) and round 2 (PR #23)
are both merged into `main`. Round 1 ran the full generic-core catalog against
the pre-round-1 code (5 discovery agents + a `codebase-health` scan-only leg)
and fixed CI authorization gaps, the pre-push hook's stale-cache bug,
`train.py`'s budget-check enforcement, an E2E coverage gap, and zero test
coverage on `geometry.py`/`eval.py`. Round 2 adversarially re-reviewed round
1's own diff (finding it held up), dug deeper into hardcoded defaults in
emitted artifacts, and applied the "sequential-guard gap auditing" dimension
round 1 first named — finding and fixing two more real instances: `scripts/
release.mjs` had no branch/provenance check (a raw feature-branch release
could pass every other check), and `.claude/skills/port-to-js/SKILL.md`'s
Phase 6 literally instructed a raw `git tag` + push that bypassed
`release.mjs` entirely (confirmed not theoretical — tag `v0.7.1`'s
`package.json` still reads `0.7.0`). Round 2 also fixed a stale budget number
in that same skill file, reconciled `pyproject.toml`'s version, ported two
config fields that had drifted from `mini_vla/` to a hardcoded JS literal,
added an asset-shape guard to the Python embeddings loader, added four new
test files (`test_embeddings.py`, `test_task.py`, `test_trainer.py`,
`test_vocab_consistency.py`), and added a new non-blocking weekly CI job
(`playwright.perf-ci.config.ts` + `.github/workflows/perf-nightly.yml`)
running the real convergence-budget spec under software-GL.

One round-2 finding could **not** be fixed by that session and is still
open, confirmed live moments before this brief was staged (`gh api repos/
lukasmueller-dev/mini-vla/branches/main/protection`): `main`'s branch
protection still has no `required_pull_request_reviews` at all, meaning
nothing GitHub-side blocks a direct push to `main` — required status checks
only gate the PR-merge button. This needs the repo owner to act directly
(a live settings change, correctly out of reach for an agent); **re-check
this first, since it may have been resolved since this brief was staged.**

Given that coverage, this round should **not** re-sweep the generic-core
catalog broadly against code both prior rounds already read closely — treat
`docs-vs-code drift`, `derived-but-unused values`, `allowlist escapes`, and a
broad `adversarial static read` of the core model/training files
(`mini_vla/model.py`, `trainer.py`, `js/src/model.ts`, `trainer.core.ts`,
`render.py`, `scene.ts`) as reasonably covered across two rounds now, and
deprioritize a full re-read of those specifically. Weight the round instead
toward:

1. **Round 2's own diff is unreviewed by a fresh session** — the same
   "agent grading its own homework" concern round 2 applied to round 1.
   Adversarially re-check, as new code, not as already-settled:
   `scripts/release.mjs`'s new provenance check (does `git rev-parse
   --abbrev-ref HEAD` behave sensibly from a detached HEAD, e.g. exactly
   the state a CI checkout leaves a runner in — does that make the release
   script unusable from CI even though it's meant for local/manual use? does
   a `git fetch origin main --quiet` failure — network blip, revoked
   credential — fail the release closed, the way it's meant to, or could it
   silently proceed past a stale comparison?); the four new test files
   (`test_embeddings.py`, `test_task.py`, `test_trainer.py`,
   `test_vocab_consistency.py`) — do they actually assert what their names
   claim, or could any pass vacuously on a regression they're meant to
   catch (round 2 verified this for one test by injecting and reverting a
   real bug — spot-check at least one more the same way); and the new
   `playwright.perf-ci.config.ts` / `perf-nightly.yml` pair, which round 2
   only partially smoke-tested (booted the harness, confirmed a test was
   discovered, never ran to completion) and which has **zero actual runs
   since merge** (`gh run list --workflow=perf-nightly.yml` — empty as of
   staging) — is it actually wired correctly, or does the cron/dispatch path
   have a bug nobody has seen fire yet? Diff to read: `git log --stat
   origin/fresh-review-1..origin/fresh-review-2` (round 2's 9 commits,
   merged as PR #23).
2. **Re-audit the live branch-protection setting** (see State above) — if
   still unresolved, re-report it plainly rather than re-deriving it from
   scratch; if resolved, verify the new configuration doesn't accidentally
   block the owner's own solo workflow (e.g. a required-approvals count
   that can't be satisfied without a second collaborator) and report the
   discrepancy if so.
3. **Named but not yet closed, from round 2's PR "known, not addressed"
   list**: zero unit coverage on the model/JS-port boundary beyond what
   round 2 added — `mini_vla/model.py`/`js/src/model.ts` and
   `mini_vla/config.py`/`js/src/config.ts` still rely on E2E-level checks
   for cross-language agreement, not fast unit tests. Give this real depth:
   what specific claims (shapes, the `AttentionPooling` masking math, the
   circular-coordinate encoding/decoding round-trip) are only ever checked
   by a full browser training run, and would a fast unit test on either
   side actually be feasible without a full TF/tfjs forward pass?
4. **Validate (or refute) the "sequential-guard gap auditing" dimension a
   third time.** It's been named by round 1 and confirmed by round 2 with
   two more real instances. Apply it once more, this round, to whatever
   guards exist post-round-2 (the new provenance check in `release.mjs` is
   itself a guard — does anything else reach `dispatch-portfolio` or a
   release artifact around it?). If it holds up again, say so explicitly in
   this round's output — three confirmed instances across two independent
   repos-worth of review effort is a strong signal it belongs in the
   toolkit's `references/review-dimensions.md` as a generic dimension (that
   file, not this repo, is where the promotion actually happens — this
   round's job is to supply the evidence, not edit that file).
5. **Known open item, carried over, likely still not fully closable**: the
   real <60s in-browser convergence budget (`js/test/perf.spec.ts`, GPU-only)
   still has no *required* CI gate — round 2's new weekly software-GL job is
   a non-blocking proxy, not a substitute. Re-confirm this framing still
   holds (round 2's own comments already say so) rather than re-litigating
   it; the more valuable check this round is item 1's "does the new proxy
   job actually work" question, not re-deciding the tradeoff.

Still run the full mission shape regardless of the above weighting:

1. **Discovery pass, blind** — work the generic-core dimensions (adversarial
   static read, docs-vs-code drift, test blind spots, config/permission
   safety, CI correctness, cross-file consistency, derived-but-unused
   values, guard-pattern blind spots, allowlist escapes, hardcoded
   defaults) before reading anything about prior rounds' findings —
   weighted per above, not skipped. This repo is not the toolkit that
   authors this skill, so no conditional dimensions apply; extend scope
   from this repo's own `CLAUDE.md` and `README.md` as prior rounds did.
2. **`codebase-health` scan-only leg**, after the discovery pass. Scan mode
   only: stop at its report, no approval step, no `health/*` branches or
   PRs. Note which language reference it used (prior rounds used
   `python.md`, `typescript.md`, `shell.md` — all still exist). Fold
   survivors into the ranked list, de-duplicated against discovery.
3. **De-duplicate** against both prior rounds only now: `gh pr view 22` and
   `gh pr view 23` for the full list of what each fixed (this file's
   summary above is a start, not a substitute for reading the actual PRs).
   A re-found item either round already fixed is replication — report it
   separately as evidence the discovery pass works, never as a new finding.
   There is still no `PROJECT_STATUS.md` / `PROJECT_ROADMAP.md` and still no
   in-repo TODO/FIXME markers as of staging, so the debt-track dedup set is
   otherwise empty — note this rather than skipping the step silently.

## Next action

1. `git rebase origin/main` (never the resume verb) — this worktree was
   staged directly off the current `main` tip (which includes both prior
   rounds), so this should be a no-op unless something else has landed
   since; confirm rather than assume. (This repo's `origin` remote resolves
   over SSH with no key configured in some sandboxes, which makes a bare
   `git fetch` fail silently — if `origin/main` looks stale, fetch via the
   HTTPS clone URL instead: `git fetch https://github.com/
   lukasmueller-dev/mini-vla.git main:refs/remotes/origin/main --force`.)
2. Run this repo's own verification gate as the baseline before reviewing:
   `npm ci && npm run typecheck`, `pip install -e ".[dev]" && pytest`, and
   the Playwright E2E suite (`npm run test:e2e` — expect it to need browser
   binaries; `npx playwright install --with-deps <browser>` first if so;
   round 2 verified `chromium-desktop` and `webkit-iphone` both green as of
   its merge, modulo two known resource-contention flakes in constrained
   sandboxes that pass on an isolated retry — don't mistake that class of
   flake for a regression, but don't wave away a *new* failure either). A
   failure here is the round's to fix or flag, not to review around.
3. After the rebase and baseline are green (or their failures logged),
   dispatch parallel sub-reviewers over the discovery-pass dimensions —
   independent dimensions, so fan them out rather than working the list
   serially. Weight fan-out time toward round 2's diff (item 1 above), the
   live branch-protection re-audit (item 2), and the model/config unit-test
   depth question (item 3), since those have the least existing coverage.

## Blockers

None yet — nothing has been attempted.

## Gotchas (unpromoted)

- This repo's CI is not opt-in per PR (unlike the toolkit repo this skill
  ships from) — `e2e.yml` triggers automatically on `push`/`pull_request`,
  so this round's own PR needs no opt-in label to get a CI run.
- In a network-restricted environment, this repo's `origin` remote has an
  SSH fetch URL with no key configured, so a bare `git fetch` (run against
  the primary clone that backs this worktree, inside `vibe-lib.sh`'s
  `ff_main_quiet`) fails silently. This bit staging *both* prior rounds'
  review sessions in some form — most recently, this round was initially
  mis-staged against a stale `main` (missing round 2's merge) before being
  caught and corrected by fetching via the HTTPS clone URL and
  fast-forwarding the primary clone's checked-out `main`. If a newly-staged
  round branch doesn't contain the previous round's merge commit, check for
  exactly this before assuming `review.sh` is broken.
- Deliverable shape: a ranked findings list, each with `file:line` and a
  concrete failure scenario; label reasoned-but-unproven claims as such.
  Fixes never replace the report — this round produces a report, not
  patches, except where the owner explicitly asks (as happened for most of
  both prior rounds' findings).
- One commit per concern if any fixes are made at all — the failure it
  prevents goes in the commit body so any single fix reverts alone.
- Get the repo owner's explicit go-ahead before any high-blast-radius
  fix — installer/uninstall safety, destructive git paths, permission or
  settings changes. Branch protection is exactly this class: round 2
  confirmed a real gap (item 2 above) but could not apply the fix itself
  (an agent's `gh api` write was correctly blocked by the permission
  classifier) — this round should re-audit, not attempt the same write
  again expecting a different result, unless the owner has given a new,
  specific go-ahead in this round's own conversation.
- End the round's output by naming any new review-dimension class its
  findings imply, and explicitly stating whether "sequential-guard gap
  auditing" (item 4 above) is now proven enough, in this round's judgment,
  to promote to `references/review-dimensions.md` in the `codebase-review`
  skill's toolkit repo (a separate repo, not edited as part of staging or
  running this round).
