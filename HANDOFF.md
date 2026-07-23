# Handoff — mini-vla / fresh-review-2

> The baton. Present tense only: where the work stands *now* and what the
> next session should do. Overwrite it each session — never append. History
> lives in git and rationale in commit bodies, not here. Before the task
> ends, promote anything durable (project status, repo instructions, commit
> body) and delete this file from the branch: a finished task hands
> nothing off, and merged, a leftover baton strays onto the default branch.

- **Repo:** mini-vla
- **Branch:** `fresh-review-2`
- **Worktree:** /root/git/worktrees/mini-vla/fresh-review-2
- **Last updated:** 2026-07-23 18:34 UTC · server (srv1841294)

## State

Not started. This is review round 2 — round 1 (branch `fresh-review-1`, PR
#22) merged into `main` earlier today. Round 1 ran the full generic-core
catalog thoroughly (5 parallel discovery-dimension agents + a
`codebase-health` scan-only leg) against the pre-round-1 code and fixed what
the owner approved: CI authorization gaps (`claude-review`'s trigger,
`dispatch-portfolio`'s tag/version bypass + unescaped JSON payload), the
`.githooks/pre-push` stale-default-branch-cache bug, `train.py`'s
print-only budget check (now enforced, and wired into `check-python`), the
E2E matrix's iOS/WebKit coverage gap for `replay.spec.ts`, zero test coverage
on `geometry.py`/`eval.py`, a stale cross-file rationale comment
(`colorLossWeight`), and mechanical lint. The owner separately had
server-side branch protection enabled on `main` (required check: `e2e`, no
force-push, no deletion) — a live GitHub setting with **no git-diff
footprint at all**.

Given that coverage, this round should **not** re-sweep the pre-round-1 code
broadly across the generic-core catalog — treat `docs-vs-code drift`,
`CI correctness and guard coverage`, and `allowlist escapes` as well-covered
there and deprioritize a full re-read. Weight the round instead toward:

1. **Round 1's own diff is unreviewed by a fresh session.** Round 1 verified
   its own fixes in the same session that wrote them — exactly the
   "agent grading its own homework" problem `codebase-review`'s design note
   exists to avoid. Adversarially re-check these files as new code, not as
   already-settled: `.githooks/pre-push` (the new live `git ls-remote`
   default-branch resolution — does it handle a remote with no HEAD symref
   at all, or a remote name containing shell-special characters, correctly?
   does `[[:blank:]]` in the sed pattern actually match what a real
   `git ls-remote --symref` emits across git versions?), `.github/workflows/
   e2e.yml` (the `author_association` allowlist — does `contains(fromJSON(...))`
   actually reject a missing/null `author_association` the way intended, or
   could an edge-case event shape read as truthy?; the `jq -r .version
   package.json` tag-match step — what if `package.json` is missing or
   malformed at that commit?), `train.py` (the `sys.exit(1)` placement —
   confirm no other caller now silently swallows a non-zero exit it didn't
   used to see), and the new tests themselves (`tests/test_geometry.py`,
   `tests/test_eval.py`, the `replay.spec.ts` `webkit-iphone` extension) —
   do they actually assert what their names claim, or could they pass
   vacuously on a regression they're meant to catch? Diff to read:
   `git log --stat 7d7ebcf~10..7d7ebcf` (round 1's 10 commits, merged as
   `7d7ebcf`).
2. **Audit the live branch-protection setting**, since it has zero footprint
   in git: `gh api repos/lukasmueller-dev/mini-vla/branches/main/protection`
   — confirm the required check is still named exactly `e2e` (matching the
   job id in `e2e.yml`; a rename of that job would silently orphan the
   required check with no CI failure to notice), confirm force-push and
   branch deletion are still blocked, and confirm `enforce_admins` is still
   the intended value (currently `false` — the owner keeps an override).
3. **Give real depth to "hardcoded defaults in emitted artifacts"** — round
   1 only lightly checked `dispatch-portfolio`'s payload/target repo under
   this dimension. Broaden it: what does `.claude/skills/port-to-js/SKILL.md`
   emit into `js/src/`, and does anything it writes assume a path, branch
   name, or repo layout that could differ from the authoring context? What
   do `scripts/release.mjs` and `js/scripts/gen-embeddings-data.mjs` stamp
   into generated output, and is any of it validated-but-unused the way
   round 1's `train.py` budget check turned out to be, before that fix?
4. **A candidate new review-dimension class round 1 surfaced but never
   added to the catalog**: *sequential-guard gap auditing* — checking not
   just whether one guard has blind spots, but whether it is the **only**
   path to the thing it gates (`scripts/release.mjs` was airtight alone,
   but was optional before round 1's fix — a raw `git tag && git push`
   reached `dispatch-portfolio` without it). Apply this lens this round: are
   there other guards in this repo whose invariant can be reached by a path
   that skips them entirely? If it holds up, name it again in this round's
   output so it's a candidate for `references/review-dimensions.md`
   (a separate toolkit-repo file, not edited as part of staging this round).
5. **Known open item, carried over, not obviously fixable**:
   `js/test/perf.spec.ts` — the actual <60s in-browser convergence budget —
   still has zero CI coverage. It's GPU-only and excluded from CI
   (`testIgnore` in `playwright.config.ts`); round 1 added Python-side
   budget enforcement (`check-python` now runs `train.py` for both presets)
   as a partial substitute, but a JS-only architecture change could still
   drift from the Python calibration undetected. Take a fresh look at
   whether there's a cheap, safe partial gate (e.g. a software-GL smoke
   bound distinct from the full GPU perf run) — but don't force a fix if
   none is safe/cheap; it's fine to re-confirm this as still open.

Still run the full mission shape regardless of the above weighting:

1. **Discovery pass, blind** — work the generic-core dimensions (adversarial
   static read, docs-vs-code drift, test blind spots, config/permission
   safety, CI correctness, cross-file consistency, derived-but-unused
   values, guard-pattern blind spots, allowlist escapes, hardcoded
   defaults) before reading anything about round 1's findings — weighted
   per above, not skipped. This repo is not the toolkit that authors this
   skill, so no conditional dimensions apply; extend scope from this repo's
   own `CLAUDE.md` and `README.md` as round 1 did.
2. **`codebase-health` scan-only leg**, after the discovery pass. Scan mode
   only: stop at its report, no approval step, no `health/*` branches or
   PRs. Note which language reference it used (round 1 used `python.md`,
   `typescript.md`, `shell.md` — all still exist). Fold survivors into the
   ranked list, de-duplicated against discovery.
3. **De-duplicate** against round 1 only now: `gh pr view 22` for the full
   list of what it fixed (this file's summary above is a start, not a
   substitute for reading the actual PR). A re-found item round 1 already
   fixed is replication — report it separately as evidence the discovery
   pass works, never as a new finding. There is still no `PROJECT_STATUS.md`
   / `PROJECT_ROADMAP.md` and still no in-repo TODO/FIXME markers as of
   staging, so the debt-track dedup set is otherwise empty — note this
   rather than skipping the step silently.

## Next action

1. `git rebase origin/main` (never the resume verb) — this worktree was
   staged directly off the current `main` tip (`7d7ebcf`, includes round
   1), so this should be a no-op unless something else has landed since;
   confirm rather than assume.
2. Run this repo's own verification gate as the baseline before reviewing:
   `npm ci && npm run typecheck`, `pip install -e ".[dev]" && pytest`, and
   the Playwright E2E suite (`npm run test:e2e` — expect it to need browser
   binaries; `npx playwright install --with-deps <browser>` first if so;
   round 1 verified `chromium-desktop` and `webkit-iphone` both green as of
   the merge). A failure here is the round's to fix or flag, not to review
   around.
3. After the rebase and baseline are green (or their failures logged),
   dispatch parallel sub-reviewers over the discovery-pass dimensions —
   independent dimensions, so fan them out rather than working the list
   serially. Weight fan-out time toward round 1's diff (item 1 above) and
   the live branch-protection audit (item 2), since those have the least
   existing coverage.

## Blockers

None yet — nothing has been attempted.

## Gotchas (unpromoted)

- This repo's CI is not opt-in per PR (unlike the toolkit repo this skill
  ships from) — `e2e.yml` triggers automatically on `push`/`pull_request`,
  so this round's own PR needs no opt-in label to get a CI run.
- **Discovered while staging this round, not yet promoted anywhere**: in a
  network-restricted environment, this repo's `origin` remote has an SSH
  fetch URL with no key configured, so a bare `git fetch` (or `git -C
  <main_repo_root> fetch` inside `vibe-lib.sh`'s `ff_main_quiet`) fails
  silently — `ff_main_quiet` swallows that failure by design (never blocks
  branching on a failed fetch), which left the primary clone's local `main`
  stale at the pre-round-1 commit. `review.sh create` initially staged this
  very round against that stale `main` (missing all of round 1's fixes)
  before the mistake was caught and corrected by hand: `git fetch
  <https-clone-url> main:refs/remotes/origin/main --force`, then fast-
  forwarding the primary clone's checked-out `main` before re-staging. If a
  future round in a similarly sandboxed environment sees a newly-staged
  round branch that doesn't contain the previous round's merge commit,
  check for exactly this before assuming `review.sh` is broken.
- Deliverable shape: a ranked findings list, each with `file:line` and a
  concrete failure scenario; label reasoned-but-unproven claims as such.
  Fixes never replace the report — this round produces a report, not
  patches, except where the owner explicitly asks (as happened for most of
  round 1's findings).
- One commit per concern if any fixes are made at all — the failure it
  prevents goes in the commit body so any single fix reverts alone.
- Get the repo owner's explicit go-ahead before any high-blast-radius
  fix — installer/uninstall safety, destructive git paths, permission or
  settings changes (branch protection now counts as exactly this class,
  and is already owner-approved and live — round 2 should audit it, not
  re-toggle it without a specific reason).
- End the round's output by naming any new review-dimension class its
  findings imply (see the "sequential-guard gap auditing" candidate above),
  so it can be added to `references/review-dimensions.md` in the
  `codebase-review` skill's toolkit repo (or, if the class is specific to
  mini-vla's Python↔JS port shape, to this repo's own `CLAUDE.md` instead).
