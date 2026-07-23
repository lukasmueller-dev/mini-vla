# Handoff — mini-vla / fresh-review-1

> The baton. Present tense only: where the work stands *now* and what the
> next session should do. Overwrite it each session — never append. History
> lives in git and rationale in commit bodies, not here. Before the task
> ends, promote anything durable (project status, repo instructions, commit
> body) and delete this file from the branch: a finished task hands
> nothing off, and merged, a leftover baton strays onto the default branch.

- **Repo:** mini-vla
- **Branch:** `fresh-review-1`
- **Worktree:** /root/git/worktrees/mini-vla/fresh-review-1
- **Last updated:** 2026-07-23 15:25 UTC · server (srv1841294)

## State

Not started. This is review round 1 for mini-vla — no prior `fresh-review-*`
round exists, so there is no coverage set to skip and no changed-since-last-round
diff to fold in. The scope below is the full generic-core catalog.

This repo is **not** the toolkit that authors the `codebase-review` skill
(mini-vla is a two-sided model repo: `mini_vla/` Python source of truth,
`js/` the ported TensorFlow.js browser artifact, `assets/` shared). So the
scope is the generic core only — no toolkit-specific conditional dimensions
(skill-quality-beyond-lint, end-to-end lifecycle execution) apply here, since
their anchor files (a skill-quality doc, a `skill-lint` binary, `bin/vibe`,
an `install.sh`) don't exist in this repo. Extend the scope instead from this
repo's own instruction file (`CLAUDE.md`) and `README.md` — read both before
starting. `CLAUDE.md` in particular documents hard product constraints (the
Python model must stay portable to `tf.layers`/tfjs-layers with only a fixed
allowed op set; the browser training loop has a hard sub-60-second budget;
any architecture change must be re-expressed in `js/src/` via the
`/port-to-js` skill) — treat those as the documented pitfalls the
adversarial-static-read dimension hunts missed instances of.

Mission — run each of these across the repo, in this order:

1. **Discovery pass, blind.** Work through the generic-core dimensions below
   before reading anything about prior findings or debt tracks:
   - Adversarial static read against this repo's own documented pitfalls
     (`CLAUDE.md`'s standing constraints above, and any pitfalls `README.md`
     documents) — assume each has at least one instance the guards missed.
     Silent-failure paths first: the bug that prints success. Give particular
     attention to the Python→JS port boundary (does `mini_vla/model.py` truly
     stay within the allowed tfjs-layers op set; does the train.py budget
     print reflect what actually ships) and to the `.githooks/pre-push` /
     `.github/workflows/e2e.yml` guard surface.
   - Docs-vs-code drift: `README.md`, `CLAUDE.md`, inline comments describing
     a sibling file, config examples — do they match the code as it is
     today? The upcoming `codebase-health` leg's docs category sweeps
     known surfaces; this is the judgment half beyond that.
   - Test blind spots and harness isolation: guards without tests, paths only
     one platform exercises (note the E2E matrix spans five browser/device
     projects — are they all pulling their weight, and is anything only
     exercised by one), assertions that pass vacuously, whether the suite
     can touch a real environment or only mocks one.
   - Config and permission safety: audit `.claude/settings.json` (if any),
     `.githooks/pre-push`, and CI permission blocks in `e2e.yml` (note the
     `claude-review` job's `id-token: write` and the cross-repo dispatch
     job's PAT) against their documented traps — wildcards wider than
     intended, a deny rule with an equivalent spelling it misses.
   - CI correctness and guard coverage: does `e2e.yml` run what it claims,
     on the platforms that matter; does the `e2e` aggregate job's `if:
     always()` and per-shard `needs` actually fail closed if a shard
     errors before producing a result; is the `claude-review` job's
     `issue_comment` trigger scoped so an arbitrary commenter can't drive it.
   - Cross-file consistency: constants, regexes, or contracts duplicated
     across `mini_vla/` and `js/src/` and kept in lockstep only by comment
     (the allowed-op-set table in the `port-to-js` skill, the embedding
     dimensions, `ATTN_GRID`/G, the budget constants in `js/src/config.ts`
     vs. the Python-side calibration notes) — verify every copy actually
     matches, not just that a pointer comment names the sibling.
   - Derived-but-unused values: a script that computes and logs a value but
     never acts on it (check `train.py`'s budget projection actually gates
     something vs. only printing; check `scripts/release.mjs` and
     `js/scripts/gen-embeddings-data.mjs` for resolved paths or flags that
     are validated and named but never read).
   - Guard patterns with blind spots: the `.githooks/pre-push` hook and any
     grep/regex-based guard in CI — check the boundary logic against the
     spellings a real violation would use, and confirm each fails loudly on
     missing input rather than reading "no such file" as "no match."
   - Allowlist escapes through unmodelled syntax: anywhere a hook or script
     vets a command or path string (pre-push hook, release script), enumerate
     constructs that execute or resolve differently than the ones it blocks.
   - Hardcoded defaults in emitted artifacts: anything this repo's own
     tooling writes into *other* repos or generates from a template —
     the cross-repo `dispatch-portfolio` job's payload/target repo, files
     the `port-to-js` skill emits into `js/src/`, anything `scripts/release.mjs`
     stamps — checked against a target differing from the authoring context.

2. **`codebase-health` scan-only leg**, run after the discovery pass above so
   the tool-driven scan doesn't anchor the reading. Invoke it in scan mode
   only: stop at its report, take no approval step, open none of its
   `health/*` branches or PRs. Note in the round's final output which
   language reference it used, or that it had none. Fold anything that
   survives into the ranked list below, de-duplicated against discovery.

3. **De-duplicate.** Only now check this repo's existing debt tracks — there
   is no `PROJECT_STATUS.md` or `PROJECT_ROADMAP.md` in this repo, and no
   in-repo TODO/FIXME markers were found in `mini_vla/` or `js/src/` as of
   staging, so the dedup set is effectively empty; note this in the output
   rather than skipping the step silently. (There *are* three merged
   `codebase-health` PRs from 2026-07-14 — dedup, complexity, docs — visible
   in `git log`; a re-found item already fixed there is replication, report
   it separately as evidence the discovery pass works, never as a new
   finding.)

## Next action

1. `git rebase origin/main` (never the resume verb — it fast-forwards to
   this branch's own upstream and would silently skip anything landed on
   `main` after this branch was cut).
2. Run this repo's own verification gate as the baseline before reviewing:
   `npm ci && npm run typecheck`, `pip install -e ".[dev]" && pytest`, and
   the Playwright E2E suite (`npm run test:e2e` — expect it to need browser
   binaries; `npx playwright install --with-deps <browser>` first if so). A
   failure here is the round's to fix or flag, not to review around.
3. After the rebase and baseline are green (or their failures logged),
   dispatch parallel sub-reviewers over the discovery-pass dimensions in
   the State section above — independent dimensions, so fan them out
   rather than working the list serially.

## Blockers

None yet — nothing has been attempted.

## Gotchas (unpromoted)

- This repo's CI is not opt-in per PR (unlike the toolkit repo this skill
  ships from) — `e2e.yml` triggers automatically on `push`/`pull_request`,
  so the round's own PR needs no opt-in label to get a CI run.
- Deliverable shape: a ranked findings list, each with `file:line` and a
  concrete failure scenario; label reasoned-but-unproven claims as such.
  Fixes never replace the report — this round produces a report, not patches.
- One commit per concern if any fixes are made at all (they should not be,
  per the previous point, except where the owner explicitly asks) — the
  failure it prevents goes in the commit body so any single fix reverts
  alone.
- Get the repo owner's explicit go-ahead before any high-blast-radius
  fix — installer/uninstall safety, destructive git paths, permission or
  settings changes. This repo doesn't ship an installer, but the
  `.githooks/pre-push` hook and CI permission blocks count as this class.
- End the round's output by naming any new review-dimension class its
  findings imply, so it can be added to `references/review-dimensions.md`
  in the `codebase-review` skill's toolkit repo (or, if the class is
  specific to mini-vla's Python↔JS port shape, to this repo's own
  `CLAUDE.md` instead).
