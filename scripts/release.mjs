#!/usr/bin/env node
// Gated release: the ONLY sanctioned way to mint a version tag. Refuses on a
// dirty tree, a branch other than main, a local main that hasn't reached
// origin (i.e. hasn't gone through a reviewed PR merge), or a package.json
// version that disagrees with the tag; runs typecheck + the full
// browser×device E2E suite (playwright.config.ts); and only then tags. The
// portfolio consumes this repo as a git-ref npm dependency (github:…#<tag>),
// so an un-gated hand-typed `git tag` is exactly how a broken artifact would
// go live.
//
//   npm run release -- v0.4.0
//
// The budget/perf gate (batches-to-converge, ms/batch ceilings) is still not
// wired in here — perf.spec.ts needs a real GPU (playwright.perf.config.ts),
// so it can't run inline in this script the way the GPU-less E2E suite does.
// The timing-calibration reconciliation this comment used to wait on landed
// in bae50ef (2026-07-13); check-python's train.py runs are the CI-side
// substitute (Python-only — see js/test/perf.spec.ts for the browser gate).
import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";

const tag = process.argv[2];
if (!/^v\d+\.\d+\.\d+$/.test(tag ?? "")) {
  console.error("usage: npm run release -- vX.Y.Z");
  process.exit(1);
}

const run = (cmd) => {
  console.log(`\n[release] ${cmd}`);
  execSync(cmd, { stdio: "inherit" });
};

const dirty = execSync("git status --porcelain").toString().trim();
if (dirty) {
  console.error("[release] working tree is dirty — commit or stash first:\n" + dirty);
  process.exit(1);
}

// Provenance gate: without this, any branch can pass every other check below
// (clean tree, version match, typecheck, E2E) and mint a tag whose commit
// never went through a reviewed PR merge into main — the checks here verify
// the CONTENT is release-worthy, not that it took the sanctioned PATH there.
const branch = execSync("git rev-parse --abbrev-ref HEAD").toString().trim();
if (branch !== "main") {
  console.error(
    `[release] must release from 'main' (currently on '${branch}') — merge via a reviewed PR first.`
  );
  process.exit(1);
}
execSync("git fetch origin main --quiet");
const localHead = execSync("git rev-parse HEAD").toString().trim();
const remoteMain = execSync("git rev-parse origin/main").toString().trim();
if (localHead !== remoteMain) {
  console.error(
    "[release] local main is not the same commit as origin/main — pull (if behind) " +
      "or push and merge via PR (if ahead) before releasing."
  );
  process.exit(1);
}

const existing = execSync(`git tag -l ${tag}`).toString().trim();
if (existing) {
  console.error(`[release] tag ${tag} already exists`);
  process.exit(1);
}

// package.json's version is not cosmetic: hosts version-stamp their asset
// directory with it (public/vla/<version>/) and pass it back as the trainer's
// assetBase, deriving BOTH from this one field via the exported "./package.json".
// If it drifts from the tag, a v0.5.0 release would serve its assets at
// /vla/0.4.0/ — a stale tab would find a path that exists and load embeddings
// from the wrong generation, which is precisely the corruption the versioned
// path exists to prevent. Bump it in the same commit as the release.
const { version } = JSON.parse(readFileSync("package.json", "utf8"));
if (`v${version}` !== tag) {
  console.error(
    `[release] package.json version is ${version}, but the tag is ${tag}.\n` +
      `[release] Hosts derive their asset path from package.json — bump it to ` +
      `${tag.slice(1)} and commit before tagging.`
  );
  process.exit(1);
}

run("npm run typecheck");
run("npx playwright test");

run(`git tag -a ${tag} -m "release ${tag} (e2e matrix green)"`);
console.log(
  `\n[release] ${tag} tagged. Push with:\n` +
    `  git push origin ${tag}\n` +
    `then bump the portfolio's mini-vla dependency to #${tag} and redeploy.`
);
