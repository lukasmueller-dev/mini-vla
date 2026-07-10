#!/usr/bin/env node
// Gated release: the ONLY sanctioned way to mint a version tag. Refuses on a
// dirty tree, runs typecheck + the full browser×device E2E suite
// (playwright.config.ts), and only then tags. The portfolio consumes this
// repo as a git-ref npm dependency (github:…#<tag>), so an un-gated hand-typed
// `git tag` is exactly how a broken artifact would go live.
//
//   npm run release -- v0.3.0
//
// Deliberately left open: the budget/perf gates (batches-to-converge,
// ms/batch ceilings) slot in here once the repo's conflicting timing
// calibrations are reconciled — add them between the E2E suite and the tag.
import { execSync } from "node:child_process";

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

const existing = execSync(`git tag -l ${tag}`).toString().trim();
if (existing) {
  console.error(`[release] tag ${tag} already exists`);
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
