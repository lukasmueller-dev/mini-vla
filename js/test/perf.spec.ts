// Convergence-PERFORMANCE E2E. Where smoke.spec asserts the pipeline RUNS
// (loose loss, no grasp floor), this asserts the model still converges within
// its BATCH budget — the number the < 60 s in-browser product budget is built
// on (browser time ≈ 2 s load + batches / 10; see CLAUDE.md and the converge
// notes in js/src/config.ts).
//
// Thresholds are DATA, not code: they live in perf-budgets.json keyed
// [task][profile], so retuning a budget or adding a task (e.g. grasping) is a
// JSON edit — this spec is task- and setup-agnostic, it just looks up the
// budget for whatever (task, profile) the project maps to.
//
// Batches-to-converge is engine-stable, so budgets are per-PROFILE. This suite
// runs on a REAL GPU (playwright.perf.config.ts — no SwiftShader) where a full
// convergence run is ~40 s and representative, and is deliberately OUT of the
// GPU-less CI smoke job (300+ software-GL batches ≈ 20 min/project). Run it
// with `npm run perf`.
import { test, expect, type Page } from "@playwright/test";
import { CONFIG } from "../src/config";
import budgets from "./perf-budgets.json" with { type: "json" };

const HARNESS = "http://localhost:5199/";
const CONVERGE_LOSS = CONFIG.trainer.converge.loss;
const BROWSER_LOAD_S = 2;
const BROWSER_BATCHES_PER_S = 10;

// The only task today is pick-up. A second task adds a sibling key to
// perf-budgets.json and a `?task=` hook to the harness; nothing here changes.
const TASK = "pickup";

interface Budget {
  maxBatchesToConverge: number;
  requireConvergedByLoss: boolean;
  minGraspRate: number;
}

/** Read the trainer telemetry, tolerating a transient window where `__smoke`
    is gone — if the page reloads mid-run (e.g. a browser that crashes its
    content process under the sustained WebGL load), evaluate would otherwise
    throw an opaque error. Report it as a fresh "loading" state so the caller's
    reload detector can fail the run legibly. */
async function snapshot(page: Page) {
  return page.evaluate(() => {
    if (!window.__smoke) return { status: "loading", batches: 0, smoothLoss: NaN };
    const s = window.__smoke.state();
    return { status: s.status, batches: s.batches, smoothLoss: s.smoothLoss };
  });
}

/** Train until the trainer converges on its own OR the batch count blows past
    the budget ceiling (fail FAST — a regression or a collapse seed must not
    grind on to the 800-batch fallback) OR the page reloads under it OR it
    errors. Polls the same public telemetry the portfolio reads. */
async function trainToConvergence(
  page: Page,
  ceiling: number,
  deadlineMs: number
) {
  const start = Date.now();
  let maxSeen = 0;
  for (;;) {
    const s = await snapshot(page);
    maxSeen = Math.max(maxSeen, s.batches);
    // batches went sharply BACKWARDS ⇒ the page reloaded and training reset;
    // it will never converge, so bail now with a legible signal.
    const reloaded = s.batches + 20 < maxSeen;
    if (
      s.status === "converged" ||
      s.status === "error" ||
      s.batches > ceiling ||
      reloaded ||
      Date.now() - start > deadlineMs
    )
      return { ...s, reloaded };
    await page.waitForTimeout(500);
  }
}

test("converges within its batch budget", async ({ page }, testInfo) => {
  // Same project→profile rule the smoke suite uses: phone-class projects train
  // the mobile profile, desktops the desktop one.
  const mobile = /iphone|android/.test(testInfo.project.name);
  const profile = mobile ? "mobile" : "desktop";

  const table = budgets as unknown as Record<
    string,
    Record<string, Budget>
  >;
  const budget = table[TASK]?.[profile];
  expect(
    budget,
    `no perf budget for ${TASK}/${profile} in perf-budgets.json`
  ).toBeTruthy();
  const ceiling = budget.maxBatchesToConverge;

  // Train FREELY to the trainer's own convergence (max=0 → no soft pause).
  await page.goto(`${HARNESS}?max=0&preset=${profile}`);
  expect(await page.evaluate(() => window.__smoke!.mode)).toBe("worker");

  // Leave headroom under the per-test timeout for the grasp-rate rollouts below.
  const st = await trainToConvergence(page, ceiling, testInfo.timeout - 60_000);
  const est = BROWSER_LOAD_S + st.batches / BROWSER_BATCHES_PER_S;

  // Always surface the measured numbers — this suite doubles as the status-quo
  // gauge, so the log line is a deliverable even on a pass.
  const line =
    `[perf] ${testInfo.project.name} ${TASK}/${profile}: ` +
    `batches=${st.batches} status=${st.status} ` +
    `smoothLoss=${Number(st.smoothLoss).toFixed(4)} ` +
    `estBrowser≈${est.toFixed(0)}s (budget ${ceiling} batches)`;
  console.log(line);
  testInfo.annotations.push({ type: "perf", description: line });

  // Within budget, converged by LOSS — not a ceiling trip, not the max-batch
  // fallback.
  expect(
    st.batches,
    `batches ${st.batches} exceeded budget ${ceiling}`
  ).toBeLessThanOrEqual(ceiling);
  expect(
    st.status,
    st.reloaded
      ? `page reloaded mid-training (browser crashed under the load?) — never converged`
      : `did not converge (status=${st.status}, batches=${st.batches})`
  ).toBe("converged");
  if (budget.requireConvergedByLoss)
    expect(
      Number(st.smoothLoss),
      `hit the max-batch fallback, not a loss crossing (smoothLoss=${st.smoothLoss})`
    ).toBeLessThan(CONVERGE_LOSS);

  // Soft closed-loop quality floor (0 disables).
  if (budget.minGraspRate > 0) {
    const gr = await page.evaluate(() => window.__smoke!.graspRate(16));
    console.log(
      `[perf] ${testInfo.project.name} ${TASK}/${profile}: ` +
        `graspRate=${gr.graspRate.toFixed(2)} (${gr.grasps}/${gr.episodes})`
    );
    expect(
      gr.graspRate,
      `grasp rate ${gr.graspRate} below floor ${budget.minGraspRate}`
    ).toBeGreaterThanOrEqual(budget.minGraspRate);
  }
});
