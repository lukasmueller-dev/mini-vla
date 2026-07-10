// End-to-end pipeline smoke, run on EVERY browser×device project: boot the
// worker-backed trainer through the harness page, train ~30 batches, then
// exercise each public surface the portfolio consumes — decode, predict,
// rollout, both painters. Asserts "the pipeline works on this engine", not
// model quality: loss thresholds are deliberately loose and there is no
// grasp-rate floor.
import { test, expect, type Page } from "@playwright/test";

const HARNESS = "http://localhost:5199/";
const MAX_BATCHES = 30;

// console noise that is expected under software GL and must not fail the run
const CONSOLE_WHITELIST = [
  /GroupMarkerNotSet/i,
  /Automatic fallback to software WebGL/i,
  /GPU stall due to ReadPixels/i,
  /WEBGL_(debug|lose)/i,
  /powerPreference option/i,
];

function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("console", (m) => {
    if (m.type() !== "error") return;
    const text = m.text();
    if (CONSOLE_WHITELIST.some((re) => re.test(text))) return;
    errors.push(`console.error: ${text}`);
  });
  return errors;
}

test("trains, decodes, predicts, rolls out and renders", async ({
  page,
}, testInfo) => {
  const mobile = /iphone|android/.test(testInfo.project.name);
  const errors = collectErrors(page);
  await page.goto(`${HARNESS}?max=${MAX_BATCHES}`);

  // the host's device-class rule picked the right profile for this viewport
  expect(await page.evaluate(() => window.__smoke!.preset)).toBe(
    mobile ? "mobile" : "desktop"
  );
  expect(await page.evaluate(() => window.__smoke!.mode)).toBe("worker");
  expect(await page.evaluate(() => window.__smoke!.state().numColors)).toBe(
    mobile ? 4 : 8
  );

  // WebKit-wedge regression (commit e0ed849): the historical failure mode is
  // status "training" with batches frozen at 1 forever — so specifically
  // require batches to pass 1, then reach the cap.
  await expect
    .poll(() => page.evaluate(() => window.__smoke!.state().batches), {
      timeout: 120_000,
    })
    .toBeGreaterThan(1);
  await expect
    .poll(() => page.evaluate(() => window.__smoke!.state().status), {
      timeout: 120_000,
    })
    .toBe("paused");

  // gradient steps actually learned something (loose: healthy runs sit ~0.4
  // at batch 30 from a ~0.8 start; this only rules out flat/NaN loss)
  const st = await page.evaluate(() => window.__smoke!.state());
  expect(st.batches).toBeGreaterThanOrEqual(MAX_BATCHES);
  expect(st.ready).toBe(true);
  expect(Number.isFinite(st.smoothLoss)).toBe(true);
  expect(Number.isFinite(st.initialLoss)).toBe(true);
  expect(st.smoothLoss).toBeLessThan(0.65);
  expect(st.smoothLoss).toBeLessThan(st.initialLoss);

  // language decode round-trip through the worker: the warm-up trains the
  // color head before batch 0, so the LAST active palette color (trained on
  // both presets) must decode to its own index with real confidence
  const pal = await page.evaluate(() => window.__smoke!.palette());
  expect(pal).toHaveLength(mobile ? 4 : 8);
  const last = pal.length - 1;
  const decoded = await page.evaluate(
    ([syn]) => window.__smoke!.decode(`lift the ${syn} block`),
    [pal[last].synonym]
  );
  expect(decoded).not.toBeNull();
  expect(decoded!.color).toBe(last);
  expect(decoded!.colorProb).toBeGreaterThan(0.5);

  // PredictResult contract — the exact shape the portfolio's rollout + viz
  // panels consume, per engine
  const pred = await page.evaluate(() =>
    window.__smoke!.predict("grab the red block")
  );
  expect(pred).not.toBeNull();
  const p = pred!;
  expect(p.attnLen).toBe(p.expectedAttnLen);
  expect(p.attnFinite).toBe(true);
  expect(p.attnMax).toBeGreaterThan(0.99); // peak-normalized map
  expect(p.attnMax).toBeLessThanOrEqual(1.001);
  expect(p.attnMin).toBeGreaterThanOrEqual(0);
  expect(p.grip).toBeGreaterThanOrEqual(0); // sigmoid head
  expect(p.grip).toBeLessThanOrEqual(1);
  for (const c of p.xy) {
    expect(c).toBeGreaterThanOrEqual(0); // soft-argmax stays inside the image
    expect(c).toBeLessThanOrEqual(1);
  }
  for (const t of p.target) {
    expect(Number.isFinite(t)).toBe(true);
    expect(Math.abs(t)).toBeLessThan(10); // raw head output: sane, not exact
  }

  // closed-loop rollout: a full episode terminates on its own with finite
  // poses and live predictions (any outcome — grasp OR timeout — passes)
  const ep = await page.evaluate(() => window.__smoke!.rollout());
  expect(ep.ended).toBe(true);
  expect(ep.frames).toBeGreaterThan(10);
  expect(ep.nonFinite).toBe(0);
  expect(ep.sawTarget).toBe(true);
  expect(ep.phases[0]).toBe("reach");

  // both painters draw non-blank output on this engine/DPR
  const paint = await page.evaluate(() => window.__smoke!.paintCheck());
  expect(paint.sceneCoverage).toBeGreaterThan(0.01);
  expect(paint.silNonWhite).toBeGreaterThan(0.005);
  expect(paint.silColored).toBeGreaterThan(0.0005); // blocks really rendered

  // no page errors, no unhandled rejections, no non-whitelisted console errors
  expect(await page.evaluate(() => window.__smoke!.state().errors)).toEqual([]);
  expect(errors).toEqual([]);
});
