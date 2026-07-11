// The replay fallback (src/trainer.replay.ts): when live training can't run
// (iOS/iPadOS dead WebGL context), a scripted-loss + real-CPU-rollout stand-in
// plays a captured bad→good policy behind the SAME surface. Two things to prove:
//   1. the replayed run is coherent — a clumsy→converged loss AND a real grasp
//      at the end (genuine CPU inference of the converged checkpoint) — and it
//      VARIES between visits (never byte-identical);
//   2. VLATrainer transparently SWAPS to it when the real path stalls, carrying
//      through to a converged, grasping run the host reads identically.
//
// Engine-independent (the replay forces the CPU backend — no GL), so one
// project suffices; gated to chromium-desktop to keep the suite fast.
import { test, expect } from "@playwright/test";
import { TRAIN_MS, WEDGE_MS } from "./budget";

const HARNESS = "http://localhost:5199/";

test.beforeEach(async ({}, testInfo) => {
  test.skip(
    testInfo.project.name !== "chromium-desktop",
    "replay path is engine-independent (CPU backend)"
  );
});

test("replays a coherent bad→good run that grasps and varies", async ({
  page,
}) => {
  await page.goto(`${HARNESS}?forceReplay=1`);
  expect(await page.evaluate(() => window.__smoke!.mode)).toBe("replay");

  // runs to "converged" like a real run
  await expect
    .poll(() => page.evaluate(() => window.__smoke!.state().status), {
      timeout: TRAIN_MS,
    })
    .toBe("converged");

  const st = await page.evaluate(() => window.__smoke!.state());
  expect(st.usingReplay).toBe(true);
  expect(st.errors).toHaveLength(0);
  // a real bad→good arc: clumsy start (anchor ~0.87), converged end (~0.012)
  expect(st.initialLoss).toBeGreaterThan(0.3);
  expect(st.loss).toBeLessThan(0.05);
  expect(st.batches).toBeGreaterThan(200);

  // the loss curve descends hard (first window ≫ last window)
  const curve = await page.evaluate(() => window.__smoke!.lossCurve());
  expect(curve.length).toBeGreaterThan(200);
  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
  expect(mean(curve.slice(0, 10))).toBeGreaterThan(mean(curve.slice(-10)) * 3);

  // real rollout: the converged policy actually grasps (genuine CPU inference)
  const gr = await page.evaluate(() => window.__smoke!.graspRate(8));
  expect(gr.graspRate).toBeGreaterThanOrEqual(0.5);

  // variety: a second visit's loss curve is not byte-identical to the first
  await page.evaluate(() => window.__smoke!.reset());
  await page.evaluate(() => window.__smoke!.start());
  await expect
    .poll(() => page.evaluate(() => window.__smoke!.state().status), {
      timeout: TRAIN_MS,
    })
    .toBe("converged");
  const curve2 = await page.evaluate(() => window.__smoke!.lossCurve());
  expect(curve2).not.toEqual(curve);
});

test("swaps to the replay when the real path stalls", async ({ page }) => {
  // a 100ms load watchdog fires long before the real path's first batch (which
  // is seconds away), so the swap engages on an otherwise-HEALTHY run — the
  // deterministic stand-in for the iOS dead-on-arrival wedge.
  await page.goto(`${HARNESS}?replayFallback=1&watchdogMs=100`);
  expect(await page.evaluate(() => window.__smoke!.mode)).toBe("worker");

  await expect
    .poll(() => page.evaluate(() => window.__smoke!.state().usingReplay), {
      timeout: WEDGE_MS,
    })
    .toBe(true);
  // carries through to a converged, grasping run behind the same surface
  await expect
    .poll(() => page.evaluate(() => window.__smoke!.state().status), {
      timeout: TRAIN_MS,
    })
    .toBe("converged");
  const st = await page.evaluate(() => window.__smoke!.state());
  expect(st.errorReason).toBeNull();
  const gr = await page.evaluate(() => window.__smoke!.graspRate(6));
  expect(gr.graspRate).toBeGreaterThanOrEqual(0.5);
});
