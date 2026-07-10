// Worker-protocol control round-trips: pause/resume/reset/restart through the
// proxy's fire-and-forget + authoritative-echo scheme, including the `gen`
// guard (no ghost batches after reset). Runs on one desktop and one mobile
// representative — the protocol is engine-independent; the smoke matrix
// already covers per-engine training.
import { test, expect } from "@playwright/test";

const HARNESS = "http://localhost:5199/";
const REPRESENTATIVES = ["chromium-desktop", "webkit-iphone"];

test.beforeEach(async ({}, testInfo) => {
  test.skip(
    !REPRESENTATIVES.includes(testInfo.project.name),
    "protocol covered on representative projects"
  );
});

test("pause / resume / reset / restart via the worker", async ({ page }) => {
  await page.goto(`${HARNESS}?max=0`);

  await expect
    .poll(() => page.evaluate(() => window.__smoke!.state().batches), {
      timeout: 120_000,
    })
    .toBeGreaterThanOrEqual(5);

  // pause: batches freeze. The command is fire-and-forget and the worker
  // finishes its in-flight gradient step first, so one more batch post may
  // land after the optimistic "paused" — drain before taking the frozen
  // reading.
  await page.evaluate(() => window.__smoke!.pause());
  await expect
    .poll(() => page.evaluate(() => window.__smoke!.state().status))
    .toBe("paused");
  await page.waitForTimeout(1000);
  const frozen = await page.evaluate(() => window.__smoke!.state().batches);
  await page.waitForTimeout(1500);
  expect(await page.evaluate(() => window.__smoke!.state().batches)).toBe(
    frozen
  );

  // paused model still serves inference (snapshot + frozen predict)
  await page.evaluate(() => window.__smoke!.snapshot());
  const pred = await page.evaluate(() =>
    window.__smoke!.predict("pick up the red block")
  );
  expect(pred).not.toBeNull();

  // resume: batches climb again
  await page.evaluate(() => window.__smoke!.resume());
  await expect
    .poll(() => page.evaluate(() => window.__smoke!.state().batches), {
      timeout: 60_000,
    })
    .toBeGreaterThan(frozen + 2);

  // reset: back to idle, and the gen guard drops any in-flight batch posts —
  // the mirror must NOT repopulate
  await page.evaluate(() => window.__smoke!.reset());
  expect(await page.evaluate(() => window.__smoke!.state().status)).toBe(
    "idle"
  );
  await page.waitForTimeout(1500);
  const after = await page.evaluate(() => window.__smoke!.state());
  expect(after.status).toBe("idle");
  expect(after.batches).toBe(0);

  // restart on the warm worker: a fresh model trains again
  await page.evaluate(() => window.__smoke!.start());
  await expect
    .poll(() => page.evaluate(() => window.__smoke!.state().batches), {
      timeout: 120_000,
    })
    .toBeGreaterThanOrEqual(2);
});
