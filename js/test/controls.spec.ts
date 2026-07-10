// Worker-protocol control round-trips: pause/resume/reset/restart through the
// proxy's fire-and-forget + authoritative-echo scheme, including the `gen`
// guard (no ghost batches after reset). Runs on one desktop and one mobile
// representative — the protocol is engine-independent; the smoke matrix
// already covers per-engine training.
import { test, expect, type Page } from "@playwright/test";
import { SETTLE_MS, TRAIN_MS, WEDGE_MS } from "./budget";

const HARNESS = "http://localhost:5199/";
const REPRESENTATIVES = ["chromium-desktop", "webkit-iphone"];

test.beforeEach(async ({}, testInfo) => {
  test.skip(
    !REPRESENTATIVES.includes(testInfo.project.name),
    "protocol covered on representative projects"
  );
});

const batchesOf = (page: Page) =>
  page.evaluate(() => window.__smoke!.state().batches);

/** The batch count once it has stopped moving.
 *
 * `pause()` is fire-and-forget: the worker is inside `await trainStep()` when
 * it arrives and finishes that step before noticing, so exactly one more batch
 * can post after the optimistic "paused". Waiting a FIXED drain window for that
 * raced the engine — a software-GL step on CI takes ~3.5s against ~50ms on a
 * laptop GPU, so the late batch landed after the sleep and the reading moved
 * under the assertion. Wait for the counter to hold still instead, for longer
 * than one step could take (SETTLE_MS).
 */
async function settledBatches(page: Page): Promise<number> {
  const deadline = Date.now() + WEDGE_MS;
  let value = -1;
  let stableSince = Date.now();
  while (Date.now() < deadline) {
    const b = await batchesOf(page);
    if (b !== value) {
      value = b;
      stableSince = Date.now();
    } else if (Date.now() - stableSince >= SETTLE_MS) {
      return value;
    }
    await page.waitForTimeout(500);
  }
  throw new Error(`batch counter never settled (last read: ${value})`);
}

test("pause / resume / reset / restart via the worker", async ({ page }) => {
  await page.goto(`${HARNESS}?max=0`);

  await expect
    .poll(() => page.evaluate(() => window.__smoke!.state().batches), {
      timeout: TRAIN_MS,
    })
    .toBeGreaterThanOrEqual(5);

  // pause: batches freeze. Status flips optimistically on the proxy, then the
  // worker drains the one gradient step it was already inside.
  const before = await batchesOf(page);
  await page.evaluate(() => window.__smoke!.pause());
  await expect
    .poll(() => page.evaluate(() => window.__smoke!.state().status))
    .toBe("paused");

  const frozen = await settledBatches(page);
  // the protocol guarantee: AT MOST the in-flight step lands, never a new one
  expect(frozen).toBeLessThanOrEqual(before + 1);
  // …and once settled it stays settled — no gradient step was started after pause
  await page.waitForTimeout(SETTLE_MS);
  expect(await batchesOf(page)).toBe(frozen);

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
      timeout: WEDGE_MS,
    })
    .toBeGreaterThan(frozen + 2);

  // reset: back to idle, and the gen guard drops any in-flight batch posts —
  // the mirror must NOT repopulate. This asserts an ABSENCE, so the wait has to
  // outlast the gradient step that was running when reset landed; a shorter one
  // would pass without ever giving a ghost post the chance to arrive.
  await page.evaluate(() => window.__smoke!.reset());
  expect(await page.evaluate(() => window.__smoke!.state().status)).toBe(
    "idle"
  );
  await page.waitForTimeout(SETTLE_MS);
  const after = await page.evaluate(() => window.__smoke!.state());
  expect(after.status).toBe("idle");
  expect(after.batches).toBe(0);

  // restart on the warm worker: a fresh model trains again
  await page.evaluate(() => window.__smoke!.start());
  await expect
    .poll(() => page.evaluate(() => window.__smoke!.state().batches), {
      timeout: TRAIN_MS,
    })
    .toBeGreaterThanOrEqual(2);
});
