// Inline no-worker fallback: hosts without module workers / OffscreenCanvas
// (older Safari) get VLATrainerCore on the main thread behind the same async
// proxy API. One engine suffices — the fallback selection logic is
// environment-driven, not engine-specific.
import { test, expect } from "@playwright/test";
import { TRAIN_MS } from "./budget";

const HARNESS = "http://localhost:5199/";

test.beforeEach(async ({}, testInfo) => {
  test.skip(
    testInfo.project.name !== "chromium-desktop",
    "fallback path is engine-independent"
  );
});

test("trains inline when OffscreenCanvas is unavailable", async ({ page }) => {
  await page.goto(`${HARNESS}?forceInline=1&max=5`);
  expect(await page.evaluate(() => window.__smoke!.mode)).toBe("inline");

  await expect
    .poll(() => page.evaluate(() => window.__smoke!.state().status), {
      timeout: TRAIN_MS,
    })
    .toBe("paused");
  const st = await page.evaluate(() => window.__smoke!.state());
  expect(st.batches).toBeGreaterThanOrEqual(5);

  // the same async API serves inference from the inline core
  const decoded = await page.evaluate(() =>
    window.__smoke!.decode("grab the red block")
  );
  expect(decoded).not.toBeNull();
  expect(decoded!.color).toBe(0);
});
