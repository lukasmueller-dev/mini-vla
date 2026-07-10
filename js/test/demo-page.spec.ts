// The shipping js/demo page, untouched: start training from its real button
// and read its real DOM readouts. Cheap insurance that the package's own
// zero-host-wiring demo never regresses; the deep assertions live in
// smoke.spec.ts against the instrumented harness.
import { test, expect } from "@playwright/test";

const DEMO = "http://localhost:5198/";

test("demo page boots, trains and pauses", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));

  await page.goto(DEMO);
  await expect(page.locator("#status")).toHaveText("idle");

  await page.locator("#primary").click();
  await expect(page.locator("#status")).toHaveText("training", {
    timeout: 120_000,
  });

  // batches climb and the loss readout goes live
  await expect
    .poll(
      async () => Number(await page.locator("#batches").textContent()),
      { timeout: 120_000 }
    )
    .toBeGreaterThan(3);
  await expect(page.locator("#loss")).not.toHaveText("—");

  // the primary button became Pause; pausing freezes the readout
  await page.locator("#primary").click();
  await expect(page.locator("#status")).toHaveText("paused");

  expect(errors).toEqual([]);
});
