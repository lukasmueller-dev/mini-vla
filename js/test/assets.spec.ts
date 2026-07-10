// Asset-base plumbing and embedding-integrity checks — the two things a host
// cannot fix from outside this package. Engine-independent (no WebGL kernel is
// involved in fetching or validating a byte array), so one desktop project
// carries them; the browser×device matrix stays in smoke.spec.ts.
//
// The harness's vite config mounts the SAME assets/ directory at three URLs:
//   /vla          the default loadEmbeddings() resolves with zero host wiring
//   /custom/base  byte-identical, non-default — proves assetBase is honored
//   /vla-short    embeddings-50d.bin one row short — a host serving a
//                 different generation of the assets than the JS reading them
import { test, expect, type Page } from "@playwright/test";
import { TRAIN_MS, WEDGE_MS } from "./budget";

const HARNESS = "http://localhost:5199/";
const CUSTOM = "/custom/base";

test.beforeEach(async ({}, testInfo) => {
  test.skip(
    testInfo.project.name !== "chromium-desktop",
    "asset plumbing is engine-independent"
  );
});

/** Every embedding-asset URL the page (or its worker) requested. */
function collectAssetRequests(page: Page): string[] {
  const seen: string[] = [];
  page.on("request", (r) => {
    const p = new URL(r.url()).pathname;
    if (/\/(embeddings-50d\.bin|vocab\.txt)$/.test(p)) seen.push(p);
  });
  return seen;
}

test("assetBase is honored on the worker path", async ({ page }) => {
  const requested = collectAssetRequests(page);
  await page.goto(`${HARNESS}?assetBase=${CUSTOM}&max=3`);
  expect(await page.evaluate(() => window.__smoke!.mode)).toBe("worker");

  // it trains, so the table it fetched from the custom base is real
  await expect
    .poll(() => page.evaluate(() => window.__smoke!.state().batches), {
      timeout: TRAIN_MS,
    })
    .toBeGreaterThan(1);

  expect(requested).toContain(`${CUSTOM}/embeddings-50d.bin`);
  expect(requested).toContain(`${CUSTOM}/vocab.txt`);
  // the default path must never be touched once a host names its own base
  expect(requested.filter((p) => p.startsWith("/vla/"))).toEqual([]);
});

test("assetBase reaches the core on the inline fallback path", async ({
  page,
}) => {
  const requested = collectAssetRequests(page);
  await page.goto(`${HARNESS}?forceInline=1&assetBase=${CUSTOM}&max=3`);
  expect(await page.evaluate(() => window.__smoke!.mode)).toBe("inline");

  await expect
    .poll(() => page.evaluate(() => window.__smoke!.state().batches), {
      timeout: TRAIN_MS,
    })
    .toBeGreaterThan(1);

  expect(requested).toContain(`${CUSTOM}/embeddings-50d.bin`);
  expect(requested.filter((p) => p.startsWith("/vla/"))).toEqual([]);
});

test("omitting assetBase still resolves the /vla default", async ({ page }) => {
  const requested = collectAssetRequests(page);
  await page.goto(`${HARNESS}?max=3`);
  expect(await page.evaluate(() => window.__smoke!.assetBase)).toBeUndefined();

  await expect
    .poll(() => page.evaluate(() => window.__smoke!.state().batches), {
      timeout: TRAIN_MS,
    })
    .toBeGreaterThan(1);

  expect(requested).toContain("/vla/embeddings-50d.bin");
  expect(requested).toContain("/vla/vocab.txt");
});

test("truncated embeddings reject, publish no table, and allow a retry", async ({
  page,
}) => {
  // ?autostart=0: nothing has loaded the embeddings yet, so probeEmbeddings
  // drives this thread's loadEmbeddings module state from a clean slate
  await page.goto(`${HARNESS}?autostart=0`);

  const bad = await page.evaluate(() =>
    window.__smoke!.probeEmbeddings("/vla-short")
  );
  expect(bad.ok).toBe(false);
  // the message names the file and both sizes — the whole point is that a
  // mismatch is diagnosable rather than silent
  expect(bad.message).toContain("/vla-short/embeddings-50d.bin");
  expect(bad.message).toContain("1000050"); // actual: one row short
  expect(bad.message).toContain("1000100"); // expected: 20002 words × 50
  // no NaN-poisoned table was published to embeddingMatrix()
  expect(bad.published).toBe(false);

  // the rejection un-cached the promise, so a retry refetches rather than
  // handing back the poisoned one
  const good = await page.evaluate(() => window.__smoke!.probeEmbeddings());
  expect(good.ok).toBe(true);
  expect(good.anyNaN).toBe(false);
  expect(good.published).toBe(true);
  expect(good.rows).toBe(20004 * 50); // VOCAB_SIZE × EMBED_DIM
});

test("a failed asset load surfaces as status error / reason assets", async ({
  page,
}) => {
  // inline core: the failure path is the same catch in VLATrainerCore.start
  // either way, and this keeps the assertion off worker-message timing
  await page.goto(`${HARNESS}?forceInline=1&assetBase=/vla-short&autostart=0`);
  await page.evaluate(() => window.__smoke!.start());

  await expect
    .poll(() => page.evaluate(() => window.__smoke!.state().status), {
      timeout: WEDGE_MS,
    })
    .toBe("error");
  const st = await page.evaluate(() => window.__smoke!.state());
  expect(st.errorReason).toBe("assets");
  expect(st.ready).toBe(false);
  expect(st.batches).toBe(0);
});
