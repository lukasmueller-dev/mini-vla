// DEV-ONLY runner (npm run gen:replay): boots the capture page, trains a REAL
// VLATrainerCore to convergence in a (SwiftShader) Chromium, and writes the
// replay fallback's policy checkpoints + manifest to assets/replay/. The
// BROWSER does the training — never tfjs-node.
//
// Requires a Chromium (npx playwright install chromium). Set
// MINIVLA_CHROMIUM=/path/to/chromium to reuse an existing binary.

import { createServer } from "vite";
import { chromium } from "playwright";
import { fileURLToPath } from "node:url";
import { writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";

const captureDir = fileURLToPath(new URL(".", import.meta.url));
const outDir = path.join(captureDir, "../../assets/replay");

const server = await createServer({
  root: captureDir,
  configFile: path.join(captureDir, "vite.config.ts"),
  logLevel: "warn",
});
await server.listen();
const base = (server.resolvedUrls?.local?.[0] ?? "http://localhost:5173/").replace(/\/$/, "");

const browser = await chromium.launch({
  executablePath: process.env.MINIVLA_CHROMIUM || undefined,
  args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
});
const page = await browser.newPage();
page.on("pageerror", (e) => console.error("[pageerror]", e.message));

console.log("[capture] " + base);
await page.goto(base, { waitUntil: "load", timeout: 30000 });

const deadline = Date.now() + 10 * 60 * 1000;
let cap = null;
while (Date.now() < deadline) {
  const st = await page.evaluate(() => globalThis.__vlaCapture ?? null);
  if (st)
    process.stdout.write(
      `\r[capture] ${st.status} b=${st.batches} ckpts=${st.checkpoints.length}          `
    );
  if (st && st.done) {
    cap = st;
    break;
  }
  await new Promise((r) => setTimeout(r, 1000));
}
process.stdout.write("\n");
await browser.close();
await server.close();

if (!cap || cap.status === "error" || cap.checkpoints.length === 0) {
  console.error("[capture] failed:", cap ? `status=${cap.status} ckpts=${cap.checkpoints.length}` : "timed out");
  process.exit(1);
}

// Dedupe by sample count (the always-on final grab can coincide with the last
// milestone) — keep first occurrence, preserving the bad→good order.
const seen = new Set();
const ckpts = cap.checkpoints.filter((c) => {
  if (seen.has(c.samples)) return false;
  seen.add(c.samples);
  return true;
});

mkdirSync(outDir, { recursive: true });
const manifest = {
  batchSize: cap.batchSize,
  cadencePerSec: cap.cadencePerSec,
  floorLoss: cap.floorLoss,
  weightSpecs: cap.weightSpecs,
  checkpoints: [],
};
ckpts.forEach((c, i) => {
  const file = `ckpt-${i}.bin`;
  writeFileSync(path.join(outDir, file), Buffer.from(new Float32Array(c.weights).buffer));
  manifest.checkpoints.push({ samples: c.samples, loss: Number(c.loss.toFixed(5)), file });
});
writeFileSync(path.join(outDir, "manifest.json"), JSON.stringify(manifest, null, 2));

const floats = ckpts[0].weights.length;
console.log(`[capture] wrote ${ckpts.length} checkpoints (${floats} floats each, ${(floats * 4) / 1024 | 0}KB) + manifest to assets/replay/`);
console.log(manifest.checkpoints.map((c) => `  ${c.file}: samples=${c.samples} loss=${c.loss}`).join("\n"));
