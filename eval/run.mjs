// Headless sweep runner: boots the eval Vite page, trains VLATrainerCore in a
// (SwiftShader) Chromium, and prints the collected metrics JSON. The BROWSER
// does the training — never tfjs-node.
//
//   npm run eval                       # defaults: colors=8 blocks=4 max=450 eval=24
//   npm run eval -- colors=4 blocks=3 max=600 set=model.mapLossWeight:2.5
//
// A Chromium is required. In a fresh clone: `npx playwright install chromium`
// (Playwright then auto-locates it). To reuse an existing binary, set
// MINIVLA_CHROMIUM=/path/to/chromium.
import { createServer } from "vite";
import { chromium } from "playwright";
import { fileURLToPath } from "node:url";
import path from "node:path";

const evalDir = fileURLToPath(new URL(".", import.meta.url));

const params = new URLSearchParams();
for (const a of process.argv.slice(2)) {
  const i = a.indexOf("=");
  if (i > 0) params.set(a.slice(0, i), a.slice(i + 1));
}
if (![...params.keys()].length) {
  params.set("colors", "8");
  params.set("blocks", "4");
  params.set("max", "450");
  params.set("eval", "24");
}

const server = await createServer({
  root: evalDir,
  configFile: path.join(evalDir, "vite.config.ts"),
  logLevel: "warn",
});
await server.listen();
const base = server.resolvedUrls?.local?.[0] ?? `http://localhost:5173/`;
const target = base.replace(/\/$/, "") + "/?" + params.toString();
console.log("[eval] " + target);

const browser = await chromium.launch({
  executablePath: process.env.MINIVLA_CHROMIUM || undefined,
  args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
});
const page = await browser.newPage();
page.on("console", (m) => {
  const t = m.text();
  if (t.startsWith("[eval]")) console.log(t);
});
page.on("pageerror", (e) => console.error("[pageerror]", e.message));

await page.goto(target, { waitUntil: "load", timeout: 30000 });

const deadline = Date.now() + 15 * 60 * 1000;
let result = null;
while (Date.now() < deadline) {
  const st = await page.evaluate(() => globalThis.__vlaLab ?? null);
  if (st)
    process.stdout.write(
      `\r[eval] ${st.status} b=${st.batches} smooth=${(st.smoothLoss || 0).toFixed(4)} rollout=${st.rollout ? JSON.stringify(st.rollout) : "…"}          `
    );
  if (st && st.done && st.rollout) {
    result = st;
    break;
  }
  await new Promise((r) => setTimeout(r, 2000));
}
process.stdout.write("\n");

await browser.close();
await server.close();

if (!result) {
  console.error("[eval] timed out before a rollout score landed");
  process.exit(1);
}
console.log(
  "[eval] RESULT " +
    JSON.stringify(
      {
        status: result.status,
        batches: result.batches,
        smoothLoss: result.smoothLoss,
        probes: result.probes?.length ?? 0,
        rollout: result.rollout,
      },
      null,
      2
    )
);
const gr = result.rollout?.graspRate ?? -1;
if (gr < 0) {
  console.error("[eval] eval failed (graspRate < 0)");
  process.exit(1);
}
console.log(`[eval] graspRate = ${(gr * 100).toFixed(1)}%`);
