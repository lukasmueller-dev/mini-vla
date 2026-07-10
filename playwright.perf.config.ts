// Convergence-PERFORMANCE matrix (js/test/perf.spec.ts). Deliberately SEPARATE
// from playwright.config.ts (the per-PR smoke suite): a perf run trains to full
// convergence (~300-450 batches), which on the GPU-less CI runners' software
// WebGL would be ~20 min/project. So this suite is OPT-IN and expects a REAL
// GPU — no SwiftShader flags — where a run is ~40 s and the wall-clock is
// representative of a portfolio visitor's device. Run locally (or in a
// dedicated nightly job) with `npm run perf`; the smoke suite stays the CI gate.
//
// Batches-to-converge is engine-stable, so the per-profile budgets in
// perf-budgets.json hold across engines; the projects below exist to gauge each
// DEVICE's wall-clock and to catch an engine-specific convergence regression.
import { defineConfig, devices } from "@playwright/test";

// Nudge headless Chromium onto the real GPU (macOS Metal) instead of its
// default software fallback — the whole point of running perf off-CI. Harmless
// if a GPU isn't available (it just falls back, only slower).
const chromiumGPU = {
  args: ["--ignore-gpu-blocklist", "--enable-gpu", "--use-angle=metal"],
};

export default defineConfig({
  testDir: "./js/test",
  testMatch: "**/perf.spec.ts",
  // A full convergence run on a real GPU is well under a minute; this ceiling
  // is slack for a slow seed (and for an engine that still software-renders).
  timeout: 1_200_000,
  expect: { timeout: 60_000 },
  // Serial: each test is a full training loop, and one GPU shared across
  // parallel runs would both contend and muddy the per-device wall-clock.
  workers: 1,
  // One retry absorbs the two ways a full ~500-batch run flakes without a
  // regression: a browser occasionally crashing its content process under the
  // sustained WebGL load (seen on headless Firefox — the page resets to
  // batches=0, which perf.spec's reload detector fails fast and legibly), and
  // the model's own seed variance (a slow/collapse init). A genuine regression
  // fails every attempt; only reruns on failure, so healthy runs cost nothing.
  retries: 1,
  reporter: [["list"]],
  // Only the instrumented harness is needed (no demo-page spec here).
  webServer: [
    {
      command: "npx vite js/test/harness --port 5199 --strictPort",
      url: "http://localhost:5199",
      reuseExistingServer: true,
      stdout: "ignore",
    },
  ],
  projects: [
    {
      name: "chromium-desktop",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1440, height: 900 },
        launchOptions: chromiumGPU,
      },
    },
    {
      name: "firefox-desktop",
      use: {
        ...devices["Desktop Firefox"],
        viewport: { width: 1440, height: 900 },
      },
    },
    {
      name: "webkit-desktop",
      use: {
        ...devices["Desktop Safari"],
        viewport: { width: 1440, height: 900 },
      },
    },
    {
      name: "webkit-iphone",
      use: { ...devices["iPhone 13"] },
    },
    {
      name: "chromium-android",
      use: { ...devices["Pixel 7"], launchOptions: chromiumGPU },
    },
  ],
});
