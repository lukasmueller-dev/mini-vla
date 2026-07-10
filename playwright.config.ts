// Browser × device E2E matrix for the js/ artifact — the version that lands
// on the portfolio. Engines: Chromium (Chrome/Edge/Android Chrome), WebKit
// (Safari; ALL iOS browsers are WebKit underneath), Firefox (Gecko). Device
// class decides the RunConfig profile via the same ≥1100px rule the host
// applies, so each project trains the preset its device class is actually
// served.
//
// Known limits, on purpose:
//  - no firefox-mobile: Playwright's Firefox doesn't support mobile emulation
//    (isMobile), and Android Firefox share doesn't justify a workaround.
//  - webkit projects verify the WEBGL_FENCE_API_ENABLED workaround
//    (trainer.core.ts maybeDisableWebGLFence, commit e0ed849) keeps training
//    advancing past batch 1 — the original iOS fence hang itself only
//    reproduces on real hardware.
import { defineConfig, devices } from "@playwright/test";

// SwiftShader software GL — same flags the headless eval runner uses; makes
// Chromium projects work on GPU-less CI runners.
const chromiumGL = {
  args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
};

export default defineConfig({
  testDir: "./js/test",
  testMatch: "**/*.spec.ts",
  // perf.spec.ts trains to full convergence and needs a real GPU — it has its
  // own opt-in config (playwright.perf.config.ts, `npm run perf`) and must NOT
  // run in this GPU-less CI smoke suite.
  testIgnore: "**/perf.spec.ts",
  // software-GL training runs are slow; individual asserts poll well under this
  // (js/test/budget.ts holds the per-assert budgets, and scales them on CI)
  timeout: process.env.CI ? 600_000 : 240_000,
  expect: { timeout: 60_000 },
  // Each worker spawns a browser holding a full WebGL training loop. On CI the
  // workflow shards by PROJECT across runners, so a job has one project to run
  // and one core-starved software-GL loop is plenty — two of them on a 4-vCPU
  // runner is what made every non-chromium project time out.
  workers: process.env.CI ? 1 : 3,
  retries: process.env.CI ? 1 : 0,
  forbidOnly: !!process.env.CI,
  reporter: process.env.CI
    ? [["list"], ["html", { open: "never" }]]
    : [["list"]],
  webServer: [
    {
      command: "npx vite js/test/harness --port 5199 --strictPort",
      url: "http://localhost:5199",
      reuseExistingServer: !process.env.CI,
      stdout: "ignore",
    },
    {
      command: "npx vite js/demo --port 5198 --strictPort",
      url: "http://localhost:5198",
      reuseExistingServer: !process.env.CI,
      stdout: "ignore",
    },
  ],
  projects: [
    {
      name: "chromium-desktop",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1440, height: 900 },
        launchOptions: chromiumGL,
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
      use: { ...devices["Pixel 7"], launchOptions: chromiumGL },
    },
  ],
});
