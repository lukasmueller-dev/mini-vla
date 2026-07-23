// Software-GL (SwiftShader) run of the convergence-perf suite
// (js/test/perf.spec.ts), wired up by .github/workflows/perf-nightly.yml.
//
// WHY THIS EXISTS: perf.spec.ts is the only gate on batches-to-converge (the
// number the <60s in-browser product budget in CLAUDE.md is built on), but it
// is excluded from the required `e2e` gate (playwright.config.ts's
// testIgnore) and only otherwise runs via playwright.perf.config.ts's opt-in,
// real-GPU-only `npm run perf`. That leaves a JS-only architecture regression
// (a port bug that inflates batches-to-converge without touching mini_vla/)
// with ZERO CI coverage — check-python in e2e.yml only calibrates the Python
// side. This config closes part of that gap by running the same spec headless
// on CI's GPU-less runners, on a schedule (see perf-nightly.yml), not on every
// PR.
//
// KNOWN LIMITATION: batches-to-converge is engine-stable (perf-budgets.json's
// own comment), so the budget check itself is trustworthy under software GL.
// But wall-clock under SwiftShader is NOT representative of a real device —
// e2e.yml estimates a full convergence run this way at ~20 min, vs. ~40s on a
// real GPU. That's why this stays a separate, non-blocking, scheduled
// workflow rather than folding into playwright.perf.config.ts or the required
// `e2e` gate.
//
// Deliberately ONE project (chromium-desktop): perf-budgets.json's budgets are
// per-PROFILE (desktop/mobile), not per-device, so one engine catches a
// regression; running the full 5-project matrix here would just multiply CI
// minutes (already ~20 min for one) for no extra signal. Use
// playwright.perf.config.ts's real-GPU matrix if a specific engine needs
// checking.
import { defineConfig, devices } from "@playwright/test";

// Same SwiftShader launch args as playwright.config.ts's chromiumGL — makes
// Chromium train on software GL instead of requiring a real GPU.
const chromiumGL = {
  args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
};

export default defineConfig({
  testDir: "./js/test",
  testMatch: "**/perf.spec.ts",
  // Real-GPU runs finish in ~40s (playwright.perf.config.ts's 1_200_000ms
  // ceiling is slack over that); software GL is on the order of ~20 min
  // (e2e.yml), so give this real headroom instead of the real-GPU ceiling.
  timeout: 1_800_000,
  expect: { timeout: 60_000 },
  // Serial, single project — no GPU contention to worry about here since
  // there's no GPU, but a single long training loop is still the only worker
  // this needs.
  workers: 1,
  // One retry absorbs seed variance / an occasional crashed content process
  // under sustained software-rendered WebGL load, same rationale as
  // playwright.perf.config.ts.
  retries: 1,
  reporter: process.env.CI
    ? [["list"], ["html", { open: "never" }]]
    : [["list"]],
  // Only the instrumented harness is needed (no demo-page spec here).
  webServer: [
    {
      command: "npx vite js/test/harness --port 5199 --strictPort",
      url: "http://localhost:5199",
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
  ],
});
