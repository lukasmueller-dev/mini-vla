// Timeout budgets for the E2E specs, in one place.
//
// CI runners have no GPU: every engine falls back to software WebGL (SwiftShader
// for Chromium, each browser's own rasterizer otherwise), so a training run that
// takes ~15s on a laptop takes minutes on a runner. Hard-coding a laptop-sized
// budget is what made the suite red on CI while `npm run release` — which runs
// this same suite locally — stayed green.
//
// Two budgets, deliberately different, because they guard different things:
//
//  - TRAIN_MS  waiting for training to make progress or settle. Generous: how
//              long a software-GL gradient loop takes is not something a spec
//              should have an opinion about.
//  - WEDGE_MS  waiting for the FIRST sign of life (batches climbing past 1) and
//              for control-message round-trips. Kept tight ON PURPOSE: the
//              WebKit fence regression (commit e0ed849) looks exactly like
//              "status=training, batches stuck at 1 forever". A budget loose
//              enough to hide a wedge is a budget that has stopped testing for
//              one — the failure must stay fast and legible, not become a
//              ten-minute timeout nobody reads.
//
// `types: []` in js/tsconfig.json means @types/node isn't loaded, so declare the
// sliver of `process` we touch rather than pulling the whole thing in.
declare const process: { env: Record<string, string | undefined> };

const CI = !!process.env.CI;

/** Training progress / convergence polls. */
export const TRAIN_MS = CI ? 300_000 : 120_000;

/** First batches, and control round-trips — must fail FAST when wedged. */
export const WEDGE_MS = CI ? 150_000 : 60_000;
