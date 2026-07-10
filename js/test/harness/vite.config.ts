import { defineConfig, type ViteDevServer } from "vite";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

// repo root is three levels up (js/test/harness/ → js/test/ → js/ → repo
// root); assets/ lives there, shared with the Python package.
const root = fileURLToPath(new URL("../../..", import.meta.url));
const assetsDir = path.join(root, "assets");

// EMBED_DIM — one int8 row of the embedding table. Literal rather than
// imported: this file isn't part of the TS project (tsconfig excludes
// *.config.ts), and the truncation only has to be "one row short".
const EMBED_DIM = 50;

/** Serve assets/ at `mount`, optionally mangling the bytes on the way out.
    Mounting the same directory at several URLs is what lets a spec prove
    `assetBase` is honored: identical bytes, different path. */
function serveAssets(
  mount: string,
  transform?: (file: string, data: Buffer) => Buffer
) {
  return {
    name: `mini-vla-serve-assets${mount.replace(/\//g, "-")}`,
    configureServer(server: ViteDevServer) {
      server.middlewares.use(mount, (req, res, next) => {
        const rel = decodeURIComponent((req.url || "/").split("?")[0]);
        const file = path.join(assetsDir, rel);
        if (!file.startsWith(assetsDir)) {
          res.statusCode = 403;
          return res.end();
        }
        try {
          const data = readFileSync(file);
          res.setHeader(
            "Content-Type",
            file.endsWith(".bin")
              ? "application/octet-stream"
              : "text/plain; charset=utf-8"
          );
          res.end(transform ? transform(file, data) : data);
        } catch {
          next();
        }
      });
    },
  };
}

// Same asset serving as the demo/eval pages: expose assets/ at /vla so
// loadEmbeddings()'s default assetBase resolves, and allow reading the
// sibling src/ (the harness imports the package source directly, and the
// trainer's module Worker lives there).
//
// Two extra mounts exist only for the specs:
//   /custom/base  byte-identical assets at a NON-default URL — assetBase.spec
//                 asserts the trainer fetches here and never touches /vla.
//   /vla-short    embeddings-50d.bin one row short, vocab.txt intact: a host
//                 serving a DIFFERENT generation of the assets than the JS
//                 consuming them, which embeddings.ts must reject loudly.
export default defineConfig({
  server: {
    fs: { allow: [root] },
  },
  plugins: [
    serveAssets("/vla"),
    serveAssets("/custom/base"),
    serveAssets("/vla-short", (file, data) =>
      file.endsWith(".bin") ? data.subarray(0, data.length - EMBED_DIM) : data
    ),
  ],
});
