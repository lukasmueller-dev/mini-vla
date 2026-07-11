import { defineConfig } from "vite";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

// repo root is two levels up (js/capture/ → js/ → repo root); assets/ lives
// there, shared with the Python package.
const root = fileURLToPath(new URL("../..", import.meta.url));
const assetsDir = path.join(root, "assets");

// Serve assets/ at /vla so loadEmbeddings()'s default assetBase resolves (the
// capture run only needs the embeddings — it TRAINS, it doesn't read replay
// checkpoints). Same middleware the demo/eval pages use.
export default defineConfig({
  server: {
    fs: { allow: [root] },
  },
  plugins: [
    {
      name: "mini-vla-serve-assets",
      configureServer(server) {
        server.middlewares.use("/vla", (req, res, next) => {
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
            res.end(data);
          } catch {
            next();
          }
        });
      },
    },
  ],
});
