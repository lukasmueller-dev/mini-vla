import { defineConfig } from "vite";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

// repo root is three levels up (js/test/harness/ → js/test/ → js/ → repo
// root); assets/ lives there, shared with the Python package.
const root = fileURLToPath(new URL("../../..", import.meta.url));
const assetsDir = path.join(root, "assets");

// Same asset serving as the demo/eval pages: expose assets/ at /vla so
// loadEmbeddings()'s default assetBase resolves, and allow reading the
// sibling src/ (the harness imports the package source directly, and the
// trainer's module Worker lives there).
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
