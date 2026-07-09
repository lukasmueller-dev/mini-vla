import { defineConfig } from "vite";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = fileURLToPath(new URL("..", import.meta.url));
const assetsDir = path.join(root, "assets");

// Serve the package's assets/ at /vla so loadEmbeddings()'s default assetBase
// ("/vla") resolves with zero host wiring — the same path the portfolio copies
// its assets into (public/vla/). Also allow Vite to read the sibling src/ (the
// demo imports the package source directly, and the trainer's module Worker
// lives there).
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
