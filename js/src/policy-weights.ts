// Compact (de)serialization of the model's TRAINABLE policy weights — the
// replay fallback's checkpoint format (see trainer.replay.ts + the capture
// script js/capture/). Deliberately EXCLUDES two layers:
//   - text_embedding: the frozen 20k×50 GloVe table (1M floats) — re-seeded
//     from the already-shipped assets/embeddings-50d.bin, never duplicated.
//   - pick_grid: the frozen soft-argmax coordinate kernel — recomputed from
//     the config formula at build time.
// so each checkpoint is only the tens-of-KB of conv/scorer/query/fusion/head
// weights. Order is model.layers order, which buildVLAModel produces
// deterministically — capture and replay build the identical architecture, so
// a positional mapping is robust even for tfjs's auto-named layers.

import type * as tfType from "@tensorflow/tfjs";
import type { TF } from "./model";

const EXCLUDED = new Set(["text_embedding", "pick_grid"]);

/** One trainable weight tensor's identity + shape (the shared layout across all
    checkpoints — shipped once in the manifest). */
export interface PolicyWeightSpec {
  /** `${layerName}/${weightIndex}` — for a fail-loud check on load. */
  name: string;
  shape: number[];
}

/** A captured policy: the shared layout + one checkpoint's flat values. */
export interface PolicyCheckpoint {
  specs: PolicyWeightSpec[];
  data: Float32Array;
}

/** Snapshot the trainable (non-embedding, non-grid) weights of `model`, in
    layer order, as a flat Float32Array + its layout. `getWeights()` returns the
    layers' LIVE variable tensors — read (dataSync) and COPY them; never dispose
    them. */
export function extractPolicyWeights(model: tfType.LayersModel): PolicyCheckpoint {
  const specs: PolicyWeightSpec[] = [];
  const chunks: Float32Array[] = [];
  for (const layer of model.layers) {
    if (EXCLUDED.has(layer.name)) continue;
    layer.getWeights().forEach((w, i) => {
      specs.push({ name: `${layer.name}/${i}`, shape: w.shape.slice() });
      chunks.push(Float32Array.from(w.dataSync() as Float32Array));
    });
  }
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const data = new Float32Array(total);
  let off = 0;
  for (const c of chunks) {
    data.set(c, off);
    off += c.length;
  }
  return { specs, data };
}

/** Load a captured checkpoint into `model` (already built by buildVLAModel, so
    the embedding + grid are seeded) — overwriting only the trainable layers in
    the same order extractPolicyWeights used. Because the viz/lang twins SHARE
    these layer objects, updating the main model updates them too. Fails loud on
    a name/shape mismatch (a checkpoint built for a different architecture). */
export function applyPolicyWeights(
  tf: TF,
  model: tfType.LayersModel,
  ckpt: PolicyCheckpoint
): void {
  let si = 0;
  let off = 0;
  for (const layer of model.layers) {
    if (EXCLUDED.has(layer.name)) continue;
    const ws = layer.getWeights();
    if (ws.length === 0) continue;
    const next = ws.map((w, i) => {
      const spec = ckpt.specs[si++];
      const name = `${layer.name}/${i}`;
      const size = w.shape.reduce((a, b) => a * b, 1);
      if (!spec || spec.name !== name || spec.shape.join(",") !== w.shape.join(","))
        throw new Error(
          `mini-vla replay: checkpoint layout mismatch at ${name} ` +
            `(expected ${w.shape.join("×")}, manifest had ${spec ? spec.shape.join("×") + " " + spec.name : "nothing"}) — regenerate with \`npm run gen:replay\``
        );
      const slice = ckpt.data.subarray(off, off + size);
      off += size;
      return tf.tensor(slice, w.shape);
    });
    layer.setWeights(next);
    next.forEach((t) => t.dispose());
  }
  if (si !== ckpt.specs.length)
    throw new Error(
      `mini-vla replay: checkpoint has ${ckpt.specs.length} weights but the ` +
        `model consumed ${si} — architecture drift; regenerate with \`npm run gen:replay\``
    );
}
