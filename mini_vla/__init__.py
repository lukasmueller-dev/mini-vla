"""mini-vla — Python model-development source of truth.

A language-conditioned spatial-attention behavior-cloning policy: a conv stack
turns a silhouette into a G×G feature map, a language query scores every cell, a
spatial softmax makes an attention map, and a soft-argmax + attended features
drive four heads (action / color / gripper / attention). Trained live against an
analytical-IK expert on synthesized pick-up scenes + slot-grammar commands.

This package is the development surface (architecture, config, wandb experiments).
The `js/` tree is the ported, in-browser TensorFlow.js artifact rendered on the
portfolio; it is regenerated FROM this package (architecture + task parity, not
weights — the browser demo retrains from scratch).

Module map (mirrors js/src/*.ts):
    config      ← config.ts        the single knob sheet
    run_config  ← run-config.ts    user-facing palette / density
    geometry    ← geometry.ts      2-link arm FK/IK, grasp predicate
    task        ← examples.ts      grammar, tokenizer, scene layouts, colors
    render      ← scene.ts         paintSilhouette (the model's-eye input)
    embeddings  ← embeddings.ts    frozen GloVe table loader
    model       ← model.ts         the Keras policy (+ viz / lang twins)
    trainer     ← trainer.core.ts  batch synthesis + training loop
    eval        ← eval/main.ts     closed-loop grasp-rate metric
"""

# NOTE: submodules are imported lazily (not eagerly here) so that a CLI can
# apply CONFIG overrides BEFORE model.py / trainer.py snapshot their module-level
# constants — the same import-order contract the JS eval harness relies on. Also
# keeps `import mini_vla.config` from pulling in TensorFlow/Pillow.

__all__ = [
    "config",
    "run_config",
    "geometry",
    "task",
    "render",
    "embeddings",
    "model",
    "trainer",
    "eval",
]
