# VLA network definition. Keras port of js/src/model.ts, kept structurally
# parallel (same layer names, same op order) so the Python→JS port skill stays
# mechanical.
#
# Vision→action is a language-conditioned SPATIAL ATTENTION readout, not a
# flatten→dense fusion. The conv stack produces a G×G feature map; a single
# language query dot-product-scores every map cell ("does this cell look like
# the commanded block?"); a spatial softmax turns the score row into an
# attention map; and the readout is the map's SOFT-ARGMAX — the expected (x, y)
# image coordinate under the map — plus its attention-weighted feature vector. A
# small dense head then regresses the target joint angles.
#
# Language: ONE attention-pooled slot with a LOCAL-CONTEXT (conv) scorer. A
# conv1d scorer (kernel 7 over the token axis, linear) scores every token from
# itself PLUS its ±3-token neighborhood, and attention-pools the sequence into a
# single vector the color head decodes and the attention query reads. The
# embedding table is PRETRAINED (a ~20k-word GloVe 50d slice) and FROZEN: only
# the scorer/head/fusion fine-tune on top of it — so unseen near-synonyms ride
# along on the frozen geometry. The token-attention scorer is LINEAR so
# trainer.attention_weights() can recompute the exact per-token weights CPU-side.
#
# PROPRIOCEPTIVE CARRY FLAG: a third input (1 scalar) tells the network whether
# the gripper is holding a block. The same (scene, sentence) has two far-apart
# action targets — the grasp point when empty-handed, the carry-home (REST)
# target when holding — and the flag is the disambiguator. It feeds the attention
# QUERY too.

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from .config import CONFIG, ConvLayer
from .task import COLORS, MAX_SEQ_LEN
from .vocab_gen import EMBED_DIM, VOCAB_SIZE

# Every architecture/optimizer knob below is tuned in config.py.
IMG_SIZE = CONFIG.model.imgSize
LEARNING_RATE = CONFIG.model.learningRate
COLOR_LOSS_WEIGHT = CONFIG.model.colorLossWeight
MAP_LOSS_WEIGHT = CONFIG.model.mapLossWeight
GRIPPER_LOSS_WEIGHT = CONFIG.model.gripperLossWeight
ACTION_HUBER_DELTA = CONFIG.model.actionHuberDelta


def _conv_out_size(size: int, layer: ConvLayer) -> int:
    """Spatial size after one conv stage (+ optional pool)."""
    stride = layer.stride or 1
    if layer.padding == "valid":
        size = (size - layer.kernel) // stride + 1
    else:
        size = math.ceil(size / stride)
    if layer.pool:
        size = size // 2
    return size


# Side length G of the attention grid — the final conv map's spatial size.
ATTN_GRID = IMG_SIZE
for _l in CONFIG.model.conv:
    ATTN_GRID = _conv_out_size(ATTN_GRID, _l)


@tf.keras.utils.register_keras_serializable(package="mini_vla")
class AttentionPooling(tf.keras.layers.Layer):
    """Masked attention-pooling: given token embeddings [B, T, D], their scores
    [B, T, 1], and the raw token ids [B, T], mask out padding (id 0), softmax the
    scores over the token axis, and return the attention-weighted sum [B, D].
    Holds no weights of its own — the trainable scorer is a separate Conv1D."""

    def call(self, inputs):
        embedded, scores, token_ids = inputs
        mask = tf.cast(tf.not_equal(token_ids, 0), tf.float32)  # [B, T]
        s = tf.squeeze(scores, axis=2)  # [B, T]
        # pad positions get -1e9 before the softmax, so they take ~0 weight
        masked = s + (mask - 1.0) * 1e9
        weights = tf.nn.softmax(masked, axis=-1)  # [B, T]
        return tf.reduce_sum(embedded * tf.expand_dims(weights, -1), axis=1)  # [B, D]

    def compute_output_shape(self, input_shape):
        emb = input_shape[0]
        return (emb[0], emb[-1])


@dataclass
class VLAModels:
    # Main policy (the one that trains): [vision, tokens, carry] →
    # [action angles (2), color softmax, attention map [G*G], gripper (1)].
    model: tf.keras.Model
    # Inference/readout twin sharing every layer: [vision, tokens, carry] →
    # [action, attention map, gripper]. One predict yields all three.
    viz: tf.keras.Model
    # Language-only training twin: [tokens] → [color]. Shares the embedding +
    # conv scorer + color head with `model` but EXCLUDES the vision branch, so a
    # gradient step updates ONLY the language weights (the warm-up basis). Its
    # own optimizer, compiled below.
    lang: tf.keras.Model


def build_vla_model(embed_matrix: np.ndarray) -> VLAModels:
    """Build + compile the policy and its viz/lang twins. `embed_matrix` is the
    dequantized pretrained GloVe table, [VOCAB_SIZE, EMBED_DIM] (embeddings.load)."""
    L = tf.keras.layers

    # Language branch first — the vision branch's attention query consumes the
    # pooled language slot, so it has to exist before the CNN is wired.
    lang_input = L.Input(shape=(MAX_SEQ_LEN,), name="language_tokens", dtype="int32")
    carry_input = L.Input(shape=(1,), name="carry_flag")  # proprioceptive flag
    embedded = L.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBED_DIM,
        trainable=False,  # frozen pretrained backbone
        name="text_embedding",
    )(lang_input)  # [T, EMBED_DIM]
    # per-token scores from the token PLUS its ±3 neighborhood (kernel 7). LINEAR
    # (no activation) so attention_weights() can recompute the same scores.
    scores_target = L.Conv1D(
        filters=1, kernel_size=7, padding="same", name="attn_score"
    )(embedded)  # [T, 1]
    lang_target = AttentionPooling(name="lang_vector")(
        [embedded, scores_target, lang_input]
    )  # [EMBED_DIM]

    # Vision CNN — data-driven from CONFIG.model.conv; plain relu convs.
    vision_input = L.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="vision_pixels")
    v = vision_input
    for i, layer in enumerate(CONFIG.model.conv):
        v = L.Conv2D(
            filters=layer.filters,
            kernel_size=layer.kernel,
            strides=layer.stride or 1,
            padding=layer.padding or "same",
            activation="relu",
            name=f"conv{i + 1}",
        )(v)
        if layer.pool:
            v = L.MaxPooling2D(pool_size=2)(v)

    # ── language-conditioned spatial attention + soft-argmax readout ──────────
    G = ATTN_GRID
    C = CONFIG.model.conv[-1].filters
    # the G×G map as a sequence of cells, row-major (i*G + j), features last
    cells = L.Reshape((G * G, C), name="vision_cells")(v)  # [B, G*G, C]
    # ONE language-conditioned query over feature space — the attention map. The
    # carry flag rides into the query: "block at the effector" changes what the
    # commanded block looks like. The kernel is rescaled by 1/√C post-build so
    # the softmax starts soft.
    pick_query = L.Dense(C, name="pick_query")(
        L.Concatenate(name="pick_query_in")([lang_target, carry_input])
    )  # [B, C]
    # per-cell match scores → spatial softmax ("where the model looks")
    pick_map = L.Activation("softmax", name="pick_map")(
        L.Dot(axes=(2, 1), name="pick_cell_scores")([cells, pick_query])
    )  # [B, G*G]
    # soft-argmax: expected (x, y) coordinate under the map. A Dense with a
    # FROZEN kernel holding each cell's center (seeded post-build) IS that
    # expectation — no custom layer needed, gradients still flow through the map.
    pick_xy = L.Dense(2, use_bias=False, trainable=False, name="pick_grid")(
        pick_map
    )  # [B, 2], gained image coords
    # attention-weighted feature readout: block size / local shape at the spot.
    pick_feat = L.Dot(axes=(1, 1), name="pick_read")([pick_map, cells])  # [B, C]

    # fusion + heads.
    fused = L.Concatenate(name="fusion_in")([pick_xy, pick_feat, lang_target, carry_input])
    dense1 = L.Dense(CONFIG.model.fusionUnits, activation="relu", name="fusion")(fused)
    action_output = L.Dense(2, activation="linear", name="action")(dense1)
    color_output = L.Dense(len(COLORS), activation="softmax", name="color")(lang_target)
    # gripper COMMAND: a learned "close now" head (0=open → 1=closed). It reads
    # the fused hidden dense1 — being "over the block" is a VISUAL fact. Trained
    # by BCE against the shared effector_over_block predicate. Kept OUT of the
    # lang twin (that graph has no vision node → no "over the block" fact).
    gripper_output = L.Dense(1, activation="sigmoid", name="gripper")(dense1)

    model = tf.keras.Model(
        inputs=[vision_input, lang_input, carry_input],
        outputs=[action_output, color_output, pick_map, gripper_output],
        name="vla",
    )
    viz = tf.keras.Model(
        inputs=[vision_input, lang_input, carry_input],
        outputs=[action_output, pick_map, gripper_output],
        name="vla_viz",
    )
    lang = tf.keras.Model(inputs=[lang_input], outputs=[color_output], name="vla_lang")

    # load the pretrained GloVe vectors into the (frozen) embedding table
    model.get_layer("text_embedding").set_weights(
        [np.asarray(embed_matrix, dtype=np.float32)]
    )

    # seed the frozen soft-argmax kernel: row i*G+j holds cell (i, j)'s center,
    # centered and gained — column 0 = x (j across), column 1 = y (i down).
    gain = CONFIG.model.attnCoordGain
    grid = np.zeros((G * G, 2), dtype=np.float32)
    for i in range(G):
        for j in range(G):
            grid[i * G + j, 0] = ((j + 0.5) / G - 0.5) * gain
            grid[i * G + j, 1] = ((i + 0.5) / G - 0.5) * gain
    model.get_layer("pick_grid").set_weights([grid])

    # temper the attention at init: scale the query kernel by 1/√C so the initial
    # cell scores are small and the softmax starts near-uniform. Only the INIT is
    # scaled; the kernel stays fully trainable.
    q = model.get_layer("pick_query")
    q_kernel, q_bias = q.get_weights()
    q.set_weights([q_kernel * (1.0 / math.sqrt(C)), q_bias])

    # Keras supports loss_weights natively (tfjs-layers didn't — hence its custom
    # weighted losses). action = Huber (the wrong-side outlier tail otherwise
    # dominates); color/map = categorical CE; gripper = binary CE. The map is
    # already a softmax and the label is a bilinear distribution → plain CE.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss=[
            tf.keras.losses.Huber(delta=ACTION_HUBER_DELTA),
            tf.keras.losses.CategoricalCrossentropy(),
            tf.keras.losses.CategoricalCrossentropy(),
            tf.keras.losses.BinaryCrossentropy(),
        ],
        loss_weights=[1.0, COLOR_LOSS_WEIGHT, MAP_LOSS_WEIGHT, GRIPPER_LOSS_WEIGHT],
    )
    # language warm-up twin: plain (unweighted) CE on the color head, its own Adam
    # so warm-up momentum doesn't touch the main optimizer's state.
    lang.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
    )

    return VLAModels(model=model, viz=viz, lang=lang)
