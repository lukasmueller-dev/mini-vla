# The behavioral-cloning training loop. Python port of js/src/trainer.core.ts,
# restructured from the browser's async rAF loop into a synchronous, headless
# `fit()` suitable for dev + wandb sweeps.
#
# Each batch synthesizes (scene layout, pose, command) states — random block
# placements, sentences from the slot grammar with word-dropout — renders each
# through the same silhouette pipeline the live rollout uses (render.py), labels
# it with the analytical-IK expert's ABSOLUTE target joint angles (plus color,
# attention map, and gripper), and runs one train_on_batch step. Grasping is a
# LEARNED action: a sigmoid gripper head is trained (BCE) to close exactly when
# the effector is fully over the commanded block — the shared effector_over_block
# predicate — and the closed-loop eval turns its rising edge into the grasp.
#
# The carry phase is policy-driven, so a carryFrac share of samples render the
# commanded block IN THE GRIPPER, set carry_flag=1, and flip the label to REST
# (bring the grasped block home). Labels do NOT depend on the (randomized)
# rendered pose except for the carried block's pixels — the network learns
# "given this scene, command and carry state, where does the arm end up".
#
# Training stops once the trailing-window action loss crosses the convergence
# threshold (or the max-batch fallback).

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .config import CONFIG
from .embeddings import embedding_matrix, load as load_embeddings
from .geometry import (
    REST,
    THETA1_RANGE,
    THETA2_RANGE,
    clamp,
    effector_over_block,
    fk,
    ik_to_x,
)
from .model import (
    ACTION_HUBER_DELTA,
    ATTN_GRID,
    IMG_SIZE,
    VLAModels,
    build_vla_model,
)
from .render import render_silhouette, to_model_input
from .task import (
    COLORS,
    LABEL_TOKEN_IDS,
    MAX_SEQ_LEN,
    PAD,
    UNK,
    Layout,
    block_of_color,
    random_layout,
    sample_command,
)
from .vocab_gen import EMBED_DIM

# All tuning knobs live in config.py; aliased to locals so the loop reads the same.
_T = CONFIG.trainer
BATCH_SIZE = _T.batchSize
RENDER_SIZE = _T.renderSize
NEAR_TARGET_FRAC = _T.nearTargetFrac
NEAR_TARGET_STD = _T.nearTargetStd
WORD_DROPOUT = _T.wordDropout
CARRY_FRAC = _T.carryFrac
GRASP_FRAC = _T.graspFrac
GRASP_JITTER_STD = _T.graspJitterStd
WARMUP_BATCHES = _T.warmupBatches
WARMUP_BATCH_SIZE = _T.warmupBatchSize
GRIP_RADIUS = CONFIG.gripper.radius
GRIP_THRESHOLD = CONFIG.gripper.threshold

CONVERGE_LOSS = _T.converge.loss
CONVERGE_WINDOW = _T.converge.window
CONVERGE_STREAK = _T.converge.streak
MIN_BATCHES = _T.converge.minBatches
MAX_BATCHES = _T.converge.maxBatches

N_COLORS = len(COLORS)
CELLS = ATTN_GRID * ATTN_GRID


@dataclass
class PredictResult:
    # Predicted ABSOLUTE target joint angles.
    target: tuple[float, float]
    # Spatial attention over the ATTN_GRID² cells, row-major, peak-normalized.
    attn: np.ndarray
    # The map's soft-argmax — where the model looks, in [0,1] image coords.
    xy: tuple[float, float]
    # The gripper head's sigmoid output (0=open → 1=closed).
    grip: float


@dataclass
class ProbeRow:
    batch: int
    buckets: dict  # mean Huber action loss per phase bucket ("reach"/"carry")
    colorAcc: float
    gripAcc: float


@dataclass
class Batch:
    n: int
    vis: np.ndarray
    lang: np.ndarray
    carryF: np.ndarray
    ysA: np.ndarray
    ysC: np.ndarray
    ysMPick: np.ndarray
    ysG: np.ndarray


def _gauss(std: float) -> float:
    """Roughly-Gaussian noise (sum of two uniforms), matching trainer.core."""
    return (random.random() + random.random() - 1) * std * 2


def _argmax_row(a: np.ndarray, row: int, width: int) -> int:
    best = 0
    base = row * width
    for k in range(1, width):
        if a[base + k] > a[base + best]:
            best = k
    return best


class VLATrainer:
    def __init__(self) -> None:
        self.models: Optional[VLAModels] = None
        self.status = "idle"
        self.loss = math.nan  # latest Huber action loss
        self.smooth_loss = math.nan  # trailing-window mean (convergence signal)
        self.initial_loss = math.nan
        self.loss_history: list[float] = []
        self.batches = 0
        self.probes: list[ProbeRow] = []
        # Probe cadence in batches; 0 (default) = off. Sweeps set it (~25).
        self.probe_every_n = 0
        self.probe_n = 24
        # Harness override of the converge.maxBatches fallback (None = use CONFIG).
        self.max_batches_override: Optional[int] = None

    @property
    def samples(self) -> int:
        return self.batches * BATCH_SIZE

    @property
    def ready(self) -> bool:
        return self.models is not None and self.batches > 0

    # ── batch synthesis ──────────────────────────────────────────────────────

    def _write_map_label(self, ys_m: np.ndarray, base: int, u: float, v: float) -> None:
        """Bilinear attention-map label over the (up to) 4 cells around (u, v) in
        the silhouette view's [0,1] coords. Soft on purpose — a hard one-hot would
        quantize the soft-argmax readout to cell centers."""
        g = ATTN_GRID
        cx = u * g - 0.5
        cy = v * g - 0.5
        j0 = math.floor(cx)
        i0 = math.floor(cy)
        fx = cx - j0
        fy = cy - i0
        for i, j, w in (
            (i0, j0, (1 - fy) * (1 - fx)),
            (i0, j0 + 1, (1 - fy) * fx),
            (i0 + 1, j0, fy * (1 - fx)),
            (i0 + 1, j0 + 1, fy * fx),
        ):
            if i < 0 or i >= g or j < 0 or j >= g or w == 0:
                continue
            ys_m[base + i * g + j] += w

    def _block_uv(self, x: float, size: float, rest: float = 0.0) -> tuple[float, float]:
        """A resting block's visual center in silhouette [0,1] (u, v) coords."""
        u = 0.5 + (x - 0.5) * CONFIG.render.sceneScale
        y_center = rest + (size * CONFIG.render.silBlockScale) / 2
        v = CONFIG.render.floorY - y_center * CONFIG.render.sceneScale
        return u, v

    def _effector_uv(self, a1: float, a2: float) -> tuple[float, float]:
        """The effector position in silhouette [0,1] coords — where a CARRIED
        block renders (the attention anchor for the carry phase)."""
        e = fk(a1, a2)
        u = 0.5 + (e["ex"] - 0.5) * CONFIG.render.sceneScale
        v = CONFIG.render.floorY - e["ey"] * CONFIG.render.sceneScale
        return u, v

    def synth_batch(self, n: int, force_carry: Optional[bool] = None) -> Batch:
        vis = np.zeros((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        lang = np.zeros((n, MAX_SEQ_LEN), dtype=np.int32)
        carry_f = np.zeros((n, 1), dtype=np.float32)
        ys_a = np.zeros((n, 2), dtype=np.float32)
        ys_c = np.zeros((n, N_COLORS), dtype=np.float32)
        ys_m = np.zeros((n, CELLS), dtype=np.float32)
        ys_g = np.zeros((n, 1), dtype=np.float32)
        ys_m_flat = ys_m.reshape(-1)

        for i in range(n):
            layout = random_layout()
            sentence = sample_command(layout)  # acted-on color is present in scene
            target = block_of_color(layout, sentence.color)
            trest = target.y or 0.0

            # carry-conditioned label: a CARRY_FRAC share render the commanded
            # block in the gripper and label the carry-phase target (REST).
            mid_carry = force_carry if force_carry is not None else random.random() < CARRY_FRAC
            # among empty-handed samples, a GRASP_FRAC share are "grasp-now"
            # positives posed tightly at the block's IK grasp pose.
            grasp_now = (not mid_carry) and random.random() < GRASP_FRAC
            carry_f[i, 0] = 1.0 if mid_carry else 0.0

            if not mid_carry:
                t1, t2 = ik_to_x(target.x, target.size, trest)  # reach: grasp point
            else:
                t1, t2 = REST  # carrying → bring it home

            if grasp_now:
                a1 = clamp(t1 + _gauss(GRASP_JITTER_STD), THETA1_RANGE[0], THETA1_RANGE[1])
                a2 = clamp(t2 + _gauss(GRASP_JITTER_STD), THETA2_RANGE[0], THETA2_RANGE[1])
            elif random.random() < NEAR_TARGET_FRAC:
                a1 = clamp(t1 + _gauss(NEAR_TARGET_STD), THETA1_RANGE[0], THETA1_RANGE[1])
                a2 = clamp(t2 + _gauss(NEAR_TARGET_STD), THETA2_RANGE[0], THETA2_RANGE[1])
            else:
                a1 = THETA1_RANGE[0] + random.random() * (THETA1_RANGE[1] - THETA1_RANGE[0])
                a2 = THETA2_RANGE[0] + random.random() * (THETA2_RANGE[1] - THETA2_RANGE[0])

            # THE gripper label (the shared invariant): close iff already carrying
            # or the effector is fully over the commanded block. Identical
            # predicate to the rollout grasp gate.
            ys_g[i, 0] = 1.0 if (mid_carry or effector_over_block(a1, a2, target.as_block(), GRIP_RADIUS)) else 0.0

            # ABSOLUTE target joint angles (not a delta from the sampled pose).
            ys_a[i, 0] = t1
            ys_a[i, 1] = t2
            ys_c[i, sentence.color] = 1.0
            # attention supervision (phase-INDEPENDENT): the commanded block
            # wherever it renders (rest spot, or the effector while carried).
            if mid_carry:
                pu, pv = self._effector_uv(a1, a2)
            else:
                pu, pv = self._block_uv(target.x, target.size, trest)
            self._write_map_label(ys_m_flat, i * CELLS, pu, pv)

            rgb = render_silhouette(a1, a2, layout, sentence.color if mid_carry else None)
            vis[i] = to_model_input(rgb)

            # word-dropout: tokens that don't carry label information occasionally
            # become <unk> so the encoder learns to handle unknown words.
            for s in range(MAX_SEQ_LEN):
                tok = sentence.tokens[s]
                if tok != PAD and tok not in LABEL_TOKEN_IDS and random.random() < WORD_DROPOUT:
                    tok = UNK
                lang[i, s] = tok

        return Batch(n=n, vis=vis, lang=lang, carryF=carry_f, ysA=ys_a, ysC=ys_c, ysMPick=ys_m, ysG=ys_g)

    def synth_lang_batch(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """A CHEAP language-only batch: sentences + color label, NO vision/IK/map.
        Feeds language_warmup — same command + word-dropout distribution."""
        lang = np.zeros((n, MAX_SEQ_LEN), dtype=np.int32)
        ys_c = np.zeros((n, N_COLORS), dtype=np.float32)
        for i in range(n):
            layout = random_layout()
            sentence = sample_command(layout)
            ys_c[i, sentence.color] = 1.0
            for s in range(MAX_SEQ_LEN):
                tok = sentence.tokens[s]
                if tok != PAD and tok not in LABEL_TOKEN_IDS and random.random() < WORD_DROPOUT:
                    tok = UNK
                lang[i, s] = tok
        return lang, ys_c

    # ── training ─────────────────────────────────────────────────────────────

    def build(self, embed_matrix: Optional[np.ndarray] = None) -> None:
        if embed_matrix is None:
            embed_matrix = load_embeddings()
        self.models = build_vla_model(embed_matrix)

    def language_warmup(self) -> None:
        """WARMUP_BATCHES text-only gradient steps on the language twin, run
        before the main loop. Trains ONLY the color head + conv scorer, so the
        attention query starts against a clean language slot. Early-stops once the
        head's trailing loss falls under 10% of its initial value."""
        assert self.models is not None
        initial = math.nan
        recent: list[float] = []
        for k in range(WARMUP_BATCHES):
            lang, ys_c = self.synth_lang_batch(WARMUP_BATCH_SIZE)
            h = self.models.lang.train_on_batch(lang, ys_c)
            loss = float(h[0] if isinstance(h, (list, tuple)) else h)
            if math.isnan(initial):
                initial = loss
            recent.append(loss)
            if len(recent) > 10:
                recent.pop(0)
            mean = sum(recent) / len(recent)
            if k >= 30 and mean < 0.1 * initial:
                return

    def _action_loss(self, result) -> float:
        """Extract the unweighted Huber action loss from a train_on_batch dict —
        the convergence signal (matches trainer.core's loss index 1)."""
        if isinstance(result, dict):
            for key in ("action_loss", "output_1_loss"):
                if key in result:
                    return float(result[key])
            # fall back to the first *_loss that isn't the aggregate
            for k, v in result.items():
                if k.startswith("action"):
                    return float(v)
        # list form: [total, action, color, map, grip]
        return float(result[1])

    def train_step(self) -> float:
        """One gradient step on a freshly synthesized micro-batch. Returns the
        batch's Huber action loss."""
        assert self.models is not None
        b = self.synth_batch(BATCH_SIZE)
        result = self.models.model.train_on_batch(
            [b.vis, b.lang, b.carryF],
            [b.ysA, b.ysC, b.ysMPick, b.ysG],
            return_dict=True,
        )
        return self._action_loss(result)

    @staticmethod
    def _huber(e: float) -> float:
        d = ACTION_HUBER_DELTA
        a = abs(e)
        return 0.5 * a * a if a <= d else d * (a - 0.5 * d)

    def run_probe(self) -> None:
        """Held-out per-phase evaluation (forward passes only). Synthesizes
        probe_n fresh samples per bucket (reach / carry), reads mean Huber action
        loss per bucket + color/gripper head accuracy, appends one ProbeRow."""
        assert self.models is not None
        buckets: dict[str, float] = {}
        head_n = 0
        color_hits = 0
        grip_hits = 0
        for carry in (False, True):
            b = self.synth_batch(self.probe_n, force_carry=carry)
            action, color, _map, grip = self.models.model(
                [b.vis, b.lang, b.carryF], training=False
            )
            action = action.numpy().reshape(-1)
            color = color.numpy().reshape(-1)
            grip = grip.numpy().reshape(-1)
            ys_a = b.ysA.reshape(-1)
            ys_c = b.ysC.reshape(-1)
            ys_g = b.ysG.reshape(-1)
            s = 0.0
            for k in range(b.n * 2):
                s += self._huber(float(action[k] - ys_a[k]))
            buckets["carry" if carry else "reach"] = s / (b.n * 2)
            for i in range(b.n):
                if _argmax_row(color, i, N_COLORS) == _argmax_row(ys_c, i, N_COLORS):
                    color_hits += 1
                if (1 if grip[i] >= GRIP_THRESHOLD else 0) == int(ys_g[i]):
                    grip_hits += 1
            head_n += b.n
        self.probes.append(
            ProbeRow(batch=self.batches, buckets=buckets, colorAcc=color_hits / head_n, gripAcc=grip_hits / head_n)
        )

    def fit(self, on_update: Optional[Callable[["VLATrainer"], None]] = None) -> None:
        """Build (if needed), warm up the language head, then run batches until
        convergence or the max-batch fallback. `on_update(self)` fires after every
        batch (wandb logging hook)."""
        if self.models is None:
            self.status = "loading"
            self.build()
        self.status = "loading"
        self.language_warmup()

        self.status = "training"
        converge_streak = 0
        while True:
            loss = self.train_step()
            self.loss = loss
            if math.isnan(self.initial_loss):
                self.initial_loss = loss
            self.loss_history.append(loss)
            self.batches += 1
            window = self.loss_history[-CONVERGE_WINDOW:]
            self.smooth_loss = sum(window) / len(window)

            if self.probe_every_n > 0 and self.batches % self.probe_every_n == 0:
                self.run_probe()

            if on_update is not None:
                on_update(self)

            if self.batches >= MIN_BATCHES and self.smooth_loss < CONVERGE_LOSS:
                converge_streak += 1
            else:
                converge_streak = 0
            max_batches = self.max_batches_override if self.max_batches_override is not None else MAX_BATCHES
            if converge_streak >= CONVERGE_STREAK or self.batches >= max_batches:
                self.status = "converged"
                if on_update is not None:
                    on_update(self)
                return

    # ── inference (used by the closed-loop eval) ─────────────────────────────

    def predict_target(
        self,
        a1: float,
        a2: float,
        tokens: list[int],
        layout: Layout,
        carry: Optional[int] = None,
    ) -> Optional[PredictResult]:
        """Render the state, run the viz twin, return the predicted ABSOLUTE
        target joint angles + the spatial-attention readout (one forward pass)."""
        if not self.ready:
            return None
        assert self.models is not None
        rgb = render_silhouette(a1, a2, layout, carry)
        v = to_model_input(rgb)[np.newaxis, ...]
        l = np.asarray([tokens], dtype=np.int32)
        c = np.asarray([[1.0 if carry is not None else 0.0]], dtype=np.float32)
        action, pick, grip = self.models.viz([v, l, c], training=False)
        action = action.numpy().reshape(-1)
        attn = pick.numpy().reshape(-1)
        grip = float(grip.numpy().reshape(-1)[0])

        # the UI's gaze point: the map's expectation in [0,1] image coords, from
        # the RAW softmax map (before the peak normalization below).
        g = ATTN_GRID
        ux = vy = peak = 0.0
        for idx in range(attn.shape[0]):
            w = float(attn[idx])
            ux += w * ((idx % g) + 0.5)
            vy += w * ((idx // g) + 0.5)
            if w > peak:
                peak = w
        inv = 1.0 / peak if peak > 0 else 0.0
        return PredictResult(
            target=(float(action[0]), float(action[1])),
            attn=attn * inv,
            xy=(ux / g, vy / g),
            grip=grip,
        )

    def decode_command(self, tokens: list[int]) -> Optional[dict]:
        """Decode the acted-on color from a token sequence via the color head
        (vision zeroed)."""
        if not self.ready:
            return None
        assert self.models is not None
        v = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        l = np.asarray([tokens], dtype=np.int32)
        c = np.zeros((1, 1), dtype=np.float32)
        _action, color, _map, _grip = self.models.model([v, l, c], training=False)
        color = color.numpy().reshape(-1)
        idx = int(np.argmax(color))
        return {"color": idx, "colorProb": float(color[idx])}

    def attention_weights(self, tokens: list[int]) -> Optional[list[float]]:
        """The language encoder's per-token ATTENTION weights, recomputed CPU-side
        from the linear conv scorer + the frozen embedding table (no forward
        pass). Normalized so the most-attended token fills its bar."""
        if not self.ready:
            return None
        embed = embedding_matrix()
        if embed is None:
            return None
        assert self.models is not None
        w = self.models.model.get_layer("attn_score").get_weights()
        kernel = w[0]  # [K, EMBED_DIM, 1]
        bias = float(w[1][0])
        kk = kernel.shape[0]
        off = kk // 2  # conv1d "same": window i-off .. i+off
        scores: list[float] = []
        for i, tok in enumerate(tokens):
            if tok == PAD:
                scores.append(-math.inf)
                continue
            sc = bias
            for k in range(kk):
                j = i + k - off
                if j < 0 or j >= len(tokens):
                    continue
                t = tokens[j]
                if t == PAD:
                    continue
                sc += float(np.dot(embed[t], kernel[k, :, 0]))
            scores.append(sc)
        finite = [s for s in scores if math.isfinite(s)]
        max_s = max(finite) if finite else 0.0
        exps = []
        total = 0.0
        for s in scores:
            if not math.isfinite(s):
                exps.append(0.0)
                continue
            e = math.exp(s - max_s)
            total += e
            exps.append(e)
        weights = [(e / total if total > 0 else 0.0) for e in exps]
        max_w = max(max(weights, default=0.0), 1e-6)
        return [wt / max_w for wt in weights]

    def loss_norm(self) -> float:
        """Loss normalized against the first batch, clamped to [0, 1]."""
        if math.isnan(self.loss) or math.isnan(self.initial_loss):
            return 1.0
        if self.initial_loss <= 0:
            return 0.0
        return max(0.0, min(1.0, self.loss / self.initial_loss))
