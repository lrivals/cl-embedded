"""
int8_c_emulation.py — Émulateur bit-exact du chemin C INT8 de la tête EWC (S3902).

Reproduit en NumPy, **sans carte**, le forward INT8 du firmware
``firmware/stm32f4_blink/src/ewc_head_int8.c`` (``ewc_int8_forward``), afin de
diagnostiquer au PC la perte d'accuracy/F1 observée sur board (Sprint 29/36) et de
balayer des variantes de quantification (accumulateur int32, scales calibrés
par-canal, activations Q15) **avant** tout flash.

Le chemin de référence (`QuantConfig.legacy_c()`) est **bit-à-bit identique** au C :
    - poids  : ``q = clip(trunc(w * 128), -127, 127)``        (cf. ``ewc_int8_from_fp32``)
    - entrée : ``q = wrap_int8(trunc(x * 128))``               (cf. ``float_to_q7``)
    - MAC    : accumulateur ``int16`` qui **wrap** à chaque ajout (overflow latent F1)
    - déquant: ``val = (acc >> 7) / 128 + bias``               (Q14 → Q7, biais FP32)
    - ReLU   : ``h = max(wrap_int8(trunc(val * 128)), 0)``     (activations Q7 clampées)

Les variantes (per-canal, int32, Q15) utilisent un chemin « calibré » : quantification
arrondie (round), accumulation int32 sans wrap, déquantification exacte
``acc * scale_w[j] * scale_act``.

Référence : S3901 (audit), ``ewc_head_int8.c``, ``ewc_mlp_multiclass.py`` (tête board).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# Types de schéma -------------------------------------------------------------
WeightScale = Literal["fixed_128", "per_tensor", "per_channel"]
ActRepr = Literal["q7_fixed", "q7_calib", "q15"]
AccDtype = Literal["int16", "int32"]


# ── Primitives entières façon C ──────────────────────────────────────────────

def _trunc_to_int(x: np.ndarray) -> np.ndarray:
    """Cast C ``(int)`` = troncature vers zéro (≠ floor pour négatifs)."""
    return np.trunc(x).astype(np.int64)


def _wrap_int8(x: np.ndarray) -> np.ndarray:
    """Cast C ``(int8_t)`` = wrap modulo 256 dans [-128, 127]."""
    return ((x.astype(np.int64) + 128) & 0xFF) - 128


def _wrap_int16(x: np.ndarray | int) -> np.ndarray | int:
    """Stockage C ``int16_t`` = wrap modulo 65536 dans [-32768, 32767]."""
    return ((np.int64(x) + 0x8000) & 0xFFFF) - 0x8000


def _sat8(x: np.ndarray) -> np.ndarray:
    """Macro C ``SAT8`` = saturation dans [-127, 127]."""
    return np.clip(x, -127, 127).astype(np.int64)


# ── Configuration d'un schéma de quantification ──────────────────────────────

@dataclass(frozen=True)
class QuantConfig:
    """Décrit un schéma de quantification émulé.

    Parameters
    ----------
    weight_scale : "fixed_128" | "per_tensor" | "per_channel"
        ``fixed_128`` = échelle 1/128 figée (firmware actuel). ``per_tensor`` =
        max|W| calibré global par couche. ``per_channel`` = max|W| par neurone de
        sortie (comme le QAT PC ``PerChannelMinMaxObserver``).
    act_repr : "q7_fixed" | "q7_calib" | "q15"
        Représentation des activations. ``q7_fixed`` = scale 1/128 (clamp ReLU>1).
        ``q7_calib`` = 8-bit calibré. ``q15`` = 16-bit (évite le clamp, fidélité 256×).
    acc_dtype : "int16" | "int32"
        Accumulateur MAC. ``int16`` reproduit l'overflow latent du firmware.
    name : str
        Étiquette lisible (clé de résultat).
    """

    weight_scale: WeightScale = "per_channel"
    act_repr: ActRepr = "q15"
    acc_dtype: AccDtype = "int32"
    name: str = "custom"

    @staticmethod
    def legacy_c() -> "QuantConfig":
        """Chemin firmware actuel (bit-exact ``ewc_head_int8.c``)."""
        return QuantConfig("fixed_128", "q7_fixed", "int16", name="legacy_c")

    @staticmethod
    def fix_acc32() -> "QuantConfig":
        """Legacy + accumulateur int32 (isole l'effet de l'overflow)."""
        return QuantConfig("fixed_128", "q7_fixed", "int32", name="fix_acc32")

    @staticmethod
    def per_tensor_calib() -> "QuantConfig":
        """Scales par-tenseur calibrés + int32 (isole l'effet du 1/128 figé)."""
        return QuantConfig("per_tensor", "q7_calib", "int32", name="per_tensor_calib")

    @staticmethod
    def per_channel_int8() -> "QuantConfig":
        """INT8 par-canal calibré (cible la cause racine ; mirroir QAT PC)."""
        return QuantConfig("per_channel", "q7_calib", "int32", name="per_channel_int8")

    @staticmethod
    def q15() -> "QuantConfig":
        """Q15 16-bit poids+activations par-canal."""
        return QuantConfig("per_channel", "q15", "int32", name="q15")

    @staticmethod
    def mixed_int8w_q15act() -> "QuantConfig":
        """Poids INT8 par-canal (RAM) + activations Q15 (évite clamp Q7)."""
        return QuantConfig("per_channel", "q15", "int32", name="mixed_int8w_q15act")


# Ordre canonique d'ablation : du firmware actuel au schéma idéal ------------
ABLATION_LADDER: list[QuantConfig] = [
    QuantConfig.legacy_c(),
    QuantConfig.fix_acc32(),
    QuantConfig.per_tensor_calib(),
    QuantConfig.per_channel_int8(),
    QuantConfig.q15(),
]


# ── Tête EWC émulée ──────────────────────────────────────────────────────────

@dataclass
class EWCHeadWeights:
    """Poids FP32 d'une tête EWCMlpMulticlass (5→32→16→2) extraits du checkpoint.

    Les matrices suivent la convention PyTorch ``nn.Linear`` : ``W[out, in]``.
    """

    w1: np.ndarray  # [H1, IN]
    b1: np.ndarray  # [H1]
    w2: np.ndarray  # [H2, H1]
    b2: np.ndarray  # [H2]
    w3: np.ndarray  # [OUT, H2]
    b3: np.ndarray  # [OUT]

    @staticmethod
    def from_state_dict(sd: dict) -> "EWCHeadWeights":
        """Construit depuis un ``state_dict`` torch (``fc1.weight`` …)."""
        def arr(k: str) -> np.ndarray:
            v = sd[k]
            return v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)

        return EWCHeadWeights(
            w1=arr("fc1.weight"), b1=arr("fc1.bias"),
            w2=arr("fc2.weight"), b2=arr("fc2.bias"),
            w3=arr("fc3.weight"), b3=arr("fc3.bias"),
        )


# ── Quantification d'une couche ──────────────────────────────────────────────

def _weight_scales(w: np.ndarray, mode: WeightScale, n_bits: int) -> np.ndarray:
    """Retourne un vecteur de scales [n_out] (broadcast par ligne de sortie)."""
    qmax = (1 << (n_bits - 1)) - 1  # 127 (int8) / 32767 (q15)
    if mode == "fixed_128":
        return np.full(w.shape[0], 1.0 / 128.0, dtype=np.float64)
    if mode == "per_tensor":
        m = float(np.max(np.abs(w))) or 1.0
        return np.full(w.shape[0], m / qmax, dtype=np.float64)
    # per_channel : un scale par neurone de sortie (ligne de W)
    m = np.max(np.abs(w), axis=1)
    m[m == 0] = 1.0
    return (m / qmax).astype(np.float64)


def _quant_weight(w: np.ndarray, scales: np.ndarray, n_bits: int) -> np.ndarray:
    """Quantifie W[out,in] avec un scale par ligne (round, saturation symétrique)."""
    qmax = (1 << (n_bits - 1)) - 1
    q = np.round(w / scales[:, None])
    return np.clip(q, -qmax, qmax).astype(np.int64)


def _act_params(act_repr: ActRepr, calib_max: float) -> tuple[float, int]:
    """Retourne (scale_act, n_bits) pour la représentation d'activation choisie."""
    if act_repr == "q7_fixed":
        return 1.0 / 128.0, 8
    if act_repr == "q7_calib":
        m = calib_max or 1.0
        return m / 127.0, 8
    # q15
    m = calib_max or 1.0
    return m / 32767.0, 16


# ── Forward ──────────────────────────────────────────────────────────────────

def _layer_legacy_c(x_q7: np.ndarray, w_q7: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Couche bit-exacte C : acc int16 wrap, >>7, déquant 1/128, ReLU Q7.

    Retourne les activations **int8 Q7** (entiers, comme le C les propage).
    """
    n_out = w_q7.shape[0]
    out_q7 = np.zeros(n_out, dtype=np.int64)
    for j in range(n_out):
        acc = np.int64(0)
        for i in range(w_q7.shape[1]):
            prod = np.int64(w_q7[j, i]) * np.int64(x_q7[i])
            acc = _wrap_int16(acc + prod)            # ← stockage int16 (overflow latent)
        val = float(np.int64(acc) >> 7) / 128.0 + float(bias[j])  # déquant Q14→Q7 + biais FP32
        out_q7[j] = max(int(_wrap_int8(_trunc_to_int(np.array(val * 128.0)))), 0)  # relu_q7
    return out_q7


def _layer_legacy_c_logits(x_q7: np.ndarray, w_q7: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Couche de sortie C : pas de ReLU, logits FP32 (cf. couche 3 du C)."""
    n_out = w_q7.shape[0]
    logits = np.zeros(n_out, dtype=np.float64)
    for j in range(n_out):
        acc = np.int64(0)
        for i in range(w_q7.shape[1]):
            acc = _wrap_int16(acc + np.int64(w_q7[j, i]) * np.int64(x_q7[i]))
        logits[j] = float(np.int64(acc) >> 7) / 128.0 + float(bias[j])
    return logits


def _forward_legacy_c(w: EWCHeadWeights, x: np.ndarray) -> np.ndarray:
    """Forward bit-exact du firmware actuel pour un échantillon x[IN]. Retourne logits[OUT]."""
    # Quantification poids = SAT8(trunc(w*128)) (cf. ewc_int8_from_fp32)
    q1 = _sat8(_trunc_to_int(w.w1 * 128.0))
    q2 = _sat8(_trunc_to_int(w.w2 * 128.0))
    q3 = _sat8(_trunc_to_int(w.w3 * 128.0))
    # Quantification entrée = wrap_int8(trunc(x*128)) (cf. float_to_q7 hôte)
    x_q7 = _wrap_int8(_trunc_to_int(x * 128.0))
    h1 = _layer_legacy_c(x_q7, q1, w.b1)
    h2 = _layer_legacy_c(h1, q2, w.b2)
    return _layer_legacy_c_logits(h2, q3, w.b3)


def _forward_calibrated(
    w: EWCHeadWeights, x: np.ndarray, cfg: QuantConfig, act_max: dict[str, float]
) -> np.ndarray:
    """Forward « calibré » (variantes) : int32, scales par-canal/tenseur, Q7/Q15.

    ``act_max`` fournit les bornes calibrées des activations d'entrée de chaque couche
    (clés ``in``, ``h1``, ``h2``), estimées une fois sur un lot représentatif.
    """
    w_bits = 16 if cfg.act_repr == "q15" else 8  # poids Q15 si schéma 16-bit, sinon int8
    if cfg.name == "mixed_int8w_q15act":
        w_bits = 8  # poids int8, activations q15

    s1 = _weight_scales(w.w1, cfg.weight_scale, w_bits)
    s2 = _weight_scales(w.w2, cfg.weight_scale, w_bits)
    s3 = _weight_scales(w.w3, cfg.weight_scale, w_bits)
    q1 = _quant_weight(w.w1, s1, w_bits)
    q2 = _quant_weight(w.w2, s2, w_bits)
    q3 = _quant_weight(w.w3, s3, w_bits)

    def quant_act(a: np.ndarray, key: str) -> tuple[np.ndarray, float]:
        sa, nb = _act_params(cfg.act_repr, act_max[key])
        qmax = (1 << (nb - 1)) - 1
        q = np.clip(np.round(a / sa), -qmax, qmax).astype(np.int64)
        return q, sa

    def dense_relu(a_q: np.ndarray, sa: float, wq: np.ndarray, sw: np.ndarray,
                   bias: np.ndarray, relu: bool) -> np.ndarray:
        # a_q[N, in] @ wq[out, in]^T = acc[N, out] en int32 (pas de wrap)
        acc = a_q.astype(np.int64) @ wq.astype(np.int64).T
        val = acc.astype(np.float64) * (sw * sa)[None, :] + bias[None, :]  # déquant par-canal
        return np.maximum(val, 0.0) if relu else val

    a = np.atleast_2d(np.asarray(x, dtype=np.float64))
    xq, sx = quant_act(a, "in")
    h1 = dense_relu(xq, sx, q1, s1, w.b1, relu=True)
    h1q, s_h1 = quant_act(h1, "h1")
    h2 = dense_relu(h1q, s_h1, q2, s2, w.b2, relu=True)
    h2q, s_h2 = quant_act(h2, "h2")
    return dense_relu(h2q, s_h2, q3, s3, w.b3, relu=False)


def calibrate_activations(w: EWCHeadWeights, X: np.ndarray) -> dict[str, float]:
    """Estime les bornes max|activation| par couche sur un lot (FP32 de référence)."""
    a = np.asarray(X, dtype=np.float64)
    in_max = float(np.max(np.abs(a))) or 1.0
    h1 = np.maximum(a @ w.w1.T + w.b1, 0.0)
    h1_max = float(np.max(np.abs(h1))) or 1.0
    h2 = np.maximum(h1 @ w.w2.T + w.b2, 0.0)
    h2_max = float(np.max(np.abs(h2))) or 1.0
    return {"in": in_max, "h1": h1_max, "h2": h2_max}


def forward_fp32(w: EWCHeadWeights, X: np.ndarray) -> np.ndarray:
    """Forward FP32 de référence (logits) — vectorisé sur X[N, IN]."""
    a = np.asarray(X, dtype=np.float64)
    h1 = np.maximum(a @ w.w1.T + w.b1, 0.0)
    h2 = np.maximum(h1 @ w.w2.T + w.b2, 0.0)
    return h2 @ w.w3.T + w.b3


def forward_quant(
    w: EWCHeadWeights, X: np.ndarray, cfg: QuantConfig,
    act_max: dict[str, float] | None = None,
) -> np.ndarray:
    """Forward émulé pour un schéma donné. Retourne les logits [N, OUT].

    ``legacy_c`` est calculé échantillon par échantillon (sémantique entière C) ;
    les variantes utilisent le chemin calibré vectorisé.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if cfg.weight_scale == "fixed_128" and cfg.act_repr == "q7_fixed" and cfg.acc_dtype == "int16":
        return np.stack([_forward_legacy_c(w, x) for x in X])
    if cfg.weight_scale == "fixed_128" and cfg.act_repr == "q7_fixed":
        # variante fix_acc32 : même quantif que legacy mais accumulateur int32
        return np.stack([_forward_fixed_acc32(w, x) for x in X])
    if act_max is None:
        act_max = calibrate_activations(w, X)
    return _forward_calibrated(w, X, cfg, act_max)


def _forward_fixed_acc32(w: EWCHeadWeights, x: np.ndarray) -> np.ndarray:
    """Legacy C mais accumulateur int32 (pas de wrap) — isole l'overflow."""
    q1 = _sat8(_trunc_to_int(w.w1 * 128.0))
    q2 = _sat8(_trunc_to_int(w.w2 * 128.0))
    q3 = _sat8(_trunc_to_int(w.w3 * 128.0))
    x_q7 = _wrap_int8(_trunc_to_int(x * 128.0))

    def layer(x_q: np.ndarray, wq: np.ndarray, bias: np.ndarray, relu: bool) -> np.ndarray:
        acc = wq.astype(np.int64) @ x_q.astype(np.int64)        # int32, pas de wrap
        val = acc.astype(np.float64) / 16384.0 + bias            # (acc>>7)/128 = acc/128²
        if not relu:
            return val
        q = _wrap_int8(_trunc_to_int(val * 128.0))
        return np.maximum(q, 0).astype(np.int64)

    h1 = layer(x_q7, q1, w.b1, relu=True)
    h2 = layer(h1, q2, w.b2, relu=True)
    return layer(h2, q3, w.b3, relu=False)


# ── Métriques de comparaison ─────────────────────────────────────────────────

def predict(logits: np.ndarray) -> np.ndarray:
    """argmax → classe prédite [N]."""
    return np.argmax(np.atleast_2d(logits), axis=1)


def softmax_prob1(logits: np.ndarray) -> np.ndarray:
    """Probabilité de la classe 1 (pour AUROC) [N]."""
    z = np.atleast_2d(logits)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e[:, 1] / e.sum(axis=1))


def agreement(logits_a: np.ndarray, logits_b: np.ndarray) -> float:
    """Taux d'accord de prédiction entre deux schémas (proxy de parité board↔PC)."""
    return float(np.mean(predict(logits_a) == predict(logits_b)))
