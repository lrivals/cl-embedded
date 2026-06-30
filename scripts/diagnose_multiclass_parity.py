#!/usr/bin/env python3
"""
diagnose_multiclass_parity.py — Diagnostic FIXME(gap1), Sprint 26 / S2611.

Reproduit, côté PC (sans board), les deux régimes de mesure F1-macro de la tête
EWC multi-classe portée sur NUCLEO-F439ZI :

    (a) inference-only  : forward/argmax sur les poids entraînés figés (aucun update)
    (b) online single-pass : forward/argmax PUIS ewc_mc_sgd_step à chaque échantillon
        (FLAG_UPDATE), exactement comme `simulate_multiclass_board.py` pilote le board.

La métrique board est un **F1-macro préquentiel cumulatif** (sur les 10 classes), calculé
sur le flux d'échantillons dans l'ordre envoyé — ce n'est PAS la F1 par-tâche hors-ligne
sur le val set (qui donnait 0.981 dans exp_S25_03). Ce script isole donc la cause du
0.507 board : (1) métrique préquentielle vs hors-ligne et (2) dérive du SGD online.

Le forward et le SGD sont une réimplémentation numpy **fidèle au C** (`ewc_head_multiclass.c`),
y compris la mise à jour in-place de w3/w2 utilisée dans le calcul du gradient amont
(lignes 156-157 / 173-174 du .c). On charge les poids depuis le checkpoint PyTorch entraîné
(`exp_S25_03/model_ewc_mc.pt`), identiques à ceux figés dans `model_weights_multiclass.h`.

Usage :
    python scripts/diagnose_multiclass_parity.py \\
        --checkpoint experiments/exp_S25_03/model_ewc_mc.pt \\
        --csv-path "data/raw/CWRU Bearing Dataset/feature_time_48k_2048_load_1.csv" \\
        --n-samples-per-task 100 \\
        [--lr 0.01] [--lambda-ewc 400.0] [--fisher-decay 0.99]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.data.cwru_dataset import get_cl_splits


class NumpyEWCMlpMulticlass:
    """Miroir numpy fidèle de ewc_head_multiclass.c (forward / sgd_step / consolidate)."""

    def __init__(self, w1, b1, w2, b2, w3, b3, lr: float, lam: float):
        # Conventions C : w1[H1][IN], w2[H2][H1], w3[N][H2] (= nn.Linear [out,in])
        self.w1 = w1.astype(np.float32).copy()
        self.b1 = b1.astype(np.float32).copy()
        self.w2 = w2.astype(np.float32).copy()
        self.b2 = b2.astype(np.float32).copy()
        self.w3 = w3.astype(np.float32).copy()
        self.b3 = b3.astype(np.float32).copy()
        self.fisher1 = np.zeros_like(self.w1)
        self.fisher2 = np.zeros_like(self.w2)
        self.fisher3 = np.zeros_like(self.w3)
        self.star_w1 = np.zeros_like(self.w1)
        self.star_w2 = np.zeros_like(self.w2)
        self.star_w3 = np.zeros_like(self.w3)
        self.lr = np.float32(lr)
        self.lam = np.float32(lam)

    def _forward(self, x):
        h1 = np.maximum(self.w1 @ x + self.b1, 0.0)
        h2 = np.maximum(self.w2 @ h1 + self.b2, 0.0)
        logits = self.w3 @ h2 + self.b3
        return h1, h2, logits

    def predict(self, x):
        _, _, logits = self._forward(x.astype(np.float32))
        return int(np.argmax(logits))

    def sgd_step(self, x, label: int):
        """Miroir exact de ewc_mc_sgd_step (in-place w3/w2 dans le grad amont)."""
        x = x.astype(np.float32)
        h1, h2, logits = self._forward(x)

        # Softmax stable + gradient CE : dout = softmax - one_hot(label)
        m = logits.max()
        e = np.exp(logits - m)
        dout = (e / e.sum()).astype(np.float32)
        dout[label] -= 1.0

        # Couche 3 : update w3 puis accumule dh2 avec w3 DÉJÀ mis à jour (cf .c:156-157)
        dh2 = np.zeros(self.w3.shape[1], dtype=np.float32)
        for j in range(self.w3.shape[0]):
            grad = dout[j] * h2 + self.lam * self.fisher3[j] * (self.w3[j] - self.star_w3[j])
            self.w3[j] -= self.lr * grad
            dh2 += self.w3[j] * dout[j]
            self.b3[j] -= self.lr * dout[j]
        dh2 *= (h2 > 0.0).astype(np.float32)

        # Couche 2 : idem, w2 in-place (cf .c:173-174)
        dh1 = np.zeros(self.w2.shape[1], dtype=np.float32)
        for j in range(self.w2.shape[0]):
            grad = dh2[j] * h1 + self.lam * self.fisher2[j] * (self.w2[j] - self.star_w2[j])
            self.w2[j] -= self.lr * grad
            dh1 += self.w2[j] * dh2[j]
            self.b2[j] -= self.lr * dh2[j]
        dh1 *= (h1 > 0.0).astype(np.float32)

        # Couche 1
        for j in range(self.w1.shape[0]):
            grad = dh1[j] * x + self.lam * self.fisher1[j] * (self.w1[j] - self.star_w1[j])
            self.w1[j] -= self.lr * grad
            self.b1[j] -= self.lr * dh1[j]

    def consolidate(self, alpha: float):
        a, oma = np.float32(alpha), np.float32(1.0 - alpha)
        self.fisher1 = a * self.fisher1 + oma * self.w1**2
        self.star_w1 = self.w1.copy()
        self.fisher2 = a * self.fisher2 + oma * self.w2**2
        self.star_w2 = self.w2.copy()
        self.fisher3 = a * self.fisher3 + oma * self.w3**2
        self.star_w3 = self.w3.copy()


def load_weights(checkpoint: Path):
    state = torch.load(checkpoint, map_location="cpu")
    if "model_state_dict" in state:
        state = state["model_state_dict"]

    def npy(key):
        return state[key].detach().cpu().float().numpy()

    return dict(
        w1=npy("fc1.weight"), b1=npy("fc1.bias"),
        w2=npy("fc2.weight"), b2=npy("fc2.bias"),
        w3=npy("fc3.weight"), b3=npy("fc3.bias"),
    )


def build_stream(csv_path: str, n_per_task: int):
    """Reproduit l'ordre d'échantillons de simulate_multiclass_board.py (X_train, premiers N)."""
    tasks = get_cl_splits(csv_path=csv_path, scenario="by_fault_type", mode="multiclass")
    stream = []  # (task_id, x, y, is_last_of_task)
    for tid, task in enumerate(tasks):
        X, y = task["X_train"], task["y_train"]
        n = min(n_per_task, len(X))
        for i in range(n):
            stream.append((tid, X[i].astype(np.float32), int(y[i]), i == n - 1))
    return tasks, stream


def prequential_f1(model: NumpyEWCMlpMulticlass, stream, *, update: bool, fisher_decay: float):
    """F1-macro cumulatif préquentiel : predict avant un éventuel update (comme le board)."""
    preds, trues = [], []
    per_task_cum = {}
    for tid, x, y, is_last in stream:
        preds.append(model.predict(x))
        trues.append(y)
        if update:
            model.sgd_step(x, y)
            if is_last:
                model.consolidate(fisher_decay)
        per_task_cum[tid] = f1_score(trues, preds, average="macro", zero_division=0)
    f1 = f1_score(trues, preds, average="macro", zero_division=0)
    return f1, per_task_cum


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnostic parité F1 multi-classe board (FIXME gap1)")
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("experiments/exp_S25_03/model_ewc_mc.pt"))
    ap.add_argument("--csv-path",
                    default="data/raw/CWRU Bearing Dataset/feature_time_48k_2048_load_1.csv")
    ap.add_argument("--n-samples-per-task", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lambda-ewc", type=float, default=400.0)
    ap.add_argument("--fisher-decay", type=float, default=0.99)
    args = ap.parse_args()

    w = load_weights(args.checkpoint)
    tasks, stream = build_stream(args.csv_path, args.n_samples_per_task)
    print(f"Stream : {len(stream)} échantillons, {len(tasks)} tâches "
          f"({args.n_samples_per_task}/tâche).")

    # (a) inference-only — poids figés
    m_inf = NumpyEWCMlpMulticlass(**w, lr=args.lr, lam=args.lambda_ewc)
    f1_inf, cum_inf = prequential_f1(m_inf, stream, update=False, fisher_decay=args.fisher_decay)

    # (b) online single-pass — reproduit le régime FLAG_UPDATE du board
    m_on = NumpyEWCMlpMulticlass(**w, lr=args.lr, lam=args.lambda_ewc)
    f1_on, cum_on = prequential_f1(m_on, stream, update=True, fisher_decay=args.fisher_decay)

    print("\n── F1-macro préquentiel cumulatif (poids entraînés exp_S25_03) ──")
    print(f"  (a) inference-only      : {f1_inf:.4f}")
    print(f"      par tâche (cumul)   : "
          + ", ".join(f"t{t}={cum_inf[t]:.3f}" for t in sorted(cum_inf)))
    print(f"  (b) online single-pass  : {f1_on:.4f}")
    print(f"      par tâche (cumul)   : "
          + ", ".join(f"t{t}={cum_on[t]:.3f}" for t in sorted(cum_on)))
    print("\nInterprétation :")
    print("  - (a) ≈ niveau attendu board en inférence pure (FLAG_UPDATE off) → valide poids/forward.")
    print("  - (b) ≈ F1 board online (~0.507) → confirme dérive SGD single-pass, pas un bug de mapping.")


if __name__ == "__main__":
    main()
