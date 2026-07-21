#!/usr/bin/env python3
"""write_s47_context.py — Cadrage N/A honnête HDC/Maha/TinyOL (Sprint 47, S4705).

Le sweep profondeur/schéma (S4703/S4704) est **EWC-only** (décision utilisateur). Ce
script produit un petit `experiments/exp_S47_context/context.json` **traçable** (pas de
résultat, du cadrage structuré) qui documente **pourquoi** les trois autres modèles ne
sont pas balayés en bits — chaque cellule explicitement **N/A justifiée**, jamais un
chiffre artificiel. Consommé par la figure de synthèse `scope_context.png` (S4706).

Justifications (traçables) :
  - HDC        : nativement entier (hypervecteurs ±1 int8, mémoire associative int16 —
                 `int16_am`, S4202 §6). Aucun **scale de poids** à réduire → la « profondeur »
                 est fixée par la structure, pas un continuum. Métrique INT8 ≡ FP32 (Δ=0, S28).
  - Mahalanobis: détecteur **sans poids appris par gradient** (fit statistique μ, Σ⁻¹). Axe
                 pertinent = **format de Σ⁻¹** (INT8 casse — grande dynamique ; Q15 récupère,
                 AUROC Pronostia −0,113 → +0,013, S34). Pas de tête neuronale à balayer en bits.
  - TinyOL     : tête entraînable → un axe de profondeur **serait** exerçable, mais le périmètre
                 est fixé EWC-only pour ce sprint (`TODO(arnaud)` : travail futur possible).

Règle « aucun chiffre inventé » : le JSON ne porte **aucun champ métrique** (cadrage pur).

Usage :
    python scripts/write_s47_context.py
"""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = _ROOT / "experiments" / "exp_S47_context"

CONTEXT: dict = {
    "sprint": 47,
    "swept_models": ["ewc"],
    "context_models": {
        "hdc": {
            "status": "na_structural",
            "reason": "Nativement entier (hypervecteurs ±1 int8, mémoire associative "
                      "int16) — aucun scale de poids à réduire ; profondeur fixée par "
                      "la structure. Métrique INT8 ≡ FP32 par construction (Δ=0).",
            "ref": "S4202§6, S28",
        },
        "mahalanobis": {
            "status": "na_format_only",
            "reason": "Détecteur sans poids appris par gradient (fit statistique μ, "
                      "Σ⁻¹). Axe pertinent = format de Σ⁻¹ (INT8 casse, Q15 récupère) ; "
                      "pas de tête neuronale à balayer en bits.",
            "ref": "S34",
        },
        "tinyol": {
            "status": "na_out_of_scope",
            "reason": "Tête entraînable → axe de profondeur exerçable, mais périmètre "
                      "fixé EWC-only pour ce sprint (décision utilisateur).",
            "ref": "S4700",
        },
    },
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "context.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(CONTEXT, f, indent=2, ensure_ascii=False)
    print(f"  → {out}")
    # Sanity (miroir de la vérification S4705) : périmètre EWC-only, tous N/A justifiés.
    assert CONTEXT["swept_models"] == ["ewc"]
    assert all(v["status"].startswith("na_") for v in CONTEXT["context_models"].values())
    assert all(v.get("reason") and v.get("ref") for v in CONTEXT["context_models"].values())


if __name__ == "__main__":
    main()
