# S4705 — Contexte HDC / Mahalanobis / TinyOL (N/A honnête)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 47 |
| **Priorité** | 🟠 Importante — cadre pourquoi le périmètre est EWC-only, sans fabriquer de cellule artificielle. |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 2h |
| **Dépendances** | S4701 (mapping) · Sprint 34 ✅ (Q15 Maha) · Sprint 28 ✅ (HDC INT8) |
| **Fichiers cibles** | `experiments/exp_S47_context/context.json` |
| **Références** | `docs/context/quantization_strategies.md` §5 (`q15`), §6 (`int16_am`) |

---

## Contexte

Le sweep profondeur/schéma (S4703/S4704) est **EWC-only** (décision utilisateur). Cette tâche documente
**pourquoi** les trois autres modèles ne sont pas balayés en bits, avec des cellules explicitement **N/A
justifiées** — jamais un chiffre artificiel. Elle produit un petit `context.json` traçable (pas de résultat, du
cadrage structuré) consommé par la figure de synthèse S4706.

## Spec

### Mapping N/A justifié

| Modèle | Statut axe profondeur | Justification (traçable) |
|--------|:---:|--------------------------|
| **HDC** | N/A structurel | Nativement entier : hypervecteurs ±1 (int8), mémoire associative int16 (`int16_am`, S4202 §6). Aucun **scale de poids** à réduire → la « profondeur » est fixée par la structure, pas un continuum. Métrique INT8 ≡ FP32 par construction (Δ = 0, Sprint 28). |
| **Mahalanobis** | N/A format-only | Détecteur **sans poids appris par gradient** (fit statistique μ, Σ⁻¹). Axe pertinent = **format de Σ⁻¹** : INT8 casse (grande dynamique), **Q15 récupère** (AUROC Pronostia −0,113 → +0,013, Sprint 34). Pas de tête neuronale à balayer en bits. |
| **TinyOL** | N/A hors-périmètre | Tête entraînable → un axe de profondeur **serait** exerçable, mais l'utilisateur a fixé le périmètre EWC-only pour ce sprint. Renvoi : travail futur possible (`TODO(arnaud)`). |

### Sortie

`experiments/exp_S47_context/context.json` :

```json
{
  "sprint": 47,
  "swept_models": ["ewc"],
  "context_models": {
    "hdc":         {"status": "na_structural",   "reason": "int16_am, pas de scale de poids", "ref": "S4202§6, S28"},
    "mahalanobis": {"status": "na_format_only",  "reason": "axe = INT8 vs Q15 de sigma_inv",  "ref": "S34"},
    "tinyol":      {"status": "na_out_of_scope",  "reason": "EWC-only (décision utilisateur)",  "ref": "S4700"}
  }
}
```

Aucun champ métrique (cadrage pur).

## Contraintes

- **Aucune cellule sub-INT8 fabriquée** pour HDC/Maha/TinyOL — le N/A est un résultat de cadrage, pas un vide.
- Chaque N/A porte une **justification et une référence** (sprint/section).

## Vérification

```bash
test -f experiments/exp_S47_context/context.json
python -c "import json; d=json.load(open('experiments/exp_S47_context/context.json')); assert d['swept_models']==['ewc']; assert all(v['status'].startswith('na_') for v in d['context_models'].values())"
```

---

## Résolution (implémentée)

_À compléter lors de l'implémentation._
