# S2408 — Tests non-régression + Documentation finale

| Champ | Valeur |
|-------|--------|
| **Sprint** | 24 |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Terminé |
| **Durée estimée** | S2408a : 30 min / S2408b : 30 min = 1h total |
| **Dépendances** | S2402a ✅ (EWC modifié), S2403a ✅ (export_onnx.py modifié), S2404a ✅ (profile_memory.py modifié) |
| **Fichiers cibles** | `tests/`, `docs/roadmap_phase2.md` |
| **Référence** | Sprint 22 S2223 (pattern tests + docs fin de sprint) |

---

## S2408a — Tests non-régression

### Objectif

Vérifier que les modifications de Sprint 24 (ajout `uint8_activations` dans EWC, extension `export_onnx.py`, extension `profile_memory.py`) n'introduisent pas de régression dans la suite de tests existante.

### Commandes

```bash
# Suite complète
pytest tests/ -v

# Tests spécifiques aux modules modifiés
pytest tests/test_ewc*.py -v           # EWC modifié (S2402a)
pytest tests/test_quantization*.py -v  # quantization.py utilisé par EWC (S2402a)
pytest tests/test_hdc*.py -v           # HDC modifié (S2402c)

# Vérifier que le flag uint8_activations n'est pas cassant en mode FP32 (backward compat)
python -c "
from src.models.ewc.ewc_mlp import EWCMlpClassifier
import torch
# Mode FP32 standard — doit fonctionner exactement comme avant Sprint 24
model = EWCMlpClassifier(input_dim=5, hidden_dims=[32, 16], output_dim=1)
assert not model.uint8_activations, 'uint8_activations doit être False par défaut'
x = torch.randn(4, 5)
out = model(x)
assert out.shape == (4, 1), f'Shape incorrecte: {out.shape}'
print('EWC backward compat OK')
"
```

### Critères de validation

- `pytest tests/ -v` : 0 régression (≥ 427 tests passent, ≥ 12 skipped) ✓
- `ruff check src/` : 0 erreur ✓
- `black --check src/` : 0 erreur ✓
- EWC en mode FP32 standard (sans `uint8_activations`) produit les mêmes outputs qu'avant Sprint 24 ✓

---

## S2408b — Roadmap update

### Mise à jour `docs/roadmap_phase2.md`

Marquer Sprint 24 comme complété et mettre à jour les statuts des Gaps :

```markdown
## Sprint 24 — Rétro-application Sprint 4 + Notebook comparatif final
**Statut** : ✅ Terminé
**Livrables** :
- 12 expériences exp_S24_01 à exp_S24_12 avec profiling unifié
- `experiments/sprint24_memory_report.json` — profiling 4 modèles × 5 datasets
- `experiments/onnx_sprint24/` — 20 fichiers ONNX valides
- `scripts/compare_all_sprints.py` — agrégateur Sprint 1–24
- `notebooks/24_comprehensive_comparison.ipynb` — notebook comparatif final

## Statut Triple Gap (mis à jour Sprint 24)
| Gap | Statut | Évidence |
|-----|--------|---------|
| Gap 1 | ✅ Comblé | 5 datasets industriels, acc > 0.85 |
| Gap 2 | ✅ Comblé | RAM max 22.4 Ko (TinyOL) < 256 Ko sur tous datasets |
| Gap 3 | ⚠️ Partiel | UINT8 forward-only validé (EWC+HDC+TinyOL), backprop FP32 |
```

### Mise à jour `S2401_analyse_improvements.md`

Cocher les trous comblés dans la matrice des améliorations Sprint 4 (passer ❌ → ✅ pour chaque entrée complétée).

---

## Bilan Sprint 24

À compléter à la fin du sprint :

| Tâche | Statut | Temps réel | Notes |
|-------|:------:|:----------:|-------|
| S2401 (analyse) | ⬜ | — | — |
| S2402a (EWC UINT8) | ⬜ | — | — |
| S2402b (exp_S24_01) | ⬜ | — | — |
| S2402c (HDC profile) | ⬜ | — | — |
| S2402d (exp_S24_02) | ⬜ | — | — |
| S2403a (ONNX étendu) | ⬜ | — | — |
| S2403b (20 ONNX) | ⬜ | — | — |
| S2404a (profiling --all) | ⬜ | — | — |
| S2404b (exp_S24_03) | ⬜ | — | — |
| S2405a (CWRU 4 modèles) | ⬜ | — | — |
| S2405b (Pronostia 2 modèles) | ⬜ | — | — |
| S2405c (Pump 3 modèles) | ⬜ | — | — |
| S2406 (agrégation) | ⬜ | — | — |
| S2407 (notebook) | ✅ | 1h | `24_comprehensive_comparison.ipynb` — 7 sections, 4 figures manuscrit |
| S2408a (tests) | ✅ | 10min | 456 passed · 12 skipped · 2 failed pre-existants (test_board_recorder EWC) · ruff clean |
| S2408b (roadmap) | ✅ | 10min | roadmap_phase2.md Sprint 24 ✅ TERMINÉ · Triple Gap table mise à jour |

**Résultats clés** :
- EWC UINT8 Δ acc : 0.000 (AA=0.911 FP32 = AA=0.911 UINT8)
- RAM max TinyOL (5 datasets) : 22.4 Ko
- Compression ratio UINT8 EWC : 4.0× (2 820 B → 705 B)
- Gap 2 : gap2_compliant=True sur toutes combinaisons mesurées

**Reporté au sprint suivant** : S4-08 `CONTRIBUTING.md` + `LICENSE` (déjà repoussé depuis Sprint 4)

**Questions pour encadrants** :
- `TODO(arnaud)` : Δ acc EWC UINT8 = X — acceptable pour Gap 3 manuscrit ?
- `TODO(dorra)` : Calibration UINT8 à chaque tâche ou initiale uniquement ?
