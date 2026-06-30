# S2205–S2209 — CMAPSS : Expériences CL PC (4 modèles)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 22 |
| **Priorité** | 🔴 (S2205, S2209) / 🟡 (S2206–S2208) |
| **Statut** | ✅ S2205–S2208 terminés (2026-05-28) — S2209 notebook à faire |
| **Durée estimée** | 1h + 1h + 1h + 30 min + 2h = 5h30 |
| **Dépendances** | S2201–S2204 ✅ (`cmapss_loader.py` + `cmapss_config.yaml` opérationnels) |
| **Fichiers cibles** | `experiments/exp_S22_01/` à `exp_S22_04/`, `notebooks/results_cmapss_cl.ipynb` |
| **Référence** | `scripts/train_ewc.py`, `train_hdc.py`, `train_tinyol.py`, `evaluation/metrics.py` |

---

## Contexte

Ces 4 expériences valident les 3 modèles principaux (EWC, HDC, TinyOL) + baseline (Mahalanobis) sur le dataset CMAPSS en scénario **domain-incremental FD001→FD002→FD003→FD004**. Elles constituent les résultats quantitatifs du manuscrit pour Gap 1 (diversité datasets industriels).

Prérequis : `cmapss_loader.py` doit être intégré dans le pipeline des scripts existants `scripts/train_*.py` via l'option `--config configs/cmapss_config.yaml`.

---

## S2205 — exp_S22_01 : EWC / CMAPSS

```bash
python scripts/train_ewc.py \
    --config configs/cmapss_config.yaml \
    --exp-id exp_S22_01 \
    --output experiments/exp_S22_01/
```

**Structure de sortie attendue** :
```
experiments/exp_S22_01/
├── config_snapshot.yaml    # copie complète de la config au moment de l'exp
├── results.json            # métriques CL complètes
├── training_curves.png     # courbes loss/acc par tâche
└── ewc_weights_final.pt    # poids après tâche 4 (FD004)
```

**`results.json` — format minimal attendu** :
```json
{
  "exp_id": "exp_S22_01",
  "model": "ewc",
  "dataset": "cmapss",
  "domain_order": ["FD001", "FD002", "FD003", "FD004"],
  "acc_final": 0.XX,
  "avg_forgetting": 0.XX,
  "backward_transfer": 0.XX,
  "auroc_final": 0.XX,
  "ram_peak_bytes": XXXX,
  "inference_latency_ms": X.X,
  "n_params": 769,
  "per_task_acc": [[...], [...], [...], [...]]
}
```

**Métriques attendues** (critères manuscrit) :

| Métrique | Valeur cible |
|----------|:------------:|
| `acc_final` | ≥ 0.75 |
| `avg_forgetting` | ≤ 0.15 |

---

## S2206 — exp_S22_02 : HDC / CMAPSS

```bash
python scripts/train_hdc.py \
    --config configs/cmapss_config.yaml \
    --exp-id exp_S22_02 \
    --output experiments/exp_S22_02/
```

**Métriques attendues** :

| Métrique | Valeur cible |
|----------|:------------:|
| `acc_final` | ≥ 0.65 |
| `avg_forgetting` | ≤ 0.20 |

**Note** : HDC est non-neuronal — pas de gradient. Sa stabilité naturelle (mise à jour additive des vecteurs classe) peut être avantageuse sur ce scénario 4 tâches.

---

## S2207 — exp_S22_03 : TinyOL / CMAPSS

```bash
python scripts/train_tinyol.py \
    --config configs/cmapss_config.yaml \
    --exp-id exp_S22_03 \
    --output experiments/exp_S22_03/
```

**Métriques attendues** :

| Métrique | Valeur cible |
|----------|:------------:|
| `acc_final` | ≥ 0.70 |
| `avg_forgetting` | ≤ 0.20 |

**Note** : TinyOL est architecture-based (tête OtO). Input `input_dim=5` doit être compatible avec la config TinyOL courante.

---

## S2208 — exp_S22_04 : Mahalanobis / CMAPSS

```bash
python scripts/train_mahalanobis.py \
    --config configs/cmapss_config.yaml \
    --exp-id exp_S22_04 \
    --output experiments/exp_S22_04/
```

**Métriques attendues** :

| Métrique | Valeur cible |
|----------|:------------:|
| `acc_final` | ≥ 0.60 |
| `avg_forgetting` | ≤ 0.25 |

---

## S2209 — Notebook `notebooks/results_cmapss_cl.ipynb`

### Sections requises

1. **Chargement résultats** : lecture des 4 `results.json` (`exp_S22_01` à `exp_S22_04`)

2. **Tableau comparatif** :

| Modèle | acc_final | avg_forgetting | BWT | RAM (Ko) | n_params |
|--------|:---------:|:--------------:|:---:|:--------:|:--------:|
| EWC | | | | | 769 |
| HDC | | | | | — |
| TinyOL | | | | | |
| Mahalanobis | | | | | |

3. **Courbes AUROC par tâche** : matrice 4×4 (modèle × tâche CL FDxxx), séquence temporelle

4. **Heatmap forgetting** : matrice `acc[task_i][after_task_j]` pour chaque modèle (cf. DeLange2021Survey Fig. 2)

5. **Comparaison avec Monitoring** : scatter `acc_final CMAPSS vs Monitoring` par modèle (contexte Gap 1)

```python
# Cellule de conclusion obligatoire
gap1_cmapss_models = [m for m in results if results[m]["acc_final"] >= 0.65]
print(f"Gap 1 — CMAPSS : {len(gap1_cmapss_models)}/4 modèles au-dessus du seuil")
# FIXME(gap1) : documenter dans manuscrit §4 si EWC atteint la cible ≥ 0.75
```

---

## Vérification end-to-end

```bash
# Lancer les 4 expériences séquentiellement
python scripts/train_ewc.py --config configs/cmapss_config.yaml --exp-id exp_S22_01 --output experiments/exp_S22_01/
python scripts/train_hdc.py --config configs/cmapss_config.yaml --exp-id exp_S22_02 --output experiments/exp_S22_02/
python scripts/train_tinyol.py --config configs/cmapss_config.yaml --exp-id exp_S22_03 --output experiments/exp_S22_03/
python scripts/train_mahalanobis.py --config configs/cmapss_config.yaml --exp-id exp_S22_04 --output experiments/exp_S22_04/

# Vérifier présence de tous les results.json
for d in experiments/exp_S22_0{1,2,3,4}/; do
    [ -f "$d/results.json" ] && echo "$d OK" || echo "$d MANQUANT"
done

# Vérifier les métriques EWC
python -c "
import json
r = json.load(open('experiments/exp_S22_01/results.json'))
assert r['acc_final'] >= 0.75, f\"acc_final {r['acc_final']} < 0.75\"
assert r['avg_forgetting'] <= 0.15, f\"forgetting {r['avg_forgetting']} > 0.15\"
print('exp_S22_01 EWC — critères manuscrit OK')
"
```

---

## Résultats mesurés (2026-05-28)

| Modèle | acc_final | avg_forgetting | BWT | RAM (B) | n_params | Cible OK ? |
| ----------- | :-------: | :------------: | :-----: | :-----: | :------: | :--------: |
| EWC | **0.914** | **0.0002** | +0.001 | 1 171 | 737 | ✅ ✅ |
| HDC | **0.862** | **0.043** | -0.039 | 14 504 | 2 048 | ✅ ✅ |
| TinyOL | **0.894** | **0.018** | -0.018 | 3 679 | 274 | ✅ ✅ |
| Mahalanobis | 0.500 | 0.419 | -0.406 | 1 532 | 30 | ❌ ❌ |

**Note Mahalanobis** : L'alternance 1-condition (FD001/FD003) / 6-conditions (FD002/FD004) provoque un catastrophic forgetting structurel pour le détecteur d'anomalie à seuil fixe. Ce comportement est un résultat scientifique (Gap 1 : limite des approches non-neuronales sur scénarios multi-condition). `FIXME(gap1)` : documenter dans §4 manuscrit.

**Câblage CMAPSS** :

- `src/evaluation/feature_importance.py` : ajout `FEATURE_NAMES_CMAPSS`
- `configs/cmapss_config.yaml` : sections `hdc`, `mahalanobis`, `n_features`, `n_classes`, `feature_bounds`
- `configs/cmapss_tinyol_config.yaml` : créé (backbone 5→16→8→4D, OtO 5D)
- `scripts/train_ewc.py`, `train_hdc.py`, `train_tinyol.py`, `train_mahalanobis.py` : branche `cmapss` / `by_domain`

## Questions ouvertes

- `TODO(arnaud)` : Si `acc_final EWC < 0.75` sur CMAPSS, doit-on tuner `ewc.lambda` avant de conclure ou garder la config par défaut (monitoring_config) pour comparabilité ? → **Résolu** : EWC atteint 0.914 sans tuning, config par défaut conservée.
- `FIXME(gap1)` : Les 4 résultats CMAPSS doivent figurer dans `docs/datasets_analysis.md` — section à créer en S2224.
