# S2-09 — Mise à jour README avec résultats préliminaires

| Champ | Valeur |
|-------|--------|
| **ID** | S2-09 |
| **Sprint** | Sprint 2 — Semaine 2 (22–29 avril 2026) |
| **Priorité** | 🟢 Nice-to-have |
| **Durée estimée** | 1h |
| **Dépendances** | S2-03 (`exp_002` — résultats HDC), S2-04 (`notebooks/02_baseline_comparison.ipynb` — comparaison visuelle) |
| **Fichiers cibles** | `README.md` |
| **Complété le** | — |

---

## Objectif

Mettre à jour `README.md` avec les résultats du Sprint 2 :
- Ajouter les résultats de M3 HDC (exp_002)
- Compléter le tableau comparatif EWC vs HDC vs Fine-tuning
- Marquer M3 HDC comme implémenté dans le tableau de progression
- Ajouter le lien vers le notebook de comparaison

**Critère de succès** : `README.md` contient les résultats numériques de `exp_002` et le tableau comparatif 3 modèles est complet.

---

## Sous-tâches

### 1. Mettre à jour la section "Résultats" (Quick Results)

Ajouter ou compléter après la section M2 EWC existante :

```markdown
### Résultats M3 HDC — exp_002 (seed=42, Dataset 2 — 3 domaines)

| Métrique | HDC | EWC Online | Fine-tuning naïf |
|----------|:---:|:----------:|:----------------:|
| AA | [à compléter après exp_002] | **0.9824** | 0.9811 |
| AF | [à compléter] | **0.0010** | 0.0000 |
| BWT | [à compléter] | +0.0000 | +0.0010 |
| RAM peak inférence | [mesurer] | **1.1 Ko** | — |
| RAM peak mise à jour | [mesurer] | **6.7 Ko** | — |
| Latence inférence | [mesurer] | **0.036 ms** | — |
| Budget 64 Ko | [% utilisés] | ✅ 10.4% | — |

> Voir `experiments/exp_002_hdc_dataset2/results/` pour les valeurs complètes.
> Analyse complète : `notebooks/02_baseline_comparison.ipynb`.
```

> **Note** : les valeurs `[à compléter]` sont des placeholders à remplacer par les résultats
> réels de `exp_002` (S2-03) avant de committer.

### 2. Mettre à jour le tableau "Indicateurs de progression"

Remplacer la ligne M3 HDC :

```markdown
| Modèle | Implémenté | Testé | Expérience | Export ONNX | RAM mesurée |
|--------|:----------:|:-----:|:----------:|:-----------:|:-----------:|
| M2 EWC + MLP | ✅ | ✅ | ✅ | ⬜ | ✅ |
| M3 HDC | ✅ | ✅ | ✅ | ⬜ | ✅ |   ← mise à jour Sprint 2
| M1 TinyOL | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| M1 + buffer UINT8 | ⬜ | ⬜ | ⬜ | N/A | ⬜ |
```

### 3. Ajouter le lien vers le notebook de comparaison

Dans la section "Quick Start" ou "Notebooks", ajouter :

```markdown
## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/01_data_exploration.ipynb` | Exploration Dataset 2 (Equipment Monitoring) |
| `notebooks/02_baseline_comparison.ipynb` | **Comparaison EWC vs HDC vs Fine-tuning** (Sprint 2) |
```

### 4. Mettre à jour la section "Commandes rapides"

Vérifier que le script HDC est bien documenté :

```bash
# Générer les vecteurs de base HDC (une seule fois)
python -m src.models.hdc.base_vectors

# Entraîner HDC
python scripts/train_hdc.py --config configs/hdc_config.yaml

# Comparer EWC vs HDC
jupyter nbconvert --to notebook --execute notebooks/02_baseline_comparison.ipynb
```

---

## Critères d'acceptation

- [ ] `README.md` contient les résultats numériques de `exp_002` (sans valeurs `null` ou `[à compléter]` dans la version committée)
- [ ] Tableau comparatif 3 modèles complet (AA/AF/BWT/RAM pour EWC, HDC, Fine-tuning)
- [ ] Indicateurs de progression : M3 HDC = ✅ sur Implémenté, Testé, Expérience, RAM mesurée
- [ ] Lien vers `notebooks/02_baseline_comparison.ipynb` ajouté
- [ ] Commande `train_hdc.py` documentée dans Quick Start

---

## Contenu à NE PAS modifier dans `README.md`

- Section "Triple Gap" — ne pas altérer le positionnement scientifique
- Section "Hardware Target (STM32N6)" — spécifications matérielles fixes
- Section "Architecture" — structure du dépôt
- Références aux superviseurs

---

## Questions ouvertes

- `TODO(arnaud)` : faut-il ajouter une figure (accuracy matrix heatmap) directement dans le README, ou uniquement pointer vers le notebook ?
- `TODO(fred)` : y a-t-il des résultats de référence industriels (Edge Spectrum) à inclure pour contextualiser les chiffres HDC ?
