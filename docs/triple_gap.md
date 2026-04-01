# Triple Gap — Positionnement scientifique du projet

> Ce document formalise le positionnement original du stage.  
> Il doit être consulté avant toute décision d'architecture ou d'expérimentation.

---

## Définition du triple gap

Le triple gap désigne l'absence simultanée, dans la littérature existante, de travaux qui :

| Gap | Critère | Status de la littérature (corpus 20 articles, avril 2026) |
|-----|---------|----------------------------------------------------------|
| **Gap 1** | Validation sur données industrielles de séries temporelles réelles avec protocole reproductible | ❌ Aucun article ne satisfait ce critère |
| **Gap 2** | Démonstration d'un CL complet sous 100 Ko RAM avec chiffres précis mesurés par composant | ❌ Aucun article ne satisfait ce critère |
| **Gap 3** | Quantification INT8 appliquée à la phase d'entraînement incrémental (backpropagation) | ❌ Aucun article ne satisfait ce critère |

> **La contribution originale du stage** est d'être le premier travail à adresser ces trois gaps simultanément.

---

## Mapping corpus → triple gap

| Article | Gap 1 | Gap 2 | Gap 3 | Score |
|---------|:-----:|:-----:|:-----:|:-----:|
| TinyOL (Ren et al., 2021) | ❌ | ❌ | ❌ | 0/3 |
| QLR-CL (Ravaglia et al., 2021) | ❌ | ❌ | ⚠️ buffer UINT8 | 0/3 |
| LifeLearner (Kwon et al., 2023) | ❌ | ⚠️ 212 Ko | ❌ | 0/3 |
| EWC (Kirkpatrick et al., 2017) | ❌ | ❌ | ❌ | 0/3 |
| HDC-EMG (Benatti et al., 2019) | ⚠️ EMG (pas PdM) | ✅ < 4 Ko | ❌ | ~1/3 |
| CL×PdM (Hurtado et al., 2023) | ⚠️ PdM mais datasets divers | ❌ | ❌ | 0/3 |
| Gradient Monitoring (Shah et al., 2025) | ⚠️ RUL industriel | ❌ | ❌ | ~0.5/3 |
| Adaptive CL (Wu et al., 2025) | ⚠️ Séries temporelles industrielles | ❌ | ❌ | ~0.5/3 |
| Dataset Distillation (Rüb et al., 2024) | ❌ | ❌ | ❌ | 0/3 |
| AR1* (Pellegrini et al., 2021) | ❌ | ❌ | ❌ | 0/3 |

**Constat** : aucun article ne dépasse 1/3. Plusieurs articles adressent partiellement le Gap 1 (données industrielles), mais aucun ne combine les trois.

---

## Contribution de ce projet au triple gap

### Gap 1 — Données industrielles

**Adressé partiellement** par les deux datasets Kaggle (simulés mais industriellement motivés). Référence scientifique : FEMTO PRONOSTIA (Nectoux et al., 2012) dans le manuscrit.

**Limitation honnête** : les datasets Kaggle sont synthétiques. Le manuscrit mentionnera explicitement cette limitation et positionnera FEMTO PRONOSTIA comme la cible expérimentale de la Phase 2 du stage (post-avril 2026).

### Gap 2 — Sub-100 Ko RAM avec chiffres précis

**Adressé** : les trois modèles sont estimés à < 15 Ko en RAM. Le profiling systématique via `tracemalloc` + mesures MCU produira les premiers chiffres précis par composant dans la littérature.

**Métrique clé** : `ram_peak_bytes` dans `evaluation/memory_profiler.py`.

### Gap 3 — Quantification INT8 pendant l'entraînement

**Adressé exploratoirement** via le buffer UINT8 (extension M1) et l'exploration INT8 backprop sur M2 (MLP minimal). Ce n'est pas une démonstration complète mais une investigation préliminaire documentée.

**Question ouverte pour Dorra** : CMSIS-NN fournit des kernels INT8 pour l'inférence ; existe-t-il un équivalent pour la mise à jour de poids ? (`arm_nn_vec_mat_mult_t_s8` ?)

---

## Utilisation dans le code

Chaque fonction critique du projet doit être annotée par rapport au triple gap :

```python
def update_oto_head(self, z: Tensor, y: Tensor) -> float:
    """
    Mise à jour en ligne de la tête OtO.
    
    Gap 2 relevance: Cette fonction s'exécute en RAM, sur Cortex-M55.
    Empreinte mesurée : voir experiments/exp_003/results/metrics.json
    
    Gap 3 relevance: Mise à jour en FP32. L'extension INT8 est dans
    l'extension buffer UINT8 (docs/models/tinyol_spec.md §7).
    """
```

---

## Critères de succès expérimentaux

Pour que ce projet "ferme" les gaps de manière crédible :

| Gap | Critère de succès minimal |
|-----|--------------------------|
| Gap 1 | Accuracy > 80 % sur Dataset 1 (temporel) avec protocole CL documenté |
| Gap 2 | `ram_peak_bytes` < 65 536 mesuré à l'exécution pour les 3 modèles |
| Gap 3 | Démonstration que le buffer UINT8 dégrade < 2 % la précision vs FP32 |
