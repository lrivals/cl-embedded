# Sprint 51 — Score système composite + contexte de comparaison matérielle

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 51 |
| **Semaine** | Indicative — à confirmer (après S49/S50, dont il consomme les sorties) |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Priorité globale** | 🟢 Bas (items 🟢 du CR) mais **structurant pour le rapport/soutenance** — définit un **indicateur composite** unique qui classe les configurations (modèle × carte × encodage) au lieu de métriques isolées, et cadre la comparaison matérielle (PC vs carte, puis carte vs carte). |
| **Durée estimée totale** | ~18h (définition score ~3h · implémentation ~4h · classement ~2h · spécifs matérielles ~2h · réorientation board↔board ~2h · figures+notebook ~3h · tests+docs ~2h) |
| **Dépendances** | **Sprint 49** (RAM totale `exp_S49_ram/`) · **Sprint 50** (énergie/inférence, autonomie) · `src/evaluation/hw_cost_model.py` ✅ · `src/evaluation/compute_cost.py` ✅ (MACs/BOPs/params) · harnais PC↔board S36 ✅ · registre figures `src/figures/` ✅ (S4201) |

## Contexte et motivation

Le CR §3 identifie un manque : **aucune métrique unique** ne capture simultanément toutes les dimensions de
performance. Il faut un **indicateur composite** adapté au contexte embarqué, pour **classer les
configurations** (modèle × carte × encodage) plutôt que de présenter des métriques isolées.

Dimensions retenues (CR §3) :

| Dimension | Métrique | Source |
|-----------|----------|--------|
| Mémoire | RAM totale (`.data + .bss + pic de pile`) | Sprint 49 |
| Calcul | Latence d'inférence (ms) | board DWT (S29/S36) |
| Énergie | Énergie/inférence (µJ) | Sprint 50 |
| Modèle | Params / MACs | `compute_cost.py` |
| Performance | Accuracy / F1 / AUROC | benchmarks (S28/S36/S46) |

Second volet (CR §1 + tâches 🟢) : **contextualiser la comparaison matérielle**. La comparaison PC↔board du CR
n'est **pas directement exploitable** (architectures trop différentes : PC multi-cœur ⇒ overhead scheduling vs
carte mono-cœur séquentielle). Le CR demande donc (a) de **détailler les spécificités des deux matériels** pour
la soutenance, et (b) de **réorienter la mise en place vers une comparaison carte↔carte** (plus pertinente).

## Décisions de cadrage (utilisateur, CR du 16 juillet 2026)

- **Score système composite** intégrant RAM + latence + énergie + params/MACs + accuracy.
- **But = classer** les configurations (modèle × carte × encodage) selon un critère global.
- **Pondérations explicites** et documentées (choix, pas une vérité) — `TODO(arnaud)` pour valider.
- **PC↔board non fusionné** : reporté comme contexte, pas comme comparaison exploitable (CR §1).
- **Réorientation board↔board** = perspective documentée (dépend d'une 2e carte, non garantie).
- **Aucun chiffre en dur** : le score se recalcule depuis les JSON S49/S50 + `compute_cost.py`.
- **Langue** : français.

## Nœud honnête : un score composite est un choix de pondération

Le score **ne redéfinit pas** ce qu'est un bon modèle : il **agrège** des dimensions hétérogènes selon une
pondération **choisie**. Deux nuances à porter :

1. **Normalisation obligatoire** : les dimensions ont des unités incomparables (octets, ms, µJ, sans-unité) →
   normaliser (min-max ou z-score) par dimension avant pondération, sinon la plus grande échelle domine.
2. **Sensibilité aux poids** : le classement dépend des poids ; fournir une **analyse de sensibilité** (le rang
   change-t-il si on privilégie RAM vs énergie ?) plutôt qu'un classement unique présenté comme absolu.
3. **PC↔board** : ne jamais comparer un score PC et un score board comme si l'un était « meilleur » — les
   conditions matérielles diffèrent (CR §1). Le score classe **à plateforme fixée**.

## Tâches

### Bloc A — Définition & implémentation

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S5101 | **Définition du score système composite** : dimensions, normalisation par dimension, schéma d'agrégation, pondérations documentées (choix explicite, `TODO(arnaud)`), analyse de sensibilité | 🟢 | `docs/context/system_score.md` | 📝 Doc |
| S5102 | **Implémentation `system_score.py`** : consomme `exp_S49_ram/` + `exp_S50_energy/` + `compute_cost.py` ; normalise, pondère, produit score + rang par configuration ; sensibilité aux poids | 🟢 | `src/evaluation/system_score.py`, `tests/test_system_score.py` | 📝 Doc |

### Bloc B — Classement & contexte matériel

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S5103 | **Classement des configurations** (modèle × carte × encodage) selon le score global, à plateforme fixée | 🟢 | `experiments/exp_S51_system_score/` | 📝 Doc |
| S5104 | **Spécifications matérielles (2 matériels)** pour la soutenance : PC utilisé (multi-cœur ⇒ overhead scheduling, cf. CR §1) + NUCLEO-F439ZI (Cortex-M4 mono-cœur, 256 Ko, FPU, pas de NPU) ; contexte de comparaison | 🟢 | `docs/context/hardware_comparison.md` | 📝 Doc |
| S5105 | **Réorientation board↔board** : documenter comment réutiliser le harnais S36 (mêmes données/éval) pour comparer 2 cartes plutôt que PC↔board — **perspective** (dépend d'une 2e carte) | 🟢 | `docs/sprints/sprint_51/S5105_board_vs_board.md` | 📝 Doc |

### Bloc C — Assemblage & clôture

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S5106 | **Figures + notebook** : classement composite (barres/radar par config), sensibilité aux poids, N/A gris, garde AST 0-chiffre-en-dur | 🟢 | `src/figures/catalogs/system_score.py`, `docs/figures/system_score/`, `notebooks/cl_eval/system_score/comparison.ipynb` | 📝 Doc |
| S5107 | **Tests + docs + clôture** : `test_system_score.py` (normalisation bornée [0,1], monotonie par dimension, sensibilité, 0-chiffre-en-dur) ; MAJ roadmap/triple_gap ; `graphify_sprint_update` | 🟢 | `tests/test_system_score.py`, `docs/roadmap_phase2.md`, `docs/triple_gap.md` | 📝 Doc |

## Ordre d'exécution recommandé

```
S5101 (définition : dimensions + normalisation + pondérations)
   │
   ▼
S5102 (system_score.py : consomme S49/S50 + compute_cost)
   │
   ├──► S5103 (classement des configurations)
   ├──► S5104 (specs 2 matériels — soutenance)
   └──► S5105 (réorientation board↔board — perspective)
                 │
                 ▼
         S5106 (figures + notebook)
                 │
                 ▼
         S5107 (tests + roadmap + triple_gap + graphify)
```

Dépend des sorties S49 (RAM) + S50 (énergie). Sans elles, le score tourne en `pending` (dimensions manquantes).

## Sources de données (Sprint 51, lecture seule)

| Source | Rôle |
| ------ | ---- |
| `exp_S49_ram/summary.json` | Dimension mémoire (RAM totale) |
| `exp_S50_energy/summary.json`, `autonomy.json` | Dimension énergie |
| `compute_cost.py` | Params / MACs / BOPs |
| Benchmarks S28/S36/S46 | Dimension performance (accuracy/F1/AUROC) |
| DWT board (S29/S36) | Dimension latence |

## Livrables

1. `docs/context/system_score.md` — définition + normalisation + pondérations + sensibilité (S5101).
2. `src/evaluation/system_score.py` + tests — calcul du score et du rang (S5102).
3. `experiments/exp_S51_system_score/` — classement par configuration (S5103).
4. `docs/context/hardware_comparison.md` — specs des 2 matériels (S5104).
5. `docs/sprints/sprint_51/S5105_board_vs_board.md` — perspective board↔board (S5105).
6. `src/figures/catalogs/system_score.py` → `docs/figures/system_score/` + notebook (S5106).
7. `tests/test_system_score.py` + MAJ roadmap/triple_gap (S5107).
