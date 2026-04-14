# Sprint 9 (Phase 1) — Finalisation, archivage & mise à jour roadmap

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 9 — Phase 1 Extension (finalisation) |
| **Priorité globale** | 🟡 Important — clôture Phase 1 propre |
| **Durée estimée totale** | ~6h |
| **Dépendances** | Sprints 7 et 8 terminés (28 notebooks créés) |

---

## Objectif

Clore proprement la Phase 1 Python :
1. Mettre à jour `03_cl_evaluation.ipynb` (archivage + liens vers nouveaux notebooks)
2. Mettre à jour `docs/roadmap.md` et `docs/roadmap_phase1.md` avec les Sprints 6–9 et leurs résultats
3. Renommer le Sprint 6 Phase 2 (MCU) en Sprint 10 dans `docs/roadmap_phase2.md`

**Critère de succès** : roadmaps à jour, `03_cl_evaluation.ipynb` dispose d'une note d'introduction pointant vers les notebooks granulaires, le projet est prêt pour les présentations d'encadrement.

---

## Tâches

| ID | Tâche | Priorité | Fichier cible | Durée est. | Dépendances |
|----|-------|:---:|---------------|:---:|-------------|
| S9-01 | Mettre à jour `notebooks/03_cl_evaluation.ipynb` : ajouter cellule d'introduction + liens vers `notebooks/cl_eval/` | 🟡 | `notebooks/03_cl_evaluation.ipynb` | 1h | Sprints 7, 8 |
| S9-02 | Mettre à jour `docs/roadmap_phase1.md` : ajouter Sprints 6–9 avec leurs tâches et résultats | 🔴 | `docs/roadmap_phase1.md` | 2h | Sprint 6, 7, 8 |
| S9-03 | Mettre à jour `docs/roadmap.md` : ligne Phase 1 = Sprints 1–9, Phase 2 = Sprint 10 | 🟡 | `docs/roadmap.md` | 30min | S9-02 |
| S9-04 | Renommer Sprint 6 → Sprint 10 dans `docs/roadmap_phase2.md` | 🟡 | `docs/roadmap_phase2.md` | 30min | — |
| S9-05 | Mettre à jour la table des indicateurs de progression dans `docs/roadmap.md` (colonne Exp. pour les nouveaux scénarios) | 🟢 | `docs/roadmap.md` | 30min | S9-03 |
| S9-06 | Mise à jour README.md : section "Notebooks" avec la structure `cl_eval/` | 🟢 | `README.md` | 1h | Sprints 7, 8 |

---

## Détail des tâches

### S9-01 — Mise à jour `03_cl_evaluation.ipynb`

Ajouter en **première cellule** (cellule Markdown) :

```markdown
# ⚠️ Note — Notebook archivé (Sprint 3)

Ce notebook a été le premier notebook d'évaluation CL du projet (Sprint 3 — S3-08).
Il couvre uniquement **TinyOL sur Dataset 1** (scénario chronologique 3 tâches).

**Pour les évaluations complètes (6 modèles × 4 scénarios × 2 datasets), voir :**

| Scénario | Dataset | Lien |
|----------|---------|------|
| By Equipment (Pump→Turbine→Compressor) | Dataset 2 Monitoring | [`cl_eval/monitoring_by_equipment/`](cl_eval/monitoring_by_equipment/) |
| By Location (ATL→CHI→HOU→NYC→SFO) | Dataset 2 Monitoring | [`cl_eval/monitoring_by_location/`](cl_eval/monitoring_by_location/) |
| By Pump_ID (P1→P2→P3→P4→P5) | Dataset 1 Pump | [`cl_eval/pump_by_pump_id/`](cl_eval/pump_by_pump_id/) |
| By Temporal Window (Q1→Q2→Q3→Q4) | Dataset 1 Pump | [`cl_eval/pump_by_temporal_window/`](cl_eval/pump_by_temporal_window/) |

Ce notebook est conservé pour la traçabilité de l'historique du Sprint 3.
```

### S9-02 — Mise à jour `docs/roadmap_phase1.md`

Ajouter les sections suivantes après le "Sprint 5 Extension" existant :

1. **Sprint 6 (Phase 1)** — table des tâches S6-01 à S6-11, avec colonnes Impl./Doc./Exec.
2. **Sprint 7 (Phase 1)** — table des tâches S7-01 à S7-14
3. **Sprint 8 (Phase 1)** — table des tâches S8-01 à S8-14
4. **Sprint 9 (Phase 1)** — table des tâches S9-01 à S9-06

Pour chaque sprint : ajouter la section "Résultats d'expériences" avec les métriques AA/AF/BWT/RAM mesurées lors de l'exécution.

### S9-03 — Mise à jour `docs/roadmap.md`

Modifier la ligne Phase 1 :
```markdown
# Avant
- [Phase 1 — Implémentation Python](roadmap_phase1.md) — Sprints 1–5, résultats expériences

# Après
- [Phase 1 — Implémentation Python](roadmap_phase1.md) — Sprints 1–9 (dont extension notebooks granulaires)
```

Modifier la ligne Phase 2 :
```markdown
# Avant
- [Phase 2 — Portage MCU](roadmap_phase2.md) — Sprint 6, Backlog

# Après
- [Phase 2 — Portage MCU](roadmap_phase2.md) — Sprint 10, Backlog
```

### S9-04 — Renommage Sprint 6 → Sprint 10 dans `roadmap_phase2.md`

Rechercher et remplacer toutes les occurrences de "Sprint 6" par "Sprint 10" dans `docs/roadmap_phase2.md`.
Le fichier `docs/sprints/sprint_6/S601_stm32_env_setup.md` reste à l'emplacement actuel (déjà commité, ne pas renommer le dossier pour éviter de casser l'historique git).

### S9-05 — Indicateurs de progression dans `docs/roadmap.md`

Ajouter des lignes dans la table des indicateurs pour les nouveaux scénarios :

```markdown
| Scénario pump_by_pump_id | ✅ | ✅ | ✅ | N/A | N/A |
| Scénario pump_by_temporal_window | ✅ | ✅ | ✅ | N/A | N/A |
| Scénario monitoring_by_location | ✅ | ✅ | ✅ | N/A | N/A |
| Notebooks cl_eval/ (28 notebooks) | ✅ | N/A | ✅ | N/A | N/A |
```

### S9-06 — README.md

Ajouter une section "Notebooks d'évaluation" avec la structure :
```markdown
## Notebooks d'évaluation CL

| Scénario | Modèles | Lien |
|----------|---------|------|
| Equipment Monitoring — by Equipment | TinyOL, EWC, HDC, KMeans, Mahalanobis, DBSCAN + Comparaison | `notebooks/cl_eval/monitoring_by_equipment/` |
| Equipment Monitoring — by Location | TinyOL, EWC, HDC, KMeans, Mahalanobis, DBSCAN + Comparaison | `notebooks/cl_eval/monitoring_by_location/` |
| Pump Maintenance — by Pump_ID | TinyOL, EWC, HDC, KMeans, Mahalanobis, DBSCAN + Comparaison | `notebooks/cl_eval/pump_by_pump_id/` |
| Pump Maintenance — by Temporal Window | TinyOL, EWC, HDC, KMeans, Mahalanobis, DBSCAN + Comparaison | `notebooks/cl_eval/pump_by_temporal_window/` |
```

---

## Résumé Phase 1 complète (post Sprint 9)

À la fin du Sprint 9, la Phase 1 Python comprend :
- **9 sprints** (au lieu des 5 initialement planifiés)
- **29 expériences** (exp_001 à exp_029)
- **6 modèles CL** : TinyOL, EWC, HDC, KMeans, Mahalanobis, DBSCAN
- **4 scénarios CL** : by_equipment (3t), by_location (5t), by_pump_id (5t), by_temporal_window (4t)
- **2 datasets** : equipment_anomaly_data.csv + Large_Industrial_Pump_Maintenance_Dataset.csv
- **28 notebooks** de présentation (`notebooks/cl_eval/`)
- Figures organisées en `notebooks/figures/cl_evaluation/{model}/{dataset}/{task}/`

---

## Critères d'acceptation

- [ ] `notebooks/03_cl_evaluation.ipynb` : cellule d'introduction avec liens vers cl_eval/ ajoutée
- [ ] `docs/roadmap_phase1.md` : Sprints 6–9 documentés avec leurs tâches
- [ ] `docs/roadmap.md` : Phase 1 = Sprints 1–9, Phase 2 = Sprint 10
- [ ] `docs/roadmap_phase2.md` : mentions de "Sprint 6" remplacées par "Sprint 10"
- [ ] `README.md` : section "Notebooks d'évaluation" ajoutée

---

## Livrable sprint 9

Phase 1 Python complète et documentée. Projet prêt pour la transition vers Phase 2 (portage MCU STM32). Roadmap à jour pour présentation aux encadrants Arnaud Dion (ISAE-SUPAERO) et Frédéric Zbierski (Edge Spectrum).

---

## Questions ouvertes

- `TODO(arnaud)` : Faut-il créer un notebook de synthèse "cross-scénarios" comparant les 4 scénarios entre eux pour un même modèle (ex : EWC sur monitoring_by_equipment vs monitoring_by_location vs pump_by_pump_id vs pump_by_temporal_window) ? Ce serait un 5ème type de notebook non planifié ici.
- `TODO(arnaud)` : Pour la deadline manuscrit préliminaire (15 avril 2026), quelles figures des notebooks de comparaison sont prioritaires pour les slides de présentation ?

---

> **⚠️ Après l'implémentation de ce sprint** : vérifier que `docs/roadmap_phase1.md` reflète fidèlement les résultats réels des expériences (métriques AA/AF/BWT/RAM), pas les valeurs mock. Confirmer que les 29 expériences sont bien exécutées avant de clore officiellement la Phase 1.
