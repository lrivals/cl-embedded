# S11-22 — Documentation Sprint 11 : bilan partie CWRU et Pronostia

| Champ | Valeur |
|-------|--------|
| **ID** | S11-22 |
| **Sprint** | Sprint 11 |
| **Priorité** | 🟢 Nice-to-have |
| **Durée estimée** | 0.5h |
| **Dépendances** | S11-18, S11-19, S11-20 (notebooks exécutés), S11-16, S11-17 (exp_100–111 produites) |
| **Fichier cible** | `docs/sprints/sprint_11/` (ce fichier + mise à jour S1100) |

---

## Objectif

Documenter les résultats obtenus sur CWRU et Pronostia (feature importance per-task, notebooks
comparatifs, cross-dataset) une fois les expériences exp_100–111 et les notebooks S11-18 à S11-20
exécutés avec succès.

---

## Contenu à renseigner

### Tableau de synthèse exp_100–111

| Exp | Modèle | Dataset | Scénario | AA | Top feature (global) | JSON produit |
| --- | ------ | ------- | -------- | -- | -------------------- | :----------: |
| exp_100 | KMeans | CWRU | by_fault_type | 0.312 | skewness | ✅ |
| exp_101 | KMeans | CWRU | by_severity | 0.450 | mean | ✅ |
| exp_102 | KMeans | Pronostia | by_condition | 0.872 | kurtosis_acc_vert | ✅ |
| exp_103 | Mahalanobis | CWRU | by_fault_type | 0.160 | skewness | ✅ |
| exp_104 | Mahalanobis | CWRU | by_severity | 0.195 | skewness | ✅ |
| exp_105 | Mahalanobis | Pronostia | by_condition | 0.898 | std_acc_vert | ✅ |
| exp_106 | EWC | CWRU | by_fault_type | 1.000 | kurtosis | ✅ |
| exp_107 | EWC | CWRU | by_severity | 0.952 | skewness | ✅ |
| exp_108 | EWC | Pronostia | by_condition | 0.982 | temporal_position ⚠️ | ✅ |
| exp_109 | HDC | CWRU | by_fault_type | 0.935 | form | ✅ |
| exp_110 | HDC | CWRU | by_severity | 0.892 | sd | ✅ |
| exp_111 | HDC | Pronostia | by_condition | 0.850 | crest_factor_acc_horiz | ✅ |

> ⚠️ exp_108 : `temporal_position` top feature EWC Pronostia → `FIXME(gap1)` — non disponible en déploiement MCU, à exclure ou traiter séparément dans S11-20.

### Notebooks comparatifs

| Notebook | Fichier | Statut |
|----------|---------|:------:|
| CWRU by_fault_type | `notebooks/cl_eval/cwru_feature_importance/by_fault_type.ipynb` | ✅ |
| CWRU by_severity | `notebooks/cl_eval/cwru_feature_importance/by_severity.ipynb` | ✅ |
| Pronostia by_condition | `notebooks/cl_eval/pronostia_feature_importance/by_condition.ipynb` | ✅ |
| Cross-dataset | `notebooks/cl_eval/cross_dataset_feature_importance.ipynb` | ⬜ |

---

## Questions à répondre après exécution

### CWRU
- `kurtosis` est-il stable en top-1 sur by_fault_type ET by_severity ?
  Si oui → feature MCU prioritaire confirmée, argument Gap 2 (sélection minimaliste).
- Les features importantes varient-elles selon le type de défaut (ball/inner/outer) ?
  Si instables → nuance à ajouter dans la section interprétabilité du manuscrit.
- `FIXME(gap2)` : noter la réduction de RAM si on passe de 9 features à top-3 pour KMeans et Mahalanobis.

### Pronostia
- `temporal_position` : importance élevée ou faible ?
  - Si élevée : `FIXME(gap1)` — annoter dans le notebook, retirer pour les expériences de publication.
  - Si faible : aucun problème.
- Canal dominant : `acc_horiz` > `acc_vert` comme attendu mécaniquement ?
- Les conditions opératoires (rpm différents) produisent-elles des rankings différents ?
  Si oui → justification empirique du scénario CL by_condition.

### Cross-dataset
- Familles statistiques universellement importantes (`rms`, `kurtosis`, `std`, `crest`) :
  lesquelles apparaissent en top-3 dans ≥ 2 datasets sur 3 ?
  → Recommandation embarquée à écrire en section 5 du notebook cross-dataset.

---

## Mise à jour à faire dans S1100

Une fois tous les notebooks exécutés, mettre à jour les statuts dans :
- [S1104_feature_importance_extension_cwru_pronostia.md](S1104_feature_importance_extension_cwru_pronostia.md) : passer S11-11 à S11-17 en `✅`
- [S1105_notebooks_feature_importance_cwru_pronostia.md](S1105_notebooks_feature_importance_cwru_pronostia.md) : passer S11-18 à S11-20 en `✅`

---

## Statut

🔄 Partiellement complété — exp_100–111 ✅ et notebooks S11-18/S11-19 ✅ · S11-20 (cross-dataset) ⬜ restant
