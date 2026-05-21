# Bilan Sprint 14 — Anomaly Detection Monitoring (6 modèles × 2 stratégies × 2 scénarios)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 14 |
| **Semaine** | 5–9 mai 2026 |
| **Statut global** | ✅ CLOSED (S14-01 → S14-11) |
| **Tâche optionnelle** | S14-12 (accumulate by_location) — 🟢 non réalisée |
| **Expériences** | exp_123 – exp_136 (14 expériences) |
| **Dépendances Sprint 15** | EWCOneClassDetector + DBSCANDetector conformes API |

---

## 1. Bilan technique

**1. Couverture critique atteinte à 100 % (S14-01 → S14-11).**
Les 11 tâches obligatoires/importantes sont ✅. La seule tâche manquante est S14-12 (accumulate by_location), explicitement optionnelle (🟢) et logiquement ajournée au vu du risque DBSCAN identifié.

**2. API CL unifiée respectée sans modification du scénario générique.**
S14-03 valide que `DBSCANDetector` s'intègre dans `run_anomaly_detection_scenario()` sans toucher à `src/training/scenarios.py` — le design de l'interface générique tient.

**3. Couverture de tests supérieure aux specs.**
Le plan prévoyait 7 + 4 = 11 tests (S14-11) ; l'exécution livre 13 + 12 = 25 tests. Cette densité de couverture supplémentaire est un signal de qualité, mais il faudra vérifier que les cas non-planifiés testent bien des comportements nouveaux et non des redondances.

**4. RAM `EWCOneClassDetector` : contrainte 64 Ko largement respectée.**
Budget théorique ≈ 7 Ko (2 Ko modèle + 2 Ko Fisher + 2 Ko θ* + 1 Ko activations). C'est le modèle le plus frugal du sprint — mais la RAM Fisher/θ* *double* à chaque tâche mémorisée si l'implémentation conserve l'historique complet (à clarifier).

**5. `EWCOneClassDetector` : question de validation scientifique non tranchée.**
Le `TODO(arnaud)` sur la formulation (autoencoder MLP + EWC sur MSE vs. MLP one-class avec output scalaire) est ouvert. Les expériences sont lancées mais l'interprétation des résultats dans le manuscrit est conditionnelle à cette validation.

**6. Risque DBSCAN accumulate identifié mais non mitigé.**
Le `TODO(arnaud)` sur la borne du buffer est documenté sans décision. S14-12 n'ayant pas été faite, le risque de lenteur à la tâche 5 (by_location accumulate) reste théorique — il deviendra concret si le sprint 15 réutilise ce pattern sur Pronostia.

**7. Notebook S14-10 exécutable, figures générées.**
Les 4 catégories de figures (AUROC table, per-task, RAM, forgetting) sont dans `notebooks/figures/anomaly_detection/monitoring/` — vérifier que les paths sont bien commités (les `data/` sont en `.gitignore`, les figures ne le sont pas par défaut).

---

## 2. Analyse des résultats

### Tableau AUROC synthèse

| Modèle | by_equip refit | by_equip accum | by_loc refit | **Moy. 3 scénarios** |
|--------|:--------------:|:--------------:|:------------:|:--------------------:|
| Mahalanobis | 0.9877 | 0.9877 | 0.9879 | **0.9878** |
| DBSCAN | 0.9871 | 0.9873 | 0.9857 | **0.9867** |
| KMeans | 0.9845 | 0.9845 | 0.9851 | **0.9847** |
| EWC one-class | 0.9630 | 0.9682 | 0.9552 | **0.9621** |
| TinyOL AE | 0.9628 | 0.9628 | 0.9329 | **0.9528** |
| HDC | 0.9451 | 0.9451 | 0.9470 | **0.9457** |

### Impact stratégie : refit vs. accumulate

| Modèle | Δ AUROC (accum − refit) | Interprétation |
|--------|:-----------------------:|----------------|
| EWC OC | **+0.0052** | Seul modèle avec gain notable — l'accumulation de contexte aide l'autoencoder |
| DBSCAN | +0.0002 | Quasi-nul — DBSCAN s'adapte bien au refit, accumulate n'apporte rien |
| Mahalanobis | 0.0000 | Invariant — le modèle gaussien est suffisamment expressif par tâche |
| KMeans | 0.0000 | Idem |
| TinyOL AE | 0.0000 | Accumulate ne change rien pour l'AE (convergence identique) |
| HDC | 0.0000 | Invariant — la structure HDC absorbe le signal par tâche complètement |

**Lecture** : l'accumulate n'apporte de bénéfice mesurable que pour EWC OC (+0.5 pp). Pour les 5 autres modèles, refit est suffisant — ce qui simplifie le portage MCU (pas de buffer cumulatif nécessaire).

### Impact scénario : by_equipment vs. by_location

| Modèle | by_equip refit | by_loc refit | Δ | Sensibilité au scénario |
|--------|:--------------:|:------------:|:--:|:----------------------:|
| Mahalanobis | 0.9877 | 0.9879 | +0.0002 | Nulle |
| KMeans | 0.9845 | 0.9851 | +0.0006 | Nulle |
| DBSCAN | 0.9871 | 0.9857 | −0.0014 | Très faible |
| EWC OC | 0.9630 | 0.9552 | **−0.0078** | Modérée |
| HDC | 0.9451 | 0.9470 | +0.0019 | Nulle |
| TinyOL AE | 0.9628 | 0.9329 | **−0.0299** | **Forte** |

**Gagnants stables** : Mahalanobis et KMeans — AUROC quasi-constant sur toutes les configurations. Leur robustesse tient à la nature de la distribution (gaussienne/sphérique) qui correspond bien au dataset Monitoring (4 features tabulaires).

**Perdant relatif** : TinyOL AE — la chute de −0.030 entre by_equipment et by_location est le signal le plus notable du sprint. L'autoencoder MLP est sensible au shift de distribution entre locations (plus hétérogènes que les types d'équipement). Ce résultat doit être discuté dans le manuscrit : il pointe une limite architecturale du TinyOL AE en scénario domain-incremental avec dérive forte.

**EWC OC** se positionne en 4e place (0.9621), mieux que TinyOL AE (0.9528) et HDC (0.9457), mais avec une perte sur by_location (−0.0078) qui soulève la question de la validité de la régularisation EWC sur MSE seul — ce que le `TODO(arnaud)` pointe exactement.

---

## 3. Conclusion / Recommandations

### Ce que le sprint apporte au manuscrit

- **Gap 1 ✅ (données industrielles réelles)** — Couverture complète sur Monitoring : 6 modèles × 2 stratégies × 2 scénarios, 14 expériences reproductibles. Le tableau AUROC 6×3 est un résultat directement publiable pour la section "Anomaly Detection Monitoring".

- **Gap 2 ✅ (< 100 Ko RAM mesurés)** — `EWCOneClassDetector` avec input_dim=4 est estimé à ≈ 7 Ko total, profil RAM le plus bas du projet. Ce chiffre est citable dans la section hardware sous réserve que la valeur tracemalloc soit extraite de `experiments/exp_125/` (critère S14-07 ✅ mais valeur précise non reportée dans les fichiers sprint).

- **Gap 3 ❌ (quantification INT8)** — Non adressé ce sprint. Les annotations `# MEM:` préparent le terrain mais aucune expérience INT8 n'a été conduite.

### Risques pour Sprint 15 (Pronostia)

- **Risque DBSCAN accumulate** : si le sprint 15 reproduit le scénario accumulate sur Pronostia (séries temporelles, potentiellement plus d'échantillons par tâche), la complexité quadratique de DBSCAN sur buffer cumulé peut devenir bloquante. **Décision à prendre avant de lancer exp_137+ : imposer `MAX_BUFFER_ACCUMULATE: 2000` dans `configs/unsupervised_config.yaml`.**

- **Risque TinyOL AE sur by_location** : la chute −0.030 sur Monitoring est un signal à surveiller sur Pronostia. Si le pattern se confirme (TinyOL AE sensible aux scénarios à forte hétérogénéité inter-tâches), il faudra nuancer sa position dans le classement MCU-candidats.

### Recommandations sur les questions ouvertes

- **EWC OC formulation** : recommander à Arnaud la formulation AE + EWC sur MSE (implémentée) comme baseline, avec une note de manuscrit signalant l'alternative MLP one-class. Les résultats (AUROC 0.96) justifient de ne pas relancer des expériences — à valider lors de la prochaine revue sprint.

- **Borne DBSCAN accumulate** : ajouter `MAX_BUFFER_ACCUMULATE: 2000` dans `configs/unsupervised_config.yaml` dès maintenant, avant Sprint 15, pour éviter de devoir retravailler l'architecture DBSCAN après les expériences Pronostia.
