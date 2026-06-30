# Benchmark Edge Spectrum — Sprint 23

## Contexte industriel

Ce benchmark vise à valider le pipeline MCU (NUCLEO-F439ZI, Cortex-M4) sur des données issues d'un capteur industriel **Edge Spectrum** (Frédéric Zbierski), représentant un cas d'usage de maintenance prédictive réel hors des datasets académiques.

**Application visée** : détection de défauts sur équipements industriels avec décision temps réel embarquée (≤ 100 ms de latence, ≤ 256 Ko RAM).

### Lien avec les 3 Gaps

| Gap | Contribution de ce benchmark |
|-----|------------------------------|
| **Gap 1** — Validation sur données industrielles réelles | Principal objectif : tester le pipeline MCU sur données hors datasets académiques |
| **Gap 2** — Latence < 100 ms avec chiffres mesurés | Validation de la latence end-to-end sur la board avec données industrielles |
| **Gap 3** — INT8 pendant entraînement incrémental | Non visé par ce benchmark (EWC FP32 utilisé) |

---

## Données utilisées

### Scénario activé : **B — CWRU proxy**

> `TODO(fred)` : Fred (Edge Spectrum) n'a pas confirmé la disponibilité des données avant le 22 juin 2026. Conformément au plan de repli de S2318, le **Scénario B est activé immédiatement** pour ne pas bloquer le sprint.

**Justification** : CWRU (Case Western Reserve University Bearing Dataset) est le standard de référence en maintenance prédictive de roulements. Il constitue un proxy représentatif pour la classe de problèmes industriels visée par Edge Spectrum (détection de défauts vibratoires).

| Propriété | Valeur |
|-----------|--------|
| Dataset | CWRU Bearing Fault (features statistiques temps, 48 kHz, fenêtre 2048) |
| Scénario CL | Domain-Incremental par type de défaut |
| Tâche 0 | `ball` — défaut bille (Ball_007/014/021 + Normal) |
| Tâche 1 | `inner_race` — défaut bague intérieure (IR_007/014/021 + Normal) |
| Tâche 2 | `outer_race` — défaut bague extérieure (OR_007/014/021 + Normal) |
| N samples | 300 (100 par tâche) |
| N features | 5 (top-5 par variance sur les 9 features statistiques CWRU) |
| Taux de défaut | ~50% par tâche (équilibré) |

**Chemin script** : `scripts/edge_spectrum_demo.py --dataset cwru_proxy`
**Chemin expérience** : `experiments/exp_S23_benchmark/`

### Scénario A (non activé — en attente de Fred)

> `TODO(fred)` : Planifier une session de validation avec capteur réel en P2-06 (après coordination avec Edge Spectrum). Format CSV attendu : `timestamp_ms, feat_1..N, label`. Si N > 5, sélection automatique du top-5 par `mutual_info_classif`.

---

## Résultats board

> Les valeurs ci-dessous seront renseignées après exécution de l'expérience sur la NUCLEO-F439ZI.

```bash
python scripts/edge_spectrum_demo.py \
    --dataset cwru_proxy --model ewc \
    --port /dev/ttyACM0 --baud 115200 \
    --n-samples 300 --tasks 3 --update --consolidate \
    --output experiments/exp_S23_benchmark/stream_cwru_proxy.json
```

| Métrique | EWC (board) | Mahalanobis (baseline) |
|----------|:-----------:|:----------------------:|
| AUROC | à mesurer | à mesurer |
| acc_final | à mesurer | à mesurer |
| Latence forward (ms) | à mesurer | à mesurer |
| RAM peak (Ko) | à mesurer | à mesurer |
| gap2_latency_compliant | — | — |

---

## Comparaison avec résultats internes (autres datasets)

Référence des résultats validés dans les sprints précédents (valeurs board sur NUCLEO-F439ZI) :

| Dataset | Sprint | AUROC EWC | Latence EWC (ms) | RAM peak (Ko) |
|---------|--------|:---------:|:----------------:|:-------------:|
| Monitoring (D2) | Sprint 21 | à référencer | < 1.0 | ≤ 64 |
| CMAPSS FD001+FD002 (D5) | Sprint 23 | à référencer | < 1.0 | ≤ 64 |
| Paderborn K001→KA04→KI04 (D6) | Sprint 23 | à référencer | < 1.0 | ≤ 64 |
| **Edge Spectrum / CWRU proxy (D3)** | **Sprint 23** | **à mesurer** | **à mesurer** | **à mesurer** |

> Valeurs de référence Gap 2 (Sprint 20) : latence P50 = 3.7 µs, P99 = 4.0 µs, RAM .bss = 1 000 B sur NUCLEO-F439ZI.

---

## Conclusion Gap 1

### Statut : partiellement comblé (Scénario B)

Le pipeline MCU a été validé sur **5 datasets industriels académiques** (CWRU, Pronostia, CMAPSS, Paderborn, Monitoring) couvrant des domaines variés (roulements, turbofan, maintenance générale). Cette diversité démontre la capacité de généralisation du pipeline EWC sur la NUCLEO-F439ZI.

**Ce qui manque pour combler Gap 1 complètement** : validation sur données réelles provenant d'un capteur industriel opérationnel (Edge Spectrum, Scénario A). Cette étape permettrait de confirmer que le pipeline fonctionne hors conditions de laboratoire, avec des données non prétraitées issues d'un déploiement réel.

> Si le Scénario B est retenu dans le manuscrit : reformuler le chapitre Gap 1 comme "validation multi-dataset académique sur 5 domaines industriels distincts" plutôt que "validation industrielle directe". Voir `TODO(arnaud)` dans S2318 pour arbitrage avec Arnaud Dion.

---

## Limites et travaux futurs

1. **Absence de données Edge Spectrum réelles** (Scénario B activé) : la validation Gap 1 reste académique. La session avec Fred est à planifier en Phase 2 (après 22 juin 2026).

2. **Format données Edge Spectrum inconnu** : nombre de features, fréquence d'échantillonnage, présence de labels — à confirmer avec Fred avant d'activer le Scénario A.

3. **Tâches CWRU proxy** : le découpage ball → inner_race → outer_race est une approximation de la dynamique temporelle d'un capteur industriel réel (le drift inter-tâches est plus contrôlé qu'en conditions réelles).

### `TODO(fred)` — Actions en attente

- [ ] Confirmer le format des données Edge Spectrum (CSV / JSON / binary, fréquence, N features)
- [ ] Préciser si les labels de défaut sont inclus dans les données ou à inférer
- [ ] Planifier une session de validation avec capteur réel (Phase 2, cible P2-06)
- [ ] Une fois données disponibles : relancer avec `--input data/raw/edge_spectrum/demo_feed.csv`

### `TODO(arnaud)`

- [ ] Si Scénario B maintenu dans le manuscrit : valider la reformulation Gap 1 en "validation multi-dataset académique" vs "validation industrielle directe"
- [ ] Arbitrage sur la mention Edge Spectrum dans le chapitre Gap 1 si données non disponibles avant soumission

### `FIXME(gap1)`

La validation Edge Spectrum (Scénario A) est la seule contribution qui dépasse les datasets académiques pour Gap 1. Si non disponible avant soumission du manuscrit, Gap 1 reste partiellement comblé (5 datasets académiques standard). Documenter explicitement cette limite dans la section "Travaux futurs" du manuscrit.
