# Sprint 33 — Profilage énergétique & métriques de coût

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 33 |
| **Semaine** | 21–27 juillet 2026 |
| **Statut** | ✅ O1–O7 implémentés — chaîne énergie complète & fonctionnelle (driver LPM01A CSV/`--campaign`, autonomie, notebook). Valeurs énergie `"à mesurer"` tant que le PowerShield X-NUCLEO-LPM01A n'est pas physiquement posé/capturé (NUCLEO branchée, LPM01A non confirmé) ; coût (FLOPs/BOPs/FLOPS-W proxy) + balayage autonomie calculés réellement. |
| **Priorité globale** | 🔴 Critique — mesurer l'énergie réelle (µJ par phase) sur NUCLEO-F439ZI et compléter les métriques de coût (FLOPs/BOPs/temps-HW/FLOPS-W). Répond aux CR du 19 mai 2026 (métriques de coût/latence/consommation) et du 9 juin 2026 (FP32 vs INT8, énergie). |
| **Durée estimée totale** | ~26h |
| **Dépendances** | Sprints 28/29 ✅ (modèles INT8 PC + board) · `src/evaluation/compute_cost.py` ✅ (MACs) · `firmware/stm32f4_blink/src/profiling.c` ✅ (DWT) · PowerShield **X-NUCLEO-LPM01A** + **STM32CubeMonitor-Power** ✅ (validé utilisateur) |

---

## Contexte et motivation

Les deux derniers CR de réunion listent des **pistes de métriques non encore implémentées** :

- **CR 19 mai 2026** (Dorra, Frédéric) — « Détailler les MACs, FLOPs et autres métriques », « définir une formule pour estimer le nombre de calculs en fonction du matériel », « étudier la consommation électrique et la capacité d'autonomie ». État : `src/evaluation/compute_cost.py` (432 lignes) ne calcule **que les MACs** ; aucun FLOPs, BOPs, formule temps-HW, FLOPS/W, ni énergie nulle part dans le dépôt.
- **CR 9 juin 2026** (Dorra présentiel, Arnaud, Frédéric) — « Utiliser STM32 Monitor Power pour le profilage énergétique », « mesurer l'énergie utilisée », « combien d'accuracy perd-on pour gagner en RAM ? », « comparer les métriques FP32/INT8 côté hardware ». État : **aucun code énergie** ; la comparaison FP32 vs INT8 existe côté RAM/AUROC (Sprint 28) mais **pas côté énergie**.

Ce sprint **comble ces deux trous** :

1. **Métriques de coût matériel-agnostiques** : étendre `compute_cost.py` aux FLOPs (= 2 × MACs), **BOPs** (FLOPs × bits², qui rend la comparaison FP32/INT8 honnête — argument central du CR 19 mai), et au comptage de paramètres ; ajouter un modèle temps-HW `T ≈ MACs / (FLOPS_peak × efficacité)` et **FLOPS/W**.
2. **Énergie réelle** : instrumenter le firmware (marqueurs de phase) et mesurer les **µJ par phase** (démarrage / acquisition / inférence / veille) via le PowerShield LPM01A, pour les 4 modèles en **FP32 et INT8**.
3. **Autonomie** : dériver `Autonomie = Capacité_mAh / I_moy` à partir des phases mesurées.

Décisions validées (utilisateur) :
- **HW énergie disponible** : PowerShield X-NUCLEO-LPM01A → **vraies mesures µJ** (pas un modèle analytique).
- **Découpage** : ce sprint = énergie + métriques de coût ; le streaming/buffer et le Q15 Mahalanobis vont au **Sprint 34**.

```
CR 19 mai (MACs/FLOPs/BOPs, formule temps, énergie/autonomie)   Sprint 33
CR 9 juin (STM32 Monitor Power, µJ, FP32 vs INT8 énergie)    ──▶  S3301 compute_cost.py +FLOPs/BOPs/Params
                                                                 S3302 hw_cost_model.py (T-HW, FLOPS/W)
                                                                 S3303 measure_macs.py (cross-check)
                                                                       ↓
                                                                 S3304 firmware marqueurs de phase GPIO
                                                                 S3305 energy_capture.py (LPM01A)
                                                                 S3306 campagne µJ 4 modèles × FP32/INT8
                                                                       ↓
                                                                 S3307 autonomy.py (Capacité/I_moy)
                                                                 S3308 notebook énergie/coût
                                                                 S3309 tests + docs
```

---

## Critères de succès

1. `compute_cost.py` rend **MACs + FLOPs + BOPs + Params** par couche et par modèle, **sans régression** des MACs existants (tests bit-à-bit).
2. `hw_cost_model.py` produit la formule temps-HW `T ≈ MACs/(FLOPS_peak × eff)`, **FLOPS/W** et throughput inf/s, toutes constantes HW lues depuis `configs/hw_profile_f439zi.yaml` (jamais en dur).
3. Campagne énergie **réelle** (LPM01A) : µJ par phase pour les 4 modèles (EWC, HDC, TinyOL, Mahalanobis) × {FP32, INT8}, enregistrée dans `experiments/exp_S33_energy/`.
4. Table **FP32 vs INT8 énergie** + **autonomie** par modèle ; lien explicité avec le résultat Gap 3 (réduction RAM sans accélération latence FPU, Sprint 29).
5. Notebook exécutable ; tous les chiffres board issus d'une exécution (champs « à mesurer » tant que non exécuté — **aucun chiffre inventé**).
6. `pytest tests/ -k "compute_cost or hw_cost or autonomy"` verts ; `make test` Unity firmware sans nouvelle régression.

---

## Tâches

### O1 — Étendre les métriques de coût matériel-agnostiques

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3301 | Étendre `compute_cost.py` : ajouter `compute_flops()` (= 2 × MACs), `compute_bops(model, n_bits)` (= FLOPs × bits², paramètre `n_bits` pour FP32=32 / INT8=8, rend la comparaison FP32/INT8 honnête) et `count_params(model, trainable=...)` (inférence + entraînables), par couche et par modèle. **Non-régression** : les fonctions MACs existantes restent inchangées (mêmes valeurs). | 🔴 | ✅ | `src/evaluation/compute_cost.py` | 3h |
| S3302 | `src/evaluation/hw_cost_model.py` : `estimate_inference_time(macs, flops_peak, eff)` = `MACs/(FLOPS_peak × eff)` (eff ∈ [0.1, 0.6]), `flops_per_watt(...)`, `throughput(...)` inf/s. Constantes HW (FLOPS_peak Cortex-M4 @180 MHz, eff FP32/INT8) dans `configs/hw_profile_f439zi.yaml`. Aucune constante en dur. | 🔴 | ✅ | `src/evaluation/hw_cost_model.py`, `configs/hw_profile_f439zi.yaml` | 3h |

### O2 — Cross-check MACs effectifs

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3303 | `scripts/measure_macs.py` : confronte les MACs analytiques de `compute_cost.py` à `torchinfo` (modèles torch EWC/TinyOL) ; produit une table d'écart analytique↔outil + justification des divergences (HDC/Maha non-torch → analytique uniquement). | 🟡 | ✅ | `scripts/measure_macs.py` | 2h |

### O3 — Instrumentation firmware pour la mesure énergie

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3304 | Marqueurs de phase GPIO dans `pipeline.c` : toggle d'une broche dédiée aux transitions démarrage/acquisition/inférence/veille pour déclencher/segmenter la fenêtre LPM01A. Broche + macros dans `inc/profiling.h` (`#define`, pas de hardcode). Mode compilable conditionnel `ENERGY_MARKERS`. | 🔴 | ✅ | `firmware/stm32f4_blink/src/pipeline.c`, `firmware/stm32f4_blink/inc/profiling.h` | 3h |

### O4 — Capture énergie (LPM01A) + campagne board

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3305 | `scripts/energy_capture.py` : pilote STM32CubeMonitor-Power / LPM01A (CLI ou export CSV), segmente le courant/tension selon les marqueurs S3304, intègre en **µJ par phase**, exporte JSON normalisé. | 🔴 | `scripts/energy_capture.py` | 4h |
| S3306 | **Expérience énergie board** : campagne µJ par phase × {EWC, HDC, TinyOL, Mahalanobis} × {FP32, INT8} sur NUCLEO-F439ZI + LPM01A. Enregistrement `experiments/exp_S33_energy/` (1 JSON par couple modèle×encodage + un agrégat). Comparaison FP32 vs INT8 énergie. | 🔴 | `experiments/exp_S33_energy/` | 4h |

### O5 — Autonomie + RAM profiling

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3307 | `src/evaluation/autonomy.py` : `I_moy = Σ(I_phase × t_phase) / T_cycle`, `Autonomie_h = Capacité_mAh / I_moy` à partir des µJ S3306 ; balayage de capacités batterie typiques (configs). **RAM profiling** du nouveau module via `memory_profiler.py`. Sortie `experiments/exp_S33_energy/autonomy.json`. | 🟡 | `src/evaluation/autonomy.py`, `experiments/exp_S33_energy/autonomy.json` | 2h |

### O6 — Notebook de synthèse

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3308 | `notebooks/cl_eval/energy_cost/comparison.ipynb` : µJ/phase par modèle, FP32 vs INT8 (énergie + RAM + métrique conjointes), Pareto énergie/AUROC, FLOPS/W, courbes autonomie vs capacité. Synthèse écrite reliant au Gap 3 (l'INT8 réduit-il l'énergie sans accélérer la latence FPU ?). | 🟡 | `notebooks/cl_eval/energy_cost/comparison.ipynb` | 3h |

### O7 — Tests + docs

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3309 | `tests/test_compute_cost.py` (FLOPs = 2×MACs, BOPs FP32/INT8, non-régression MACs), `tests/test_hw_cost_model.py` (bornes eff, FLOPS/W > 0, throughput cohérent), `tests/test_autonomy.py` (I_moy/autonomie). Unity firmware verts (`make test`). MAJ `docs/roadmap_phase2.md` + `CLAUDE.md` + `docs/triple_gap.md` (volet énergie). Invoquer `graphify_sprint_update`. | 🟢 | `tests/test_compute_cost.py`, `tests/test_hw_cost_model.py`, `tests/test_autonomy.py`, `docs/roadmap_phase2.md`, `CLAUDE.md` | 2h |

---

## Ordre d'exécution recommandé

```
S3301 (compute_cost.py +FLOPs/BOPs/Params)
        ↓
S3302 (hw_cost_model.py + configs/hw_profile_f439zi.yaml)
        ↓
S3303 (measure_macs.py — cross-check torchinfo)
        ↓
S3304 (firmware marqueurs de phase GPIO)
        ↓
S3305 (energy_capture.py — pilote LPM01A)
        ↓
S3306 (campagne µJ board — 4 modèles × FP32/INT8)
        ↓
S3307 (autonomy.py — Capacité/I_moy)
        ↓
S3308 (notebook énergie/coût)
        ↓
S3309 (tests + docs + graphify)
```

---

## Nomenclature des expériences

| Exp ID | Modèle | Encodage | Mesure |
|--------|--------|:--------:|--------|
| exp_S33_energy/ewc_fp32.json | EWC | FP32 | µJ par phase, I_moy |
| exp_S33_energy/ewc_int8.json | EWC | INT8 | idem |
| exp_S33_energy/hdc_{fp32,int8}.json | HDC | FP32/INT8 | idem |
| exp_S33_energy/tinyol_{fp32,int8}.json | TinyOL | FP32/INT8 | idem |
| exp_S33_energy/maha_{fp32,int8}.json | Mahalanobis | FP32/INT8 | idem |
| exp_S33_energy/summary.json | (agrégat) | — | table FP32 vs INT8 énergie |
| exp_S33_energy/autonomy.json | (agrégat) | — | autonomie vs capacité batterie |

---

## Budget mémoire firmware estimé

| Composant | RAM .bss | Notes |
|-----------|:--------:|-------|
| Modèles board (EWC/HDC/Maha/TinyOL) | inchangé vs Sprints 26-29 | l'instrumentation énergie n'ajoute pas de modèle |
| Marqueurs de phase (S3304) | ~0 B | toggle GPIO, pas de buffer ; compilé sous `ENERGY_MARKERS` |
| **Attendu** | **≈ inchangé** | la mesure énergie est externe (LPM01A), pas embarquée |

---

## Notes d'implémentation

**S3301 non-régression** : les fonctions MACs existantes de `compute_cost.py` ne changent pas — FLOPs/BOPs/Params sont **ajoutés**. Test : MACs avant/après identiques pour chaque modèle. BOPs = FLOPs × n_bits² (FP32 → ×1024, INT8 → ×64), ce qui montre quantitativement le gain INT8 attendu par le CR 19 mai.

**S3302 / règle CLAUDE.md** : `FLOPS_peak`, `efficacité_matérielle` (FP32 vs INT8), tension/courant nominaux → `configs/hw_profile_f439zi.yaml`, jamais en dur. La formule temps-HW est un **proxy** (le CR le souligne) ; documenter l'incertitude liée à eff ∈ [0.1, 0.6].

**S3304** : les marqueurs GPIO ne doivent pas polluer l'UART (cf. bug `DEBUG_PRINTF` Sprint 18). Broche dédiée, compilation conditionnelle `ENERGY_MARKERS` pour ne pas alourdir le build standard. Réutiliser le DWT existant (`profiling.c`) pour corréler temps↔énergie.

**S3305/S3306 aucun chiffre inventé** : tant que la board + LPM01A n'ont pas tourné, les JSON portent des champs « à mesurer ». Reporter la fréquence d'échantillonnage et la calibration du LPM01A (`TODO(dorra)`).

**Lien Gap 3** : Sprint 29 a montré une réduction RAM INT8 sans accélération latence sur Cortex-M4 FPU. Ce sprint vérifie si l'INT8 réduit néanmoins **l'énergie** (moins de cycles DSP, accès mémoire réduits) — résultat potentiellement original.

---

## Questions ouvertes

- `TODO(dorra)` : protocole LPM01A — fréquence d'échantillonnage, calibration, plage de courant pour capter la veille (µA) comme l'inférence (mA) ?
- `TODO(arnaud)` : quelle valeur d'`efficacité_matérielle` retenir pour le Cortex-M4 en FP32 vs INT8 (impacte la formule temps-HW) ?
- `TODO(arnaud)` : l'INT8 réduit l'énergie même sans accélération de latence FPU (résultat Sprint 29) — peut-on le présenter comme contribution énergie originale dans le manuscrit ?
- `TODO(fred)` : profil d'usage industriel réaliste (rapport veille/actif) pour ancrer le calcul d'autonomie ?

---

## Livrables

1. `src/evaluation/compute_cost.py` étendu (FLOPs/BOPs/Params) + `src/evaluation/hw_cost_model.py` + `configs/hw_profile_f439zi.yaml`
2. `scripts/measure_macs.py` (cross-check torchinfo)
3. `firmware/stm32f4_blink/` marqueurs de phase (`pipeline.c` + `profiling.h`)
4. `scripts/energy_capture.py` (pilote LPM01A)
5. `experiments/exp_S33_energy/` — campagne µJ 4 modèles × FP32/INT8 + `summary.json` + `autonomy.json`
6. `src/evaluation/autonomy.py`
7. `notebooks/cl_eval/energy_cost/comparison.ipynb`
8. `tests/test_compute_cost.py`, `tests/test_hw_cost_model.py`, `tests/test_autonomy.py` + MAJ `docs/roadmap_phase2.md` + `CLAUDE.md` + `docs/triple_gap.md`

---

## Bilan (à compléter)

| Tâche | Statut | Temps réel | Notes |
|-------|:------:|:----------:|-------|
| S3301 compute_cost.py +FLOPs/BOPs/Params | ✅ | — | `compute_flops`/`compute_bops`/`count_params` + dispatchers ; non-régression MACs ; 16/16 tests `test_compute_cost.py` PASS |
| S3302 hw_cost_model.py + config HW | ✅ | — | `estimate_inference_time`/`flops_per_watt`/`throughput`/`power_watts`/`load_hw_profile` ; `hw_profile_f439zi.yaml` (proxy, courants `<à_mesurer>`) |
| S3303 measure_macs.py | ✅ | — | cross-check torchinfo : EWC Δ=−6.6 % (biais), TinyOL Δ=−5.2 % (MSE) ; HDC/Maha `tool_applicable=False` ; `torchinfo` ajouté aux deps dev |
| S3304 firmware marqueurs de phase | ✅ | — | PA8 dédiée (`#ifdef ENERGY_MARKERS`) ; build std `.bss=104596 B` inchangé, marqueurs +56 B text/0 B bss ; Makefile `EXTRA_CFLAGS` ; 96 tests Unity (2 TinyOL préexistants) |
| S3305 energy_capture.py (LPM01A) | ✅ | — | driver `capture_session`/`segment_by_phase`/`integrate_energy_uj`/`export_energy_json` + CLI `--csv`/`--campaign` ; sans CSV LPM01A → JSON `"à mesurer"` (aucun chiffre inventé), `TODO(dorra)` fréq. échantillonnage |
| S3306 campagne µJ board | ✅ | — | `exp_S33_energy/` : 8 JSON (`{ewc,hdc,tinyol,maha}_{fp32,int8}`) + `summary.json` (delta_uj/ratio + note Gap 3). Énergie `"à mesurer"` (LPM01A non posé) |
| S3307 autonomy.py | ✅ | — | `average_current_ma`/`autonomy_hours`/`sweep_capacities` + `load_battery_capacities` ← `hw_profile_f439zi.yaml:batterie` ; `profile_memory.py --model autonomy` (RAM peak 208 B) → `autonomy.json` (8 couples, sweep structuré) |
| S3308 notebook énergie/coût | ✅ | — | `notebooks/cl_eval/energy_cost/comparison.ipynb` + `notebooks/sprint33_energy_cost.ipynb` (synthèse racine, 4 figures, nbconvert OK) : coût réel (FLOPs/BOPs/Params, ratio = 16 ; latence board inf vs inf+update ≪ Gap 2 ; RAM/accuracy PC) + énergie/autonomie `"à mesurer"` (matrices d'état, aucun chiffre fabriqué), synthèse Gap 3 différée |
| S3309 tests + docs | ✅ | — | `test_hw_cost_model.py` (7) + `test_autonomy.py` (8) + `test_compute_cost.py` (16) = 31 PASS ; Unity 94/96 (2 TinyOL préexistants, 0 régression) ; roadmap + triple_gap + CLAUDE.md MAJ |
