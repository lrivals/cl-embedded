# Sprint 46 — Comparaison des moments de quantification (avant / après / les deux)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 46 |
| **Semaine** | 16 – 22 juillet 2026 |
| **Statut** | ✅ Implémenté (S4601–S4608) — PC/émulateur + **board réelle NUCLEO-F439ZI** pour la colonne `both` (S4608) |
| **Priorité globale** | 🔴 Critique — première étude qui compare **frontalement** les trois moments de quantification (QAT avant l'entraînement, PTQ après, et les deux enchaînés) sur les mêmes modèles/datasets/métriques. Consolide le message Gap 3 « quantifier ≠ quantifier : le *moment* et la *calibration* dominent ». |
| **Durée estimée totale** | ~32h (cadrage/taxonomie ~4h · harnais PC ~9h · expériences EWC+TinyOL ~8h · contexte HDC/Maha ~3h · figures+notebook ~4h · tests+docs ~3h · board différée ~1h doc) |
| **Dépendances** | Sprint 28 ✅ (QAT PC `EWCMlpInt8Classifier`, `benchmark_int8_fp32.py`) · Sprint 39 ✅ (émulateur PTQ bit-exact `int8_c_emulation.py`, sweeps) · Sprint 34 ✅ (Q15 Maha) · Sprint 24 ✅ (HDC/TinyOL INT8) · `src/figures/` registre ✅ (S4201) · loaders Monitoring/Pronostia ✅ |

## Contexte et motivation

Le Gap 3 a été abordé par **fragments dispersés dans le temps** : Sprint 28 a montré que le **QAT**
(fake-quant pendant l'entraînement) préserve la métrique (Δ≤0.006) ; Sprints 36/39 ont montré que la
**PTQ naïve** embarquée s'effondre (F1 0.07–0.15) ; Sprint 39 a montré qu'une **PTQ calibrée / per-canal**
récupère (émulateur bit-exact) ; Sprint 34 a montré Q15 pour Mahalanobis. **Aucune expérience ne compare
ces moments côte à côte**, sur les mêmes modèles, les mêmes datasets et la même métrique.

Ce sprint construit cette comparaison directe des **trois moments de quantification** :

- **Avant l'entraînement (QAT)** — fake-quant dans la boucle d'apprentissage (straight-through).
- **Après l'entraînement (PTQ)** — quantification des poids d'un modèle FP32 déjà entraîné.
- **Les deux (QAT → export PTQ)** — entraîner avec fake-quant puis exporter les poids appris à travers
  le noyau PTQ calibré du firmware. **C'est le chemin réel vers la carte** : le firmware ne fait jamais de
  fake-quant à l'inférence, il exécute un noyau entier ; « les deux » mesure donc ce que l'on déploie
  vraiment.

Périmètre : **EWC** en priorité, puis **TinyOL**, sur **Monitoring (D2)** et **Pronostia (D4)**.

## Décisions de cadrage (utilisateur, 16 juillet 2026)

- **3-way complet pour EWC + TinyOL** uniquement (ces deux modèles ont une vraie boucle
  d'entraînement à fake-quant → l'axe avant/après/les-deux a un sens).
- **HDC et Mahalanobis en contexte N/A honnête** : HDC est **nativement entier** (vecteurs int8 + AM
  int16 → la quantification est *structurelle*, métrique INT8 ≡ FP32 par construction, pas d'axe
  avant/après) ; Mahalanobis est **PTQ-only** (aucun entraînement) et son axe pertinent est **INT8 vs
  Q15** (récupération de Σ⁻¹, Sprint 34), pas le moment. On ne fabrique **aucune cellule 3-way
  artificielle** pour ces deux modèles.
- **« Les deux ensemble » = QAT puis export PTQ** : entraîner `EWCMlpInt8Classifier` (fake-quant), puis
  extraire ses poids et les passer dans `int8_c_emulation.forward_quant` (per-tensor calibré / per-canal).
- **PC (émulateur bit-exact) prioritaire, board différée** : la NUCLEO est indisponible (cf. S39/S40) ;
  les cellules board portent `« à mesurer »`, aucun chiffre inventé.
- **Aucun chiffre en dur** : toute valeur sort d'un run de script ; les tables de résultats de ce doc
  portent `pending` tant que le harnais n'a pas tourné.
- **Langue** : français.

## Nœud honnête : ce que la comparaison mesure et ce qu'elle ne prétend pas

Comparer les moments de quantification **ne redéfinit pas** ce qu'est un bon modèle CL — les métriques
FP32 de référence sont établies (Sprints 22–36). Ce sprint isole **une seule variable** : *quand* et
*comment* on quantifie, à modèle/données/seed fixés. Il ne prétend pas non plus que les quatre modèles
sont comparables sur le même axe : **seuls EWC et TinyOL ont un axe avant/après**. HDC et Maha sont
documentés pour **cadrer** (pourquoi l'axe ne s'applique pas), pas pour remplir une grille. Enfin, la
colonne « les deux » est la seule **fidèle au déploiement** (noyau entier réel) ; « avant » (fake-quant à
l'inférence) est une **borne haute optimiste** que la carte n'atteint pas telle quelle — ce sprint rend
cet écart explicite.

## Tâches

### Bloc A — Cadrage & taxonomie

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4601 | **Taxonomie des 3 moments + mapping par modèle** (before=QAT, after=PTQ, both=QAT→export PTQ ; EWC/TinyOL = 3-way, HDC structurel N/A, Maha INT8-vs-Q15) ; réconciliation voie/métrique de référence par modèle ; introduction de la clé config `quant_moment ∈ {before, after, both}` | 🔴 | `docs/sprints/sprint_46/S4601_cadrage_taxonomie.md`, `docs/context/quantization_moments.md` | 📝 Doc |

### Bloc B — Harnais PC

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4602 | **Harnais unifié `run_s46_quant_moment.py`** : itère (modèle × dataset × moment), réutilise `train_ewc`+`EWCMlpInt8Classifier` (before), `int8_c_emulation.forward_quant` (after), et **câble le chemin `both`** (QAT-train → `from_state_dict` → `forward_quant`) ; métriques natives + RAM poids + proxy latence ; schéma JSON aligné S28/S39 | 🔴 | `scripts/run_s46_quant_moment.py` | 📝 Doc |

### Bloc C — Expériences PC

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4603 | **EWC 3-way** × {Monitoring, Pronostia} × {fp32, before, after, both} + configs | 🔴 | `configs/quant_moment/ewc_*.yaml`, `experiments/exp_S46_ewc/` | 📝 Doc |
| S4604 | **TinyOL 3-way** × {Monitoring, Pronostia} (fake-quant online + calibration PTQ) | 🟠 | `configs/quant_moment/tinyol_*.yaml`, `experiments/exp_S46_tinyol/` | ✅ |
| S4605 | **Contexte HDC + Maha** (N/A honnête : HDC structurel INT8≡FP32 ; Maha INT8 vs Q15) — pas de cellule 3-way artificielle | 🟠 | `experiments/exp_S46_context/` | ✅ |

### Bloc D — Assemblage & clôture

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4606 | **Figures + notebook** : catalogue `quant_moment.py` (registre S4201) → PNG `docs/figures/quantization_moment/` (barres moment×modèle×dataset, heatmap, N/A gris, 0 chiffre en dur) + notebook galerie | 🟠 | `src/figures/catalogs/quant_moment.py`, `docs/figures/quantization_moment/`, `notebooks/cl_eval/quant_moment/comparison.ipynb` | ✅ |
| S4607 | **Tests + docs** : `test_s46_quant_moment.py` (structure JSON, 3 moments, N/A honnête, garde 0-chiffre-en-dur) + MAJ roadmap/triple_gap + `graphify_sprint_update` | 🟡 | `tests/test_s46_quant_moment.py`, `docs/roadmap_phase2.md`, `docs/triple_gap.md` | 📝 Doc |

### Bloc E — Board différée

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4608 | **Board différée** : flasher la colonne `both` (QAT→export PTQ via `export_weights_c.py`), mesurer latence DWT/`.bss`/parité board↔PC → `« à mesurer »` tant que NUCLEO indisponible | 🟢 | `experiments/exp_S46_board/` (différé) | 📝 Doc (board différée) |

## Ordre d'exécution recommandé

```
S4601 (taxonomie + mapping + clé quant_moment)
   │
   ▼
S4602 (harnais PC : before / after / both câblés)
   │
   ├──► S4603 (EWC 3-way × 2 datasets)  ── prioritaire
   ├──► S4604 (TinyOL 3-way × 2 datasets)
   └──► S4605 (HDC/Maha contexte N/A)
                 │
                 ▼
         S4606 (figures + notebook)
                 │
                 ▼
         S4607 (tests + roadmap + triple_gap)
                 │
                 ▼
         S4608 (board différée « à mesurer »)
```

Le chemin sans carte s'arrête à S4607 (tout PC/émulateur). S4608 reste faisable dès l'accès NUCLEO.

## Sources de données (Sprint 46, lecture seule)

| Dataset | Loader / scénario CL | Rôle Sprint 46 |
| ------- | -------------------- | -------------- |
| Monitoring (D2) | `get_cl_dataloaders` — domain-incrémental, 3 tâches (Pump→Turbine→Compressor) | Colonnes EWC + TinyOL 3-way |
| Pronostia (D4) | `get_pronostia_dataloaders` — domain-incrémental par condition, 3 tâches | Colonnes EWC + TinyOL 3-way |

Configs de référence réutilisées : `configs/ewc_int8_{monitoring,pronostia}.yaml` (voie QAT),
`configs/quant_intermediate/*_{monitoring,pronostia}.yaml` (voie PTQ), `configs/mahalanobis_{int8,q15}_{monitoring,pronostia}.yaml` (contexte Maha).

## Livrables

1. `docs/context/quantization_moments.md` — taxonomie des 3 moments + mapping par modèle (S4601).
2. `scripts/run_s46_quant_moment.py` — harnais unifié 3-way, incluant le chemin **both** (S4602).
3. `configs/quant_moment/{ewc,tinyol}_{monitoring,pronostia}.yaml` — configs par (modèle, dataset)
   portant la clé `quant_moment`.
4. `experiments/exp_S46_ewc/`, `exp_S46_tinyol/`, `exp_S46_context/` — résultats JSON (métrique, RAM,
   proxy latence, delta vs fp32) par moment.
5. `src/figures/catalogs/quant_moment.py` → PNG `docs/figures/quantization_moment/` + notebook galerie.
6. `tests/test_s46_quant_moment.py` — tests structure/honnêteté.
7. MAJ `docs/roadmap_phase2.md` + `docs/triple_gap.md` (§ Gap 3).

## Questions ouvertes

- `TODO(dorra)` : le QAT actuel (`EWCMlpInt8Classifier`) utilise un fake-quant **always-on** sans
  `prepare_qat`/gel d'observateurs/warm-up FP32. Faut-il un QAT canonique complet, ou le fake-quant
  léger suffit-il comme borne « avant » ? (impacte l'interprétation de la colonne `before`).
- `TODO(arnaud)` : métrique de référence par modèle — EWC AUROC binaire (voie S28) vs F1 multiclasse
  (voie S39). S4601 fixe la voie ; confirmer que l'AUROC binaire est la bonne référence pour le message.
- `TODO(dorra)` : « les deux » sur board — le noyau PTQ exporté depuis des poids QAT récupère-t-il le
  F1 mieux que la PTQ depuis des poids FP32 ? (à confirmer S4608 quand la carte est disponible).

## Bilan (à compléter)

| Tâche | Statut | Temps réel | Notes |
|-------|:------:|:----------:|-------|
| S4600 | 📝 Doc | — | Overview + cadrage |
| S4601 | ✅ | — | Taxonomie 3 moments + clé `quant_moment` |
| S4602 | ✅ | — | Harnais `run_s46_quant_moment.py` (chemin `both` câblé) |
| S4603 | ✅ | — | EWC 3-way × 2 ds mesuré (`exp_S46_ewc/`) |
| S4604 | ✅ | — | TinyOL 3-way × 2 ds mesuré (`exp_S46_tinyol/`) ; collapse recon-error honnête |
| S4605 | ✅ | — | Contexte HDC/Maha N/A mesuré (`exp_S46_context/`) ; mode `--moment context` |
| S4606 | ✅ | — | Catalogue figures `quant_moment` (4 PNG) + notebook galerie ; garde AST étendue |
| S4607 | ✅ | — | `test_s46_quant_moment.py` **9 PASS** (schéma 4 moments, N/A HDC/Maha, câblage `both`, garde 0-chiffre, déterminisme QAT) + MAJ roadmap/triple_gap/CLAUDE.md ; `pytest -k "quant or figures"` 38 PASS 0 régression |
| S4608 | ✅ (board réelle) | — | Colonne `both` mesurée NUCLEO-F439ZI : head **QAT multiclasse** `EWCMlpMulticlassInt8` → kernel v2 calibré → **F1 0.9213/0.9072, parité 1.000, lat 65/68 µs (Gap 2 ✅), `.bss` 101/106 Ko (Gap 3 ✅, ÷4), 0 CRC ; A/B `both` ≥ `after` (+0.004/+0.008)** |
