# Sprint 47 — Profondeur & schéma de quantification pour EWC (sub-INT8, granularité, symétrie)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 47 |
| **Semaine** | 23 – 29 juillet 2026 |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Priorité globale** | 🔴 Critique — ouvre le **second axe** de l'exploration quantification pour EWC, **orthogonal** au Sprint 46. S46 répond à *quand* quantifier (avant/après/les-deux) ; S47 répond à *à quelle profondeur et avec quel schéma* : **jusqu'où descendre en bits (sub-INT8) avant que EWC casse, et quelle calibration (granularité / symétrie) rachète la métrique**. Consolide le message Gap 3 « quantifier ≠ quantifier ». |
| **Durée estimée totale** | ~30h (cadrage/taxonomie ~4h · extension émulateur + harnais ~8h · sweep profondeur×granularité ~7h · axe symétrie ~3h · contexte HDC/Maha ~2h · figures+notebook ~4h · tests+docs ~2h) |
| **Dépendances** | Sprint 39 ✅ (émulateur bit-exact `int8_c_emulation.py` — `n_bits` et granularité **déjà paramétrés**) · Sprint 28 ✅ (QAT PC, benchmark) · Sprint 34 ✅ (Q15 = 16-bit, borne haute) · `src/utils/quantization.py` ✅ (affine zero-point) · `src/figures/` registre ✅ (S4201) · loaders Monitoring/Pronostia ✅ · **Sprint 46** (axe moment — S47 est le complément) |

## Contexte et motivation

Le projet a exploré la quantification par **fragments** :

- **Formats** (`docs/context/quantization_strategies.md`, S4202) : `int8_ptq_legacy`, `int8_v2` per-tensor/per-channel, `q15`, `int16_am` — tous à **8 ou 16 bits**.
- **Moments** (Sprint 46) : QAT (avant) / PTQ (après) / QAT→PTQ (les-deux).

**Aucune étude n'a exploré la profondeur en bits en-dessous de 8**, ni comparé frontalement la granularité
(per-tensor vs per-channel) et la symétrie (signé symétrique vs affine à zero-point) comme **variables d'un
même balayage**. Or c'est la question directe du Gap 3 RAM : l'INT8 donne ÷4, mais **jusqu'où peut-on
descendre** (INT4 = ÷8, INT2 = ÷16) avant que la tête EWC perde sa métrique, et **quel schéma repousse ce mur** ?

Ce sprint construit ce balayage **profondeur × granularité × symétrie** :

- **Profondeur** : `weight_bits ∈ {8, 6, 4, 3, 2, ternaire, binaire-poids}` (l'émulateur `int8_c_emulation.py`
  paramétrise déjà `n_bits` dans `_weight_scales`/`_quant_weight`/`_act_params` — seul `QuantConfig` doit l'exposer).
- **Granularité** : `per_tensor` vs `per_channel` (déjà supportées par l'émulateur) — hypothèse : la per-channel
  repousse le « cliff » de plusieurs bits.
- **Symétrie** : signé symétrique (`round(w/s)`, actuel) vs **affine à zero-point** (`round(w/s)+z`, réutilise
  `src/utils/quantization.py::compute_scale_zero_point`) — pertinent surtout pour les **activations post-ReLU
  asymétriques** (≥ 0).

Périmètre : **EWC uniquement**, sur **Monitoring (D2)** et **Pronostia (D4)** (décision utilisateur).

## Décisions de cadrage (utilisateur, 17 juillet 2026)

- **Axe retenu : profondeur + schéma**, complémentaire (orthogonal) à l'axe *moment* du Sprint 46.
- **EWC-only × {Monitoring, Pronostia}** — pas de grille multi-modèle. HDC et Mahalanobis sont documentés
  en **contexte N/A honnête** (S4705) : HDC est **nativement entier** (pas d'axe de profondeur de poids par
  scale), Mahalanobis est **format-only** (son axe est INT8-vs-Q15 pour Σ⁻¹, Sprint 34) et n'a pas de tête
  neuronale à balayer en bits. **Aucune cellule sub-INT8 artificielle** pour ces modèles.
- **PC (émulateur bit-exact) prioritaire** : tout le sweep tourne sans carte. Le portage board est le **Sprint 48**.
- **Métrique de référence EWC** : **AUROC binaire** (voie QAT S28), reprise de la décision S4601 pour cohérence
  avec l'axe moment (`TODO(arnaud)` à confirmer).
- **Aucun chiffre en dur** : toute valeur sort d'un run de script ; les tables de résultats portent `pending`
  tant que le harnais n'a pas tourné.
- **Langue** : français.

## Nœud honnête : ce que le balayage mesure et ce qu'il ne prétend pas

Descendre en bits **ne redéfinit pas** ce qu'est un bon modèle CL — les métriques FP32 de référence sont
établies (Sprints 22–36). Ce sprint isole **la profondeur de quantification et son schéma**, à
modèle/données/seed fixés. Deux nuances d'honnêteté à porter dans les figures et le texte :

1. **RAM théorique vs RAM matérialisée** : un poids INT4 **stocké dans un conteneur `int8`** n'économise
   **rien de plus** que l'INT8. Le gain **÷8 (INT4) / ÷16 (INT2)** n'est réel qu'avec un **kernel bit-packé**
   (2 poids/octet en INT4, 4 poids/octet en INT2). Sur PC (émulateur), la RAM est **théorique** (calculée
   depuis `weight_bits`) ; la **RAM `.bss` mesurée** est l'affaire du Sprint 48. Les figures S47 étiquettent
   la RAM « théorique (bit-packée) ».
2. **L'émulateur ne mesure ni latence ni RAM réelles** — il mesure la **métrique** (AUROC) et une **RAM
   analytique**. La latence sub-INT8 (dépacking + MAC FPU) est un `TODO(dorra)` tranché board (S48).

## Tâches

### Bloc A — Cadrage & taxonomie

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4701 | **Taxonomie profondeur/schéma + mapping EWC-only** (bit-widths 8→1 + ternaire ; granularité per-tensor/per-channel ; symétrie signé/affine-zero-point ; RAM théorique vs bit-packing ; orthogonalité avec l'axe moment S46) ; introduction des clés config `weight_bits`, `granularity`, `symmetry` | 🔴 | `docs/context/quantization_depth.md`, `docs/sprints/sprint_47/S4701_cadrage_taxonomie.md` | 📝 Doc |

### Bloc B — Harnais PC

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4702 | **Extension émulateur + harnais `run_s47_quant_depth.py`** : exposer `n_bits`/ternaire/binaire dans `QuantConfig` (preset `subint8(bits, granularity, symmetry)`), brancher le zero-point affine (réutilise `src/utils/quantization.py`), itérer (dataset × bits × granularité × symétrie) ; métrique AUROC + RAM théorique + proxy latence ; schéma JSON aligné S28/S39 | 🔴 | `src/utils/int8_c_emulation.py` (extension), `scripts/run_s47_quant_depth.py` | 📝 Doc |

### Bloc C — Expériences PC

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4703 | **Sweep profondeur × granularité** : EWC × {Monitoring, Pronostia} × bits {8,6,4,3,2,ternaire,binaire} × {per-tensor, per-channel} + configs → courbe métrique-vs-bits, **identification du « cliff »**, ratio RAM théorique | 🔴 | `configs/quant_depth/ewc_*.yaml`, `experiments/exp_S47_depth/` | 📝 Doc |
| S4704 | **Axe symétrie / zero-point** : signé symétrique vs affine (activations post-ReLU ≥ 0) aux bits critiques identifiés en S4703 | 🟠 | `configs/quant_depth/ewc_sym_*.yaml`, `experiments/exp_S47_symmetry/` | 📝 Doc |
| S4705 | **Contexte HDC / Maha / TinyOL** (N/A honnête : HDC structurellement entier, Maha format-only cf. S34, portée limitée à EWC) — pas de cellule sub-INT8 artificielle | 🟠 | `experiments/exp_S47_context/` | 📝 Doc |

### Bloc D — Assemblage & clôture

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4706 | **Figures + notebook** : catalogue `quant_depth.py` (registre S4201) → PNG `docs/figures/quantization_depth/` (courbe AUROC-vs-bits par granularité, heatmap bits×granularité, RAM théorique vs bits, symétrie aux bits critiques ; N/A gris, garde AST 0-chiffre-en-dur) + notebook galerie | 🟠 | `src/figures/catalogs/quant_depth.py`, `docs/figures/quantization_depth/`, `notebooks/cl_eval/quant_depth/comparison.ipynb` | 📝 Doc |
| S4707 | **Tests + docs** : `test_s47_quant_depth.py` (structure JSON, dégradation monotone bits↓, N/A honnête, garde 0-chiffre-en-dur) + MAJ roadmap/triple_gap + `graphify_sprint_update` | 🟡 | `tests/test_s47_quant_depth.py`, `docs/roadmap_phase2.md`, `docs/triple_gap.md` | 📝 Doc |

### Bloc E — Pointeur board

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4708 | **Cadrage board (renvoi Sprint 48)** : sélectionne les configs gagnantes (bits les plus bas préservant l'AUROC × granularité × symétrie) à porter sur NUCLEO au Sprint 48 | 🟢 | `docs/sprints/sprint_47/S4708_pointeur_board.md` (renvoi S48) | 📝 Doc |

## Ordre d'exécution recommandé

```
S4701 (taxonomie + clés weight_bits/granularity/symmetry)
   │
   ▼
S4702 (extension émulateur : n_bits/ternaire + zero-point + harnais)
   │
   ├──► S4703 (sweep profondeur × granularité)  ── prioritaire
   ├──► S4704 (axe symétrie aux bits critiques)
   └──► S4705 (HDC/Maha contexte N/A)
                 │
                 ▼
         S4706 (figures + notebook)
                 │
                 ▼
         S4707 (tests + roadmap + triple_gap)
                 │
                 ▼
         S4708 (pointeur board → Sprint 48)
```

Tout le sprint est PC/émulateur — aucune carte requise. Le portage board est le **Sprint 48**.

## Sources de données (Sprint 47, lecture seule)

| Dataset | Loader / scénario CL | Rôle Sprint 47 |
| ------- | -------------------- | -------------- |
| Monitoring (D2) | `get_cl_dataloaders` — domain-incrémental, 3 tâches (Pump→Turbine→Compressor) | Colonnes EWC sweep profondeur/schéma |
| Pronostia (D4) | `get_pronostia_dataloaders` — domain-incrémental par condition, 3 tâches | Colonnes EWC sweep profondeur/schéma |

Configs de référence réutilisées : `configs/ewc_int8_{monitoring,pronostia}.yaml` (voie QAT/AUROC),
émulateur `int8_c_emulation.py` (chemin PTQ bit-exact déjà per-tensor/per-channel).

## Livrables

1. `docs/context/quantization_depth.md` — taxonomie profondeur/granularité/symétrie + mapping EWC-only (S4701).
2. `src/utils/int8_c_emulation.py` (extension) + `scripts/run_s47_quant_depth.py` — émulateur sub-INT8 + harnais (S4702).
3. `configs/quant_depth/ewc_{monitoring,pronostia}*.yaml` — configs portant `weight_bits`/`granularity`/`symmetry`.
4. `experiments/exp_S47_depth/`, `exp_S47_symmetry/`, `exp_S47_context/` — résultats JSON (AUROC, RAM théorique,
   proxy latence, delta vs fp32) par (bits, granularité, symétrie).
5. `src/figures/catalogs/quant_depth.py` → PNG `docs/figures/quantization_depth/` + notebook galerie.
6. `tests/test_s47_quant_depth.py` — tests structure/monotonie/honnêteté.
7. MAJ `docs/roadmap_phase2.md` + `docs/triple_gap.md` (§ Gap 3).

## Questions ouvertes

- `TODO(dorra)` : le gain RAM sub-INT8 exige un **kernel bit-packé** (2 poids/octet INT4, 4/octet INT2) ; sans
  packing, INT4 ≡ INT8 en RAM réelle. Sur PC la RAM est **théorique** ; le coût du dépacking + MAC FPU est
  tranché board (Sprint 48).
- `TODO(arnaud)` : métrique de référence EWC = **AUROC binaire** (voie S28), reprise de S4601 pour cohérence
  avec l'axe moment — confirmer.
- `TODO(dorra)` : le **zero-point affine** aide-t-il vraiment aux très bas bits (INT3/INT2), ou la per-channel
  suffit-elle ? (S4704 tranche côté PC ; Sprint 48 confirme board).

## Bilan (à compléter)

| Tâche | Statut | Temps réel | Notes |
|-------|:------:|:----------:|-------|
| S4700 | 📝 Doc | — | Overview + cadrage |
| S4701–S4708 | 📝 Doc | — | Documentés ; implémentation à venir (Bloc E → Sprint 48) |
