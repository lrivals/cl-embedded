# Sprint 40 — Rédaction d'un article standalone : EWC PC↔Board & INT8 vs FP32 (Pronostia + Monitoring)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 40 |
| **Semaine** | 5 – 11 juillet 2026 |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir (Bloc A/B board différés si carte indisponible) |
| **Priorité globale** | 🔴 Critique — livrable manuscrit/publication (deadline manuscrit préliminaire 15 avril, contribution triple gap) |
| **Durée estimée totale** | ~45h (Bloc A ~12h · Bloc B ~10h board · Bloc B/C notebook+rédaction ~23h) |
| **Dépendances** | Sprint 36 ✅ (PC↔board EWC apparié) · Sprint 39 🟡 (émulateur + ablation ✅ ; kernel v2 + board différés) · carte NUCLEO-F439ZI + `references.bib` projet |

## Contexte et motivation

Deux campagnes récentes et **complémentaires** sur le modèle **EWC** (M2), datasets **Pronostia**
(class-incremental) et **Monitoring** (domain-incremental), forment le matériau d'un article :

- **Sprint 36** — comparaison **appariée PC ↔ NUCLEO-F439ZI** (frozen/online × conditions `5feat`/`all`).
  Résultats mesurés carte réelle : **parité FP32 exacte** (frozen 1.000, online 0.96–0.99), **Gap 2**
  respecté (inférence 48–65 µs, inférence+MAJ 239–340 µs ≪ 100 ms), **Δacc_final PC↔board ≤ 0.007**. Sur la
  même carte, l'axe **INT8 vs FP32** montre un **effondrement F1** (0.07–0.15 vs FP32 ≈0.92 ; accord
  INT8↔FP32 0.59–0.74) — la PTQ embarquée « legacy » ne préserve pas la métrique.
- **Sprint 39** — **diagnostic par émulateur C bit-exact** (`src/utils/int8_c_emulation.py`) : la chute
  n'est PAS imputable à INT8 en soi mais au **kernel legacy** (scale figé `1/128`, accumulateur int16). Le
  facteur dominant est le **scale calibré** : l'échelle d'ablation `legacy_c → per_tensor_calib` gagne
  **+0.88 F1** (Pronostia 0.066→0.946, Monitoring 0.118→0.920), **per-channel** ≈ FP32 (Δ≤0.02) et **Q15** =
  FP32 exact. Cette récupération est aujourd'hui validée **au PC uniquement** (board différée, S3915).

**Besoin.** Ces résultats sont dispersés (2 sprints, 2 notebooks, JSON séparés) et l'axe « récupération
INT8 » manque de validation matérielle. Pour un article rigoureux à **résultats comparables et cohérents
(mêmes conditions d'exécution)**, ce sprint : (a) **complète le kernel v2 + la validation board différée**
du Sprint 39 pour obtenir la récupération INT8 **réelle sur carte** ; (b) **unifie** les données dans un
notebook de synthèse ; (c) rédige un **article standalone LaTeX en deux versions (FR + EN)**.

## Nœud honnête

- La comparaison board **INT8 vs FP32 à conditions strictement identiques** n'existe aujourd'hui que pour le
  **kernel legacy** (Sprint 36 → effondrement F1). La **récupération** (per-channel/Q15 ≈ FP32) est pour
  l'instant **émulateur PC** (Sprint 39). Le **Bloc B** de ce sprint lève ce point en flashant le kernel v2.
- **Règle « aucun chiffre inventé »** : tant que la carte n'a pas streamé le v2, les cellules board
  correspondantes portent `"à mesurer"` dans le notebook et l'article ; `experiments/exp_S40_board_v2/`
  reste absent jusqu'à l'exécution réelle. L'article distingue explicitement **« mesuré board »** vs
  **« émulé PC »**.
- **Paradoxe latence INT8** assumé : le kernel déquantifie vers FP32 dans la boucle → pas d'accélération sur
  Cortex-M4 FPU (RAM ÷4 sans gain latence). Le vrai chemin entier (SIMD/CMSIS-NN) reste différé
  (`S3910/S3917`, `TODO(dorra)`) et est présenté comme **problème ouvert / travaux futurs**.

## Message scientifique de l'article (fil directeur)

1. **Portabilité EWC sur MCU** : parité FP32 board↔PC **exacte** (frozen) / quasi-exacte (online), sous
   budget **Gap 2** (<100 ms) et **Gap 3** RAM (<256 Ko) — mesuré sur carte réelle.
2. **Piège de la PTQ naïve** : quantification INT8 « legacy » (scale figé) → **effondrement F1** malgré RAM
   ÷4 ⇒ le gain mémoire ne suffit pas, la calibration est décisive.
3. **Récupération par kernel calibré** : per-channel / Q15 restaurent la F1 (Δ≤0.02) → validé **émulateur
   bit-exact PC ET carte réelle** (apport du sprint).
4. **Honnêteté** : distinction « mesuré board » vs « émulé PC » ; paradoxe latence INT8 (pas d'accélération
   FPU sans SIMD).

## Tâches

### Bloc A — Prérequis données (compléter Sprint 39 différé)

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4001 | Kernel C **v2** (int32 + scale calibré per-canal + option Q15) + export `--int8-v2` + tests Unity host (parité C↔émulateur, `test_v1_unchanged`) | 🔴 | `firmware/stm32f4_blink/src/ewc_head_int8_v2.c`, `inc/ewc_head_int8_v2.h`, `scripts/export_weights_c.py`, `firmware/.../tests/test_ewc_int8_v2.c` | 📝 Doc |
| S4002 | **Board** : flasher v2, mesurer latence/`.bss`/F1/accord INT8↔FP32/parité board↔PC (per-channel, q15 × pronostia, monitoring × frozen, online) ; A/B v1 vs v2 | 🔴 | `scripts/run_s40_board_v2.py` → `experiments/exp_S40_board_v2/` | 📝 Doc (board différée) |

### Bloc B — Synthèse unifiée

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4003 | Notebook de synthèse unifié (recharge exp_S36 + exp_S39 + exp_S40) → figures article + PNG haute-déf | 🔴 | `notebooks/cl_eval/article_ewc_int8/synthesis.ipynb` → `docs/figures/sprint40_article/` | 📝 Doc |

### Bloc C — Rédaction article (standalone, FR + EN)

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4004 | Squelette LaTeX partagé + `references.bib` + `Makefile` article (`make fr` / `make en`) | 🔴 | `docs/article/ewc_int8_mcu/` | 📝 Doc |
| S4005 | Rédaction **version française** complète (texte, tables, figures) | 🔴 | `docs/article/ewc_int8_mcu/main_fr.tex` | 📝 Doc |
| S4006 | Rédaction **version anglaise** (miroir strict FR, cohérence numérique FR≡EN) | 🔴 | `docs/article/ewc_int8_mcu/main_en.tex` | 📝 Doc |

### Bloc D — Clôture

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4007 | Tests Python (figures↔JSON, FR≡EN) + build LaTeX + roadmap/`triple_gap.md`/`CLAUDE.md` + `graphify_sprint_update` | 🟡 | `tests/test_sprint40_article.py`, `docs/roadmap_phase2.md`, `docs/triple_gap.md` | 📝 Doc |

## Ordre d'exécution recommandé

```
(prérequis Sprint 39) S3907–S3909 ─┐
S4001 (kernel v2 + tests host) ─────┼→ S4002 (board réelle, quand carte dispo)
                                    │
exp_S36 ✅ · exp_S39 ✅ ────────────┴→ S4003 (notebook synthèse) → S4004 → S4005 → S4006 → S4007
```

> **Chemin sans carte** : S4001 (firmware + émulateur, PC-only), S4003–S4006 restent faisables sur données
> émulateur + exp_S36/exp_S39 ; les cellules board de l'article portent `"à mesurer"` jusqu'au flash réel.

## Nomenclature des expériences

| Exp ID | Contenu | Plateforme |
|--------|---------|:----------:|
| `exp_S36_*` (existant) | PC↔board EWC apparié (FP32 + INT8 legacy), frozen/online, 5feat/all | PC + Board |
| `exp_S39_ablation/`, `exp_S39_quant_sweep/` (existant) | Ablation F1 + balayage schémas | PC (émulateur) |
| `exp_S40_board_v2/` | Kernel v2 board : latence, `.bss`, F1, accord INT8↔FP32, parité | Board (différé) |

## Livrables

1. `docs/sprints/sprint_40/` (ce dossier) — specs S4000–S4007.
2. `firmware/.../ewc_head_int8_v2.c/.h` + export `--int8-v2` + tests Unity host (Bloc A).
3. `experiments/exp_S40_board_v2/` — mesures board v2 (quand carte disponible).
4. `notebooks/cl_eval/article_ewc_int8/synthesis.ipynb` + `docs/figures/sprint40_article/*.png`.
5. **`docs/article/ewc_int8_mcu/`** — article standalone LaTeX **FR + EN** compilable.
6. `tests/test_sprint40_article.py` + entrée roadmap + section Gap 3 `triple_gap.md`.

## Questions ouvertes

- `TODO(arnaud)` : cible de soumission de l'article (workshop TinyML, revue, annexe manuscrit) ? Périmètre :
  EWC seul, ou étendre aux 4 modèles du balayage S3906 ?
- `TODO(dorra)` : toolchain CMSIS-NN pour trancher le paradoxe latence INT8 avant soumission (S3910/S3917) ?
- `FIXME(gap3)` : si le board v2 confirme la récupération F1 (per-channel/Q15 ≈ FP32), reformuler le
  Gap 3 « partiel » → contribution : **RAM ÷4 SANS perte de métrique** (kernel calibré) sur MCU réel.

## Bilan (à compléter)

| Tâche | Statut | Temps réel | Notes |
|-------|:------:|:----------:|-------|
| S4000 | 📝 Doc | — | Overview + cadrage article |
| S4001–S4007 | 📝 Doc | — | Documentés ; implémentation à venir (Bloc B board différée si carte indisponible) |
