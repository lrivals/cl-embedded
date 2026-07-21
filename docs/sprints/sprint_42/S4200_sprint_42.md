# Sprint 42 — Bibliothèque de figures de présentation + explication des stratégies de quantification

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 42 |
| **Semaine** | 13 – 19 juillet 2026 |
| **Statut** | ✅ Implémenté (7 juillet 2026) — infra + 17 figures + inventaire + notebook + tests |
| **Priorité globale** | 🟠 Haute — support transversal présentations/manuscrit (deadline manuscrit préliminaire, soutenances) |
| **Durée estimée totale** | ~30h (infra ~8h · inventaire ~3h · figures ~14h · notebook+tests+docs ~5h) |
| **Dépendances** | Sprint 22 ✅ (INT8 Python+C) · Sprint 28 ✅ (QAT PC) · Sprint 29 ✅ (board INT8) · Sprint 34 ✅ (Q15 Maha) · Sprint 36 ✅ (PTQ board F1) · Sprint 39 🟡 (émulateur + ablation ✅) · Sprint 40 📝 (kernel v2) |

## Contexte et motivation

Le projet a accumulé depuis le Sprint 22 **plusieurs stratégies de quantification distinctes**, avec des
résultats parfois opposés (le QAT PC préserve la métrique, la PTQ board legacy l'effondre, Q15 récupère).
Ces résultats sont **dispersés** dans ~8 sprints, 6 notebooks et des dizaines de JSON d'expériences. Il
n'existe aujourd'hui **aucune figure qui explique** :

1. **ce que chaque stratégie fait aux données** (mapping affine INT8, grille Q15, fake-quant vs PTQ,
   déquantification FPU) ;
2. **où** dans le pipeline PC→export→firmware chaque transformation s'applique ;
3. **quel impact mesuré** chacune a sur le fonctionnement du modèle (métrique, RAM, latence).

Chaque présentation ou chapitre de manuscrit qui aborde la quantification ré-improvise ces explications.
Ce sprint met en place une **bibliothèque de génération de figures réutilisable** (style commun, chargement
des `experiments/exp_*`, catalogue régénérable) dont le **premier cas d'usage** est le catalogue complet
« stratégies de quantification ». L'infrastructure est volontairement **générale** : elle servira aux
figures futures de présentations/rapports **non liées à un sprint spécifique** (timeline, gaps, scénarios).

## Décisions de cadrage (utilisateur, 6 juillet 2026)

- **Périmètre quantification** : **toutes les stratégies** — FP32 référence, INT8 affine (QAT fake-quant PC
  Sprint 28 **vs** PTQ embarquée legacy Sprints 29/36 **vs** kernel v2 calibré per-tensor/per-channel
  Sprints 39/40), **Q15** (Mahalanobis Sprint 34, EWC v2 Sprint 39), **HDC int16-AM** (Sprints 22/29).
- **Trois familles de figures** : pédagogiques/conceptuelles + pipeline/flux de données + impact mesuré.
- **Infrastructure générique** : module figures réutilisable, la quantification est le premier catalogue.
- **Langue** : **français** (textes et labels), cible présentations + manuscrit.

## Les stratégies à couvrir (matériau existant)

| Stratégie | Transformation des données | Où | Impact connu (source) |
|-----------|---------------------------|-----|----------------------|
| **FP32** (référence) | aucune — float32 partout (poids, activations, MAJ CL) | PC + board (FPU Cortex-M4) | baseline métrique/latence ; RAM ×1 |
| **INT8 QAT fake-quant (PC)** | quantize→dequantize simulé pendant l'entraînement ; le gradient voit l'erreur de quantification | PC (`src/models/ewc/ewc_mlp_int8.py`, Sprint 28) | métrique **préservée** (Δ≤0.006 EWC) voire régularisante (TinyOL) — `exp_S28_PC_*` |
| **INT8 PTQ legacy (board)** | conversion one-shot post-entraînement, échelle **fixe 1/128**, accumulateur int16 | firmware (`ewc_head_int8.c`, `ewc_int8_from_fp32`, Sprints 29/36) | RAM ÷4 ✅ mais **effondrement F1** (0.07–0.15 vs ≈0.92) — `exp_S36_board_*_int8_*` |
| **INT8 v2 calibré (per-tensor / per-channel)** | scale calibré sur les données, accumulateur int32, déquant FPU | émulateur PC (`src/utils/int8_c_emulation.py`, Sprint 39) + firmware v2 (`ewc_head_int8_v2.c`, Sprint 40) | **récupération** F1 (+0.88 legacy→per-tensor ; per-channel ≈ FP32) — `exp_S39_ablation/`, `exp_S39_quant_sweep/` |
| **Q15 (int16)** | `sigma_inv_` (Maha) / poids (EWC v2) en int16 Q15, scale par-tenseur `max·/32767` ; `mu_` reste INT8 ; déquant→FP32 FPU | PC + firmware (`mahalanobis_q15.c`, Sprint 34) | RAM ÷2, **AUROC recouvrée** (Pronostia −0.113→+0.013), parité board exacte — `exp_S34_maha_q15/` |
| **HDC int16-AM** | mémoire associative en int16 (hypervecteurs bipolaires) | PC + firmware (`hdc.c`, Sprints 22/29) | RAM ×2.33, métrique Δ=0 — `exp_S28_PC_ewc_hdc/`, `exp_S29_board_int8/` |

**Messages transversaux à faire porter par les figures** (déjà établis, jamais illustrés ensemble) :
- *quantifier ≠ quantifier* : le **moment** (pendant l'entraînement vs après) et la **calibration de
  l'échelle** dominent le choix du format (QAT ✅ / PTQ figée ❌ / PTQ calibrée ✅) ;
- la **dynamique des tenseurs** décide du format : `sigma_inv_` grande dynamique → INT8 écrase, Q15 tient ;
- **paradoxe latence** : sur Cortex-M4 FPU, la déquant dans la boucle rend l'INT8 *plus lent* que FP32 —
  le gain est **RAM uniquement** (chemin entier SIMD/CMSIS-NN = travaux futurs, `TODO(dorra)`).

## Règles d'honnêteté (héritées, non négociables)

- **Aucun chiffre en dur** dans le code des figures : toute valeur affichée est **chargée depuis un JSON**
  de `experiments/` (ou recalculée depuis un checkpoint/émulateur). Un test l'impose (S4207).
- Les cellules sans mesure réelle (ex. board kernel v2 si le flash S40 n'a pas eu lieu) affichent
  **`« à mesurer »`** ou sont masquées en gris — jamais extrapolées.
- Les figures d'impact distinguent explicitement **« mesuré board »** vs **« émulé PC »** vs « PC natif ».
- Figures pédagogiques : les distributions/poids montrés proviennent de **vrais checkpoints/datasets** du
  projet (pas de gaussiennes synthétiques présentées comme des données réelles ; le synthétique est permis
  pour illustrer un *mécanisme* s'il est étiqueté comme tel).

## Tâches

### Bloc A — Infrastructure réutilisable

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4201 | Module `src/figures/` : style commun (`style.py`), loaders `experiments/` (`loaders.py`), **registre de catalogues** (`registry.py`) + CLI `scripts/generate_figures.py --catalog <nom>` régénérant `docs/figures/<catalog>/` | 🔴 | `src/figures/`, `scripts/generate_figures.py` | ✅ Implémenté |

### Bloc B — Contenu quantification

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4202 | **Inventaire de référence** des stratégies : doc unique (taxonomie, formules, code source, expériences, résultats-pointeurs) — source de vérité textuelle des slides/manuscrit | 🔴 | `docs/context/quantization_strategies.md` | ✅ Implémenté |
| S4203 | **Figures pédagogiques** : mapping affine INT8, grille Q15 vs INT8, fake-quant (STE) vs PTQ, erreur de quantification sur vrais poids, cas `sigma_inv_` grande dynamique | 🔴 | `src/figures/catalogs/quant_pedagogy.py` → `docs/figures/quantization/pedagogy/` | ✅ Implémenté |
| S4204 | **Figures pipeline/flux** : où chaque quantification s'applique (entraînement PC → export → firmware → déquant FPU), un schéma par stratégie + un schéma comparatif | 🟠 | `src/figures/catalogs/quant_pipeline.py` → `docs/figures/quantization/pipeline/` | 📝 Doc |
| S4205 | **Figures d'impact mesuré** depuis les JSON existants : métrique préservée/dégradée par stratégie, ablation S39, récupération Q15 S34, RAM Gap 3, paradoxe latence | 🔴 | `src/figures/catalogs/quant_impact.py` → `docs/figures/quantization/impact/` | 📝 Doc |

### Bloc C — Assemblage & clôture

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4206 | Notebook catalogue (galerie commentée : chaque figure + son explication FR prête à copier en slide/manuscrit), nbconvert OK | 🟠 | `notebooks/cl_eval/quantization_figures/catalog.ipynb` | 📝 Doc |
| S4207 | Tests Python (registre, 0 chiffre en dur, régénération idempotente, honnêteté « à mesurer ») + roadmap + `CLAUDE.md` + `graphify_sprint_update` | 🟡 | `tests/test_figures_library.py`, `docs/roadmap_phase2.md` | 📝 Doc |
| S4208 | **Extension présentation EWC-only** (post-sprint) : doc narratif 3-axes FP32/INT8, **catalogue séparé `quantization_ewc`** (7 PNG EWC, sans Q15, isole I4 RAM et retire les résidus Q15) + **complétion mesures board v2 monitoring** (frozen+online) qui débloque la cellule « à mesurer » | 🟠 | `docs/context/quantization_presentation.md`, `src/figures/catalogs/quant_ewc.py` → `docs/figures/quantization_ewc/` | ✅ Implémenté |

## Ordre d'exécution recommandé

```
S4201 (infra src/figures/) ──┬→ S4203 (pédagogie) ──┐
S4202 (inventaire doc) ──────┤→ S4204 (pipeline) ───┼→ S4206 (notebook catalogue) → S4207 (tests + clôture)
                             └→ S4205 (impact JSON) ─┘
```

S4202 peut démarrer en parallèle de S4201 (pur doc). S4203–S4205 sont indépendantes entre elles.

## Sources de données (expériences existantes, lecture seule)

| Exp / artefact | Contenu utilisé | Figures |
|----------------|-----------------|:-------:|
| `exp_S28_PC_ewc_hdc/`, `exp_S28_PC_tinyol_maha/` | QAT PC 4×5 : métrique + RAM | S4205 |
| `exp_S29_board_int8/` | 20 cellules board INT8 : RAM ×2.70–4.00, latence | S4205 |
| `exp_S34_maha_q15/` | corrélation de rang + AUROC INT8 vs Q15 | S4203, S4205 |
| `exp_S36_board_{frozen,online}_int8_*` | effondrement F1 PTQ legacy, accord INT8↔FP32 | S4205 |
| `exp_S39_ablation/`, `exp_S39_quant_sweep/` | attribution de la perte F1 par facteur, balayage schémas | S4205 |
| `exp_S40_board_v2/` (si disponible) | récupération board réelle kernel v2 | S4205 (« à mesurer » sinon) |
| checkpoints EWC/Maha + `src/utils/int8_c_emulation.py` | vrais poids/tenseurs pour les figures pédagogiques | S4203 |

## Livrables

1. `docs/sprints/sprint_42/` (ce dossier) — specs S4200–S4207.
2. `src/figures/` (style, loaders, registre) + `scripts/generate_figures.py` — infrastructure pérenne.
3. `docs/context/quantization_strategies.md` — référence textuelle unique.
4. `docs/figures/quantization/{pedagogy,pipeline,impact}/*.png` — figures FR régénérables.
5. `notebooks/cl_eval/quantization_figures/catalog.ipynb` — galerie commentée.
6. `tests/test_figures_library.py` + roadmap + statut `CLAUDE.md`.

## Extensibilité prévue (hors périmètre de ce sprint, mais guidant l'API)

- Nouveaux catalogues futurs enregistrables sans toucher à l'infra : `timeline`, `triple_gap`,
  `scenarios_cl`, figures de soutenance — un fichier `src/figures/catalogs/<nom>.py` + une entrée registre.
- Option future `--lang en` (S4201 prévoit labels centralisés dans chaque catalogue pour rendre la
  traduction mécanique) — **non implémentée ici** (décision utilisateur : FR).
- Réutilisation des helpers existants `src/evaluation/plots.py` quand pertinent — ne pas dupliquer.

## Questions ouvertes

- `TODO(arnaud)` : les figures pédagogiques (mapping affine, STE) doivent-elles citer les notations du
  manuscrit (chapitres S41) pour cohérence de symboles (s, z, x̂) ?
- `TODO(dorra)` : validation des schémas pipeline S4204 (exactitude du chemin déquant FPU et du point
  d'application des scales per-channel dans le kernel v2) ?

## Bilan (à compléter)

| Tâche | Statut | Temps réel | Notes |
|-------|:------:|:----------:|-------|
| S4200 | ✅ | — | Overview + cadrage |
| S4201 | ✅ | ~8h | Infra `src/figures/` (style, loaders, registre) + CLI (déjà livrée) |
| S4202 | ✅ | ~3h | `docs/context/quantization_strategies.md` (déjà livré) |
| S4203 | ✅ | ~5h | Catalogue `quantization/pedagogy` P1–P6 (déjà livré) |
| S4204 | ✅ | ~3h | Catalogue `quantization/pipeline` F1–F5 + `schematic.py` partagé |
| S4205 | ✅ | ~4h | Catalogue `quantization/impact` I1–I6, 0 chiffre en dur (garde AST) |
| S4206 | ✅ | ~2h | `catalog.ipynb` 49 cellules, nbconvert OK, valeurs chargées |
| S4207 | ✅ | ~2h | `test_figures_library.py` 7/7 PASS, 714 collectés 0 erreur, docs+graphify |
| S4208 | ✅ | ~3h | Doc narratif + catalogue `quantization_ewc` (7 PNG) + 2 cellules board v2 monitoring mesurées (voir Extension) |

**Livrables réels** : bibliothèque `src/figures/` (style/loaders/registre/`schematic.py` + 3 catalogues) · CLI `scripts/generate_figures.py` · **17 PNG** sous `docs/figures/quantization/{pedagogy,pipeline,impact}/` · inventaire `docs/context/quantization_strategies.md` · notebook-galerie `notebooks/cl_eval/quantization_figures/catalog.ipynb` · `tests/test_figures_library.py` (7 PASS). Règles d'honnêteté respectées (badges plateforme, « à mesurer », métriques nommées, 0 littéral de résultat). `TODO(arnaud)` (notations manuscrit) et `TODO(dorra)` (scales per-channel kernel v2) laissés ouverts.

## Extension S4208 (16 juillet 2026) — variante présentation EWC-only + complétion board v2

**Motivation.** Les figures d'impact du sprint (S4205) juxtaposent plusieurs modèles (EWC, HDC,
TinyOL, Maha) et plusieurs stratégies (dont Q15) dans un même graphe. Pour le fil de présentation
FP32/INT8 centré **tête EWC**, il fallait des figures **isolant EWC** et **retirant Q15**, sans
écraser le jeu d'origine.

**Décisions utilisateur.** (1) Version narrative de l'inventaire en **3 temps** — `fp32` référence /
pourquoi comparer `int8_qat` (QAT PC) et `int8_ptq_legacy` (PTQ board figée) **n'est pas pertinent**
(erreur de catégorie : moment/lieu/calibration/mesure diffèrent, la perte legacy vient du **scale
figé** pas du post-training, ablation S39) / `int8_v2` comme meilleure approche ; **Q15 écarté du fil**.
(2) **Dossier de figures séparé** pour ne pas modifier les `quantization/*` existants.
(3) Périmètre mesures board : **per_channel / monitoring / {frozen, online}** uniquement (Q15 et
int8_legacy hors périmètre EWC).

**Livrables.**
- `docs/context/quantization_presentation.md` — doc narratif 3-axes (aucun chiffre inventé, repris de
  l'inventaire S4202 ; renvoie Q15/int16_am vers `quantization_strategies.md`).
- `src/figures/catalogs/quant_ewc.py` (catalogue `quantization_ewc`, enregistré dans
  `catalogs/__init__.py`) → **7 PNG** sous `docs/figures/quantization_ewc/` : réutilise les loaders de
  `quant_impact`/`quant_pedagogy` (**0 chiffre en dur**). **I4 isolée EWC** (`ram_gap3_ewc` : ratio ×4
  par dataset, plus de HDC/TinyOL/Maha) ; **résidus Q15 retirés** de `metrique_par_strategie_ewc` (I1),
  `ablation_perte_f1_ewc` (I2), `erreur_quantification_poids_ewc` (P4) et couleur Q15 de
  `qat_vs_ptq_resultats_ewc` (I6) ; `paradoxe_latence_ewc` (I5) et `mapping_affine_int8_ewc` (P1)
  réexportés tels quels. Génération : `python scripts/generate_figures.py --catalog quantization_ewc`.
- **Mesures board réelles NUCLEO-F439ZI** (kernel INT8 v2 per-channel, tête EWC × monitoring, via
  `run_s40_board_v2.py`), qui comblent la seule cellule « à mesurer » des figures EWC
  (`_board_v2_f1('monitoring','frozen')`) :

| Cellule | F1_faulty | Parité board↔PC | Latence P50 | Gap 2 | CRC | RAM | Fichier |
|---|---|---|---|---|---|---|---|
| frozen | **0,9173** | **1,000** (`exact_vs_emulator`) | 65 µs | ✅ | 0 | ×4 | `exp_S40_board_v2/results_per_channel_monitoring_frozen.json` |
| online | **0,9016** | 0,989 (`approx`, float32 board vs float64 PC) | 577 µs (inf 65 + MAJ 512) | ✅ | 0 | ×4 | `…_monitoring_online.json` |

`.bss=101 236 B`, n_streamé ≈ 7 670. Après mesure, `qat_vs_ptq_resultats_ewc.png` affiche le F1 board
réel (0,92) au lieu de « à mesurer ».

**Contrôles.** `test_figures_library.py` **7/7 PASS** · dossier `docs/figures/quantization/` d'origine
**intact** (aucun écrasement) · aucune modification de code figures/firmware (le pipeline consomme la
valeur dès que le JSON board existe). **Reste** (optionnel) : rafraîchir la figure d'origine
`quantization/impact/qat_vs_ptq_resultats.png` (lit aussi cette cellule) ; actualiser la note « online
v2 à mesurer » du doc de présentation (désormais mesuré).
