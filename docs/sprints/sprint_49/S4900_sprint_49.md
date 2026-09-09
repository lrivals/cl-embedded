# Sprint 49 — Mesure RAM complète (`.data + .bss + pic de pile`) + relance des expériences + fixes plots/terminologie

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 49 |
| **Semaine** | Indicative — à confirmer (post-Sprint 48) |
| **Statut** | ✅ S4901–S4907 implémentés (board réelle NUCLEO-F439ZI, 32 cellules `exp_S49_ram/` ; S4904–S4907 doc/figures/terminologie/tests lecture seule, sans carte) |
| **Priorité globale** | 🔴 Critique — répond aux **deux actions 🔴 « RAM »** du CR du 16 juillet 2026 : les mesures RAM rapportées jusqu'ici sont **incomplètes** (`.bss` seul, sans pic de pile), donc **sous-estimées**. Ce sprint généralise la formule `RAM totale = .data + .bss + pic de pile` à **toutes** les expériences, relance les mesures sous conditions définies, produit une doc structurée, et corrige plots + terminologie. |
| **Durée estimée totale** | ~22h (cadrage ~3h · instrumentation unifiée ~5h · relance expériences ~5h · doc structurée ~3h · fixes plots ~3h · terminologie ~1h · tests+docs ~2h) |
| **Dépendances** | **Firmware `profiling.c` ✅** (`profiling_stack_peak_bytes`/`profiling_total_ram_bytes` + sentinelle position-dépendante peinte au startup — le mécanisme de stack painting **existe déjà**) · `scripts/run_ram_board.py` ✅ · `scripts/ram_breakdown.py` ✅ · `docs/presentation_ram_measurement.md` ✅ · `src/evaluation/memory_profiler.py` ✅ (`tracemalloc` PC) · registre figures `src/figures/` ✅ (S4201) · accès carte NUCLEO-F439ZI (pour le `.bss`+pile réel board) |

## Contexte et motivation

Le CR du 16 juillet fixe la **méthode de mesure RAM complète** :

```
RAM totale = .data + .bss + pic de pile
```

- **`.data`** : globales initialisées.
- **`.bss`** : globales à zéro / non constantes.
- **Pic de pile** : maximum atteint par la call stack durant l'exécution.

> ⚠️ **Correction du CR** : « la mesure précédente n'incluait pas le pic de pile — les valeurs de RAM
> rapportées étaient **incomplètes**. À corriger dans toutes les expériences. »

**Point clé de cadrage** : le **mécanisme** de mesure existe déjà. Le firmware
[`profiling.c`](../../../firmware/stm32f4_blink/src/profiling.c) implémente le **stack painting**
(sentinelle position-dépendante peinte `[_ebss, _estack)` au boot, scan du high-water mark), expose
`profiling_stack_peak_bytes()` et `profiling_total_ram_bytes()` (= `.data + .bss + pic`), et
[`presentation_ram_measurement.md`](../../presentation_ram_measurement.md) documente la formule. Le problème
n'est **pas** l'absence d'outil : c'est que **les expériences ne remontent majoritairement que `.bss`**. Ce
sprint **généralise** l'usage de l'outil existant, **relance** les mesures, et **documente**.

Le CR demande aussi deux corrections transverses : (1) les **plots** portent des étiquettes d'étapes
« réseau de neurones » incohérentes avec les modèles réellement utilisés, et les **instants de mesure du pic
de pile** doivent être alignés sur les phases réelles (inférence / mise à jour CL) ; (2) remplacer partout
« RAM modulable » par « **mémoire non constante** ».

## Décisions de cadrage (utilisateur, CR du 16 juillet 2026)

- **Formule officielle** : `RAM totale = .data + .bss + pic de pile` — à appliquer à **toutes** les expériences.
- **Méthode pic de pile** : **stack painting** (motif connu type `0xDEADBEEF`, scan de la zone intacte) — déjà
  implémentée dans `profiling.c`, à **réutiliser telle quelle** (ne pas réécrire).
- **Conditions de relance identiques** à la comparaison de référence (CR §1) : **mêmes groupes de données pour
  l'entraînement, même protocole de mise à jour et d'évaluation** — base équitable.
- **Historique du pic de pile** : conserver l'**évolution du pic durant l'exécution** (intéressant pour le
  rapport, CR §2 conclusion), pas seulement la valeur finale.
- **Terminologie** : « RAM modulable » → « mémoire non constante » ; ne PAS toucher le CR lui-même (qui énonce
  l'instruction).
- **Aucun chiffre en dur** : les JSON de résultats portent `pending` tant qu'un run n'a pas tourné ; les
  figures rechargent depuis les JSON (garde AST 0-chiffre, comme S42/S44/S47).
- **Langue** : français.

## Nœud honnête : ce sprint généralise, il n'invente pas

Le stack painting et la formule `.data + .bss + pic` **existent déjà** (Sprint 20, `profiling.c`,
`presentation_ram_measurement.md`). Ce sprint ne prétend pas les créer : il **corrige une sous-estimation
systématique** (les expériences rapportaient `.bss` seul) en **branchant l'outil existant partout** et en
**re-mesurant**. Deux nuances à porter dans le texte et les figures :

1. **`.bss` seul ≠ RAM totale** : la pile vit **hors `.bss`** (zone libre au-dessus, cf.
   `presentation_ram_measurement.md`). Rapporter `.bss` seul **oublie la pile** — c'est la correction du CR.
2. **Le pic de pile dépend de la phase** : la mise à jour CL (SGD embarqué) creuse plus la pile que
   l'inférence seule. Les instants de mesure doivent donc coller aux **phases réelles des modèles** (inférence
   vs inférence+MAJ), pas à des étapes génériques de réseau de neurones — c'est le fix plots du CR.

## Tâches

### Bloc A — Cadrage & méthode

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4901 | **Cadrage RAM complète** : formaliser `RAM totale = .data + .bss + pic de pile`, recenser l'infra existante (`profiling_total_ram_bytes`, `run_ram_board.py`, `ram_breakdown.py`), fixer les **instants de mesure alignés sur les phases réelles** (inférence / mise à jour CL), spécifier le schéma JSON par cellule (`data_bytes`, `bss_bytes`, `stack_peak_bytes`, `total_ram_bytes`, `stack_history`) | 🔴 | `docs/sprints/sprint_49/S4901_cadrage_ram_complete.md`, `docs/context/ram_measurement.md` | 📝 Doc |

### Bloc B — Instrumentation & relance

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4902 | **Instrumentation unifiée** : source unique de la RAM totale par (modèle, dataset, condition, encodage) — board via `profiling_total_ram_bytes` (réutilise `run_ram_board.py`), PC via `tracemalloc` peak (`memory_profiler.py`) ; capture de l'**historique du pic de pile** (échantillonnage périodique du high-water mark durant le stream) | 🔴 | `scripts/run_ram_full_sweep.py` (réutilise `run_ram_board.py`/`ram_breakdown.py`/`memory_profiler.py`) | 📝 Doc |
| S4903 | **Relance des expériences sous conditions définies** (mêmes groupes de données / même protocole MAJ+éval, CR §1) : 4 modèles (EWC, HDC, TinyOL, Mahalanobis) × datasets × {fp32, int8} → RAM totale (pile incluse) ; N/A honnête si un modèle ne s'applique pas | 🔴 | `experiments/exp_S49_ram/` (JSON par cellule, `pending` avant run) | 📝 Doc |

### Bloc C — Documentation & figures

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4904 | **Doc structurée des consommations RAM par expérience** (export pour le rapport, CR action 🔴) : tableau `.data`/`.bss`/pic de pile/total par cellule + **historique d'évolution du pic** ; sections par modèle/dataset ; valeurs rechargées depuis `exp_S49_ram/` | 🔴 | `docs/context/ram_report.md`, `experiments/exp_S49_ram/summary.json` | ✅ |
| S4905 | **Fixes plots** (CR action 🟡) : retirer les étiquettes d'étapes « réseau de neurones » incohérentes, **adapter les instants de mesure du pic de pile aux phases réelles** (inférence / MAJ CL), tracer l'historique du pic ; corriger les figures de `presentation_ram_measurement.md` | 🟡 | `src/figures/catalogs/ram_full.py`, `docs/figures/ram_full/`, MAJ `docs/presentation_ram_measurement.md` | ✅ |
| S4906 | **Terminologie** (CR action 🟡) : remplacer « RAM modulable » → « **mémoire non constante** » — **4 occurrences réelles** dans `docs/presentation_ram_measurement.md` (lignes 221/227/234) et `scripts/ram_breakdown.py` (ligne 254 de la table) ; **ne pas modifier le CR** (lignes 65/172 = l'instruction elle-même) | 🟡 | `docs/presentation_ram_measurement.md`, `scripts/ram_breakdown.py` | ✅ |

### Bloc D — Clôture

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4907 | **Tests + docs + clôture** : `test_ram_full.py` (`total == data+bss+pile`, monotonie/positivité du pic, N/A honnête, garde AST 0-chiffre-en-dur sur `ram_full.py`) ; MAJ roadmap/triple_gap ; `graphify_sprint_update` | 🟡 | `tests/test_ram_full.py`, `docs/roadmap_phase2.md`, `docs/triple_gap.md` | ✅ |

## Ordre d'exécution recommandé

```
S4901 (cadrage : formule + instants de mesure + schéma JSON)
   │
   ▼
S4902 (instrumentation unifiée board+PC, historique du pic)
   │
   ▼
S4903 (relance expériences sous conditions définies → exp_S49_ram/)
   │
   ├──► S4904 (doc structurée ram_report.md)
   ├──► S4905 (fixes plots : étiquettes + instants pic)
   └──► S4906 (terminologie « mémoire non constante »)
                 │
                 ▼
         S4907 (tests + roadmap + triple_gap + graphify)
```

Le mécanisme (stack painting) est déjà en place ; le chemin critique est **S4902 → S4903** (brancher + relancer).
La partie board (`.bss`+pile réels) requiert la carte ; la partie PC (`tracemalloc`) tourne sans carte.

## Sources de données (Sprint 49, lecture seule)

| Dataset | Loader / scénario CL | Rôle Sprint 49 |
| ------- | -------------------- | -------------- |
| Monitoring (D2) | `get_cl_dataloaders` — domain-incrémental | RAM totale 4 modèles × {fp32,int8} |
| Pronostia (D4) | `get_pronostia_dataloaders` — domain/class-incrémental | RAM totale (dont mise à jour CL, pic de pile plus élevé) |
| CMAPSS (D5) / CWRU (D3) | loaders respectifs | Cellules complémentaires selon applicabilité modèle (N/A honnête) |

Conditions de relance = **identiques à la comparaison de référence du CR** (mêmes groupes de données,
même protocole MAJ+éval).

## Livrables

1. `docs/context/ram_measurement.md` — méthode RAM complète + instants de mesure (S4901).
2. `scripts/run_ram_full_sweep.py` — instrumentation unifiée board+PC + historique du pic (S4902).
3. `experiments/exp_S49_ram/` — JSON par cellule (`data_bytes`/`bss_bytes`/`stack_peak_bytes`/`total_ram_bytes`/`stack_history`, `pending` avant run) + `summary.json` (S4903–S4904).
4. `docs/context/ram_report.md` — doc structurée des consommations RAM par expérience (S4904).
5. `src/figures/catalogs/ram_full.py` → PNG `docs/figures/ram_full/` (étiquettes corrigées, instants alignés, historique du pic) + MAJ `presentation_ram_measurement.md` (S4905–S4906).
6. `tests/test_ram_full.py` + MAJ `roadmap_phase2.md`/`triple_gap.md` (S4907).
