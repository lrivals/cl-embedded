# S4903 — Relance des expériences sous conditions définies (RAM totale, pile incluse)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 49 |
| **Priorité** | 🔴 Critique — produit les mesures RAM complètes que S4904 documente et S4905 trace. |
| **Statut** | ✅ Implémenté — board réelle NUCLEO-F439ZI : 16 board (14 mesurées + 2 N/A) + 16 PC (8 mesurées + 8 N/A) |
| **Durée estimée** | 5h |
| **Dépendances** | S4902 (orchestrateur) · accès carte NUCLEO-F439ZI (board réel) · loaders Monitoring/Pronostia/CMAPSS/CWRU ✅ |
| **Fichiers cibles** | `experiments/exp_S49_ram/` (JSON par cellule) |
| **Références** | CR §1 (conditions identiques) · §7 action 🔴 « relancer toutes les expériences » |

## Contexte

Le CR demande de **relancer toutes les expériences** avec la RAM complète et sous **conditions définies**
(mêmes groupes de données, même protocole MAJ+éval). Cette tâche exécute le sweep via S4902 et remplit
`exp_S49_ram/`.

## Spec

### 1. Grille

| Axe | Valeurs |
|-----|---------|
| Modèle | EWC, HDC, TinyOL, Mahalanobis |
| Dataset | Monitoring (D2), Pronostia (D4) — cœur ; CMAPSS/CWRU en complément selon applicabilité |
| Encodage | fp32, int8 |
| Plateforme | board (référence), pc (proxy tracemalloc) |
| Phases | idle / inference / update (S4901) |

**N/A honnête** : une cellule non applicable (ex. modèle sans mode int8, ou métrique mono-classe) porte
`status: "na"` + `na_reason`, jamais un chiffre fabriqué.

### 2. Conditions identiques (CR §1)

Mêmes groupes de données pour l'entraînement, **même protocole de mise à jour et d'évaluation** que la
comparaison de référence — consignés dans `conditions{}` de chaque JSON.

### 3. Sorties par cellule

JSON au schéma S4901 rempli avec valeurs **mesurées** : `data_bytes`, `bss_bytes`,
`stack_peak_inference_bytes`, `stack_peak_update_bytes`, `total_ram_bytes`, `stack_history`, `0 CRC`.

## Format de sortie

`experiments/exp_S49_ram/{model}_{dataset}_{condition}_{encoding}_{platform}.json` — remplis dès accès carte
(board) et immédiatement (PC). `pending` tant que non exécuté.

## Contraintes

- Board = mesure réelle DWT/`.bss`/pile, **0 CRC** attendu (cf. sprints précédents).
- Ne jamais additionner board + PC ; ne jamais inventer un `total_ram_bytes`.
- Historique du pic conservé (CR §2).

## Vérification

```bash
# après exécution réelle :
ls experiments/exp_S49_ram/*.json | wc -l                       # cellules produites
python -c "import json;d=json.load(open('experiments/exp_S49_ram/ewc_pronostia_5feat_fp32_board.json'));\
assert d['total_ram_bytes'] == d['data_bytes']+d['bss_bytes']+max(d['stack_peak_inference_bytes'],d['stack_peak_update_bytes'])"
grep -l '"status": "na"' experiments/exp_S49_ram/*.json          # N/A honnêtes tracés
```

## Implémentation ✅ (board réelle NUCLEO-F439ZI)

Sweep exécuté via `scripts/run_ram_full_sweep.py` (S4902). **32 cellules** dans
`experiments/exp_S49_ram/` (condition `5feat`, seed 42) :

- **PC** (16) : 8 `fp32` mesurées (tracemalloc forward) + 8 `int8` `na` (proxy PC = FP32 ; gain INT8
  analytique poids ÷4).
- **Board** (16) : **14 mesurées** (EWC/HDC/TinyOL/Maha × 2 datasets × fp32 + EWC/HDC/TinyOL × int8)
  + **2 `na`** (Maha × int8 = build dédié `-DMAHA_INT8`, hors périmètre RAM).

**Invariant `total = .data + .bss + max(pic_inf, pic_upd)` vérifié sur les 14 cellules board.**

Chiffres board mesurés (`.data = 460 B` partout) :

| Dataset | k | `.bss` | idle | inférence | update | total | % 256 Ko |
|---|---|---|---|---|---|---|---|
| Monitoring | 4 | 100 152 B | 4 236 B | 4 328–4 416 B | 4 328–4 712 B | 104 940–105 324 B | 40,0–40,2 % |
| Pronostia | 5 | 105 036 B | 4 244 B | 4 336–4 424 B | 4 336–4 720 B | 109 832–110 216 B | 41,9–42,0 % |

Constats mesurés :

- **Gap 2 ✅** : RAM totale **40–42 % de 256 Ko** sur toutes les cellules.
- **`idle ≠ 0`** (≈ 4,24 Ko) : trame `pipeline_run()` réservée dès le boot (thèse « trame
  partagée » confirmée mesurée).
- **Seul EWC creuse la pile en `update`** (4 688–4 720 B > inférence, SGD backward) ; HDC/Maha/TinyOL
  `inference == update` (trame partagée dominée par `hv[HDC_DIM]` = 4 Ko).
- **INT8 ≡ FP32 en `.bss`** ici : la conversion INT8 est faite **au runtime** depuis les poids FP32
  (les deux structs coexistent en `.bss`) → l'INT8 embarqué **ne réduit pas** le `.bss` (cohérent
  Sprints 29/36 ; le gain RAM INT8 réel exige le bit-packing, cf. Sprint 48).
- Board (référence matérielle) et PC (proxy tracemalloc) restés **séparés**, jamais additionnés.
