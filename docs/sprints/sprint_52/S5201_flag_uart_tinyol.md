# S5201 — Flag UART TinyOL, poids à dimension quelconque, re-mesure carte

| Champ | Valeur |
|-------|--------|
| **Sprint** | 52 |
| **Origine** | Action **2.6** de `docs/context/actions_restantes_resultats.txt` (priorité 1) |
| **Statut** | 🟡 en cours — code ✅, tests ✅, re-mesure carte en cours |
| **Carte** | NUCLEO-F439ZI réelle, `/dev/ttyACM0` |

## Le défaut

`scripts/sensor_stream.py` ne comportait aucune branche pour `--model tinyol` : `model_flags`
restait à `0`, valeur qui correspond au chemin d'inférence **par défaut** du firmware
(`pipeline.c`, branche `else`) — le détecteur de **Mahalanobis**. Le commentaire qui
justifiait cette absence (« tinyol et mahalanobis n'ont pas de flag dédié ») était faux :
`pipeline.c` sélectionne bien la route TinyOL sur `PROTO_FLAG_TINYOL_MODE` (0x80).

Conséquence mesurable : les JSON `exp_S35_board_{5feat,all}_tinyol_*` étaient numériquement
identiques à leurs homologues `mahalanobis` sur les 10 paires (accuracy, F1, latence p50/p99,
`.bss`, jusqu'au champ `date`). Le manuscrit publiait cette duplication au tableau 5.1.

**Le défaut ne se limitait pas au flag.** Trois verrous supplémentaires auraient fait tourner
la route TinyOL à **poids nuls** même avec le bon flag :

1. `pipeline.c` et `tinyol.c` gardaient la copie des poids par `TINYOL_IN == WEIGHTS_NATIVE_DIM`,
   avec `WEIGHTS_NATIVE_DIM` **figé à 5** → toute condition `k ≠ 5` (Monitoring 5feat = 4
   variables, `all`, `best`) laissait l'auto-encodeur à zéro en `.bss` ;
2. aucun driver n'exportait de poids TinyOL (`export_weights_c.py --mahal --ewc-head` seulement) ;
3. `export_weights_tinyol.py` était câblé en dur à 5 entrées.

Un quatrième point, latent : la route TinyOL déclarait `float recon[EWC_IN]` et appelait
`tinyol_reconstruction_error(..., EWC_IN)` alors que `tinyol_decode` écrit `TINYOL_OUT`
flottants — dimensions distinctes par modèle en condition `best`/`all`, donc débordement de
pile possible.

## Le correctif

| Fichier | Changement |
|---------|------------|
| `scripts/sensor_stream.py` | branche `elif args.model == "tinyol": model_flags = FRAME_FLAGS_TINYOL_MODE` ; commentaire remplacé par une garde explicite (« Mahalanobis est le seul modèle à `flags=0` ») |
| `firmware/.../src/pipeline.c` | garde `TINYOL_IN == TINYOL_NATIVE_DIM` (repli `WEIGHTS_NATIVE_DIM`, pattern S3507) ; `float recon[TINYOL_OUT]` ; MSE sur `TINYOL_OUT` |
| `firmware/.../src/tinyol.c` | même garde dans `tinyol_init` |
| `scripts/export_weights_tinyol.py` | `TinyOLBoard(dim=k)` ; section C dimensionnée sur le modèle + `#define TINYOL_NATIVE_DIM k` ; entraînement `--dataset/--condition` via `load_condition_arrays` ; `--dim` (vérification) ; `--emit-test-reference` |
| `scripts/train_board_reference.py` | `--model tinyol` → `checkpoints/tinyol_board.pt` + sidecar `.threshold.json` |
| `scripts/run_feature_condition_board.py` | entraînement + export TinyOL par cellule ; TinyOL promu dans `PARITY_MODELS` ; `_pc_pred_tinyol` |
| `scripts/run_board_threshold_sweep.py` | idem à `k = 5` |

Le seuil `TINYOL_THRESHOLD` reste calibré P95 × 1,5 sur les MSE d'entraînement, et voyage avec
le checkpoint (sidecar JSON) pour que la réplique PC applique **exactement** le seuil embarqué.

### Parité par construction

L'entraînement de référence consomme `load_condition_arrays(dataset, condition, "tinyol")` —
la même source de vérité que `sensor_stream.py --condition` (S3508). Carte et PC voient donc
les mêmes colonnes, et la parité est vérifiable, non postulée.

## Tests

- `tests/test_tinyol_board_flag.py` — **13 PASS**. Contient la garde anti-régression du bug :
  *tout modèle de `--model` sans branche de flag exécute silencieusement Mahalanobis ; un seul
  est admis à l'être*. Vérifie aussi que la section générée suit la dim k, qu'un ré-export ne
  laisse pas deux `TINYOL_NATIVE_DIM`, et que les gardes firmware n'utilisent plus la constante figée.
- `firmware/stm32f4_blink` : `make test` → **141 tests, 0 échec**.
  Les **2 échecs TinyOL préexistants** (dette 4.2 de l'audit) disparaissent. Cause identifiée :
  (a) `model_weights.h` était désynchronisé de son propre golden `REF_EMB_T0_SEED42`, recopié à
  la main dans `test_tinyol.c` et jamais régénéré ; (b) `test_tinyol_predict_normal_zero_weights`
  utilisait `TINYOL_THRESHOLD` — un seuil *calibré* valant ~1e-5 — comme oracle d'un cas à poids
  nuls dont la MSE vaut 0,0088 : le test échouait dès que l'export calibrait le seuil.
  Le golden vit désormais dans `tests/tinyol_reference.h`, **généré avec les poids**.
- **Couplage rendu visible, puis fermé.** Après la campagne S32, `make test` est retombé en
  échec sur `test_tinyol_forward_delta` : le header portait les poids de la dernière cellule
  exportée alors que le golden datait encore du seed 42. Le test faisait exactement son
  travail. Les deux drivers passent désormais `--emit-test-reference` à chaque export, de
  sorte que le couple (poids, golden) reste cohérent dans l'arbre de travail.
- `.bss` par défaut **invariant à 105 036 B** ; builds `TINYOL_IN=4` et `TINYOL_IN=5` avec poids
  d'une autre dim : compilent tous deux (poids simplement non copiés).

## Mesures carte

Campagne `run_feature_condition_board.py --port /dev/ttyACM0 --skip-existing` (checkpoints
Maha/EWC réutilisés ⇒ non-régression vérifiable), puis `run_board_threshold_sweep.py`.

### Grille S35 — 15 cellules TinyOL sur carte réelle (✅ terminée)

| condition | jeu | k | lat. p50 | F1 | accuracy | parité | `.bss` |
|-----------|-----|---|---------:|----:|---------:|-------:|-------:|
| 5feat | monitoring | 4 | 81 µs | 0,9268 | 0,9800 | 1,000 | 100 152 |
| 5feat | cwru | 5 | 85 µs | 1,0000 | 1,0000 | 1,000 | 105 036 |
| 5feat | pronostia | 5 | 85 µs | 0,8889 | 0,9800 | 1,000 | 105 036 |
| 5feat | cmapss | 5 | 85 µs | 0,3000 | 0,9067 | 1,000 | 105 036 |
| 5feat | paderborn | 5 | 85 µs | 0,0000 | 0,3333 | 1,000 | 105 036 |
| all | monitoring | 4 | 81 µs | 0,9268 | 0,9800 | 1,000 | 100 152 |
| all | cwru | 9 | 100 µs | 1,0000 | 1,0000 | 1,000 | 124 680 |
| all | pronostia | 13 | 114 µs | 0,6667 | 0,9333 | 1,000 | 144 516 |
| all | cmapss | 21 | 148 µs | 0,1000 | 0,8800 | 1,000 | 184 864 |
| all | paderborn | 7 | 92 µs | 0,0000 | 0,3333 | 1,000 | 114 832 |
| best | monitoring | 2 | 74 µs | 0,8718 | 0,9667 | 1,000 | 99 568 |
| best | cwru | 7 | 92 µs | 0,9963 | 0,9933 | 1,000 | 99 488 |
| best | pronostia | 5 | 85 µs | 0,3571 | 0,8800 | 1,000 | 102 972 |
| best | cmapss | 1 | 69 µs | **N/A** | **N/A** | **N/A** | 156 896 |
| best | paderborn | 2 | 74 µs | 0,0000 | 0,3333 | 1,000 | 85 876 |

**Parité 44/44 sur l'ensemble de la grille** (les 45 cellules à parité moins la cellule N/A),
**0 erreur CRC**, **Gap 2 respecté partout** (pire cas TinyOL 148 µs, très loin de 100 ms).
Les cellules EWC et Mahalanobis, re-streamées dans la même passe avec les checkpoints
d'origine, conservent leur parité — **aucune régression**.

L'écart de latence avec Mahalanobis (69–148 µs contre 3–5 µs) est la signature attendue d'un
auto-encodeur `k→32→16→k` face à une distance en dimension k, et croît avec `k` comme prévu.

### La cellule `best × cmapss` : N/A honnête, pas un échec de parité

Cette cellule est ressortie à parité 0,000 (150 mismatches sur 150). Diagnostic : la config
`configs/best_features/tinyol_cmapss.yaml` ne retient qu'une variable, **`T2`**, qui est un
**capteur constant** de CMAPSS (écart-type nul). L'auto-encodeur s'entraîne donc sur des zéros,
son seuil calibré P95 × 1,5 vaut **exactement 0**, et la comparaison stricte `MSE > seuil` se
joue sur l'arrondi : le PC en float64 reste à 0 et prédit `normal`, la carte en float32 passe
juste au-dessus et prédit `anomalie` — systématiquement.

Ce n'est ni une perte de parité ni un défaut de portage : **la cellule ne mesure rien**.
`_degenerate_reason()` la détecte désormais (variance nulle en entrée, ou seuil calibré nul) et
consigne `parity_ok: null` + `na_reason`, avec métrique à `null` — la latence, le `.bss` et le
CRC restent, eux, des mesures légitimes.

> Dette héritée mise au jour : la sélection `best` de S3501 a pu retenir une variable
> constante. Le symptôme est local (1 cellule sur 60), mais la procédure de sélection gagnerait
> à écarter les colonnes à variance nulle. À traiter hors de cette action.

### Grille S32 — balayage de seuil, 15 cellules TinyOL (✅ terminée)

3 jeux (`cmapss`, `pronostia`, `battery`) × 5 seuils, `k = 5` : **parité 45/45**, **0 CRC**,
Gap 2 respecté. TinyOL est à **85 µs** sur les 15 cellules (dimension constante), contre
**5 µs** pour Mahalanobis : plus aucune cellule des deux modèles ne coïncide, alors qu'elles
étaient auparavant identiques au chiffre près.

Une cellule (`battery`, seuil 133) a raté son flash au premier passage — échec transitoire de
l'ST-LINK, sans rapport avec le correctif ; relancée seule, elle est passée.

## Bilan des mesures

| Grille | Cellules TinyOL | Parité | CRC | Gap 2 |
|--------|-----------------|--------|-----|-------|
| S35 (3 conditions × 5 jeux) | 15 (14 mesurées + 1 N/A) | 44/44 | 0 | ✅ 69–148 µs |
| S32 (3 jeux × 5 seuils) | 15 | 45/45 | 0 | ✅ 85 µs |

## Aval

- `src/figures/catalogs/manuscrit_final.py` : `INVALID_BOARD_MODELS` vidé (les cellules carte
  TinyOL redeviennent traçables) ; figures régénérées.
- Manuscrit ch. 5 : ligne TinyOL du tableau 5.1, réserve sous le tableau, phrase Paderborn,
  légende de la figure « accuracy contre F1 ».
- `docs/context/actions_restantes_resultats.txt` : action 2.6 → FAIT.
