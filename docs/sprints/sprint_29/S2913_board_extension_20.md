# S2913 — Extension board INT8 5→20 (grille 4 modèles × 5 datasets)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 29 (extension O8 — board 4×5 complet) |
| **Priorité** | 🔴 |
| **Statut** | ✅ Implémenté (28 juin 2026) — driver `run_s29_board_extend.py` + **15 cellules mesurées board réelle** → grille 20/20 (18 streamées dont 2 métrique N/A mono-classe, 2 non mesurables encodeur TinyOL), 0 erreur CRC |
| **Durée estimée** | 6h (dont mesures board réelles) |
| **Dépendances** | S2912 ✅ (Maha INT8 firmware) · `scripts/run_s29_board_int8.py` ✅ · NUCLEO-F439ZI connectée (`/dev/ttyACM0`) |
| **Fichiers cibles** | `scripts/run_s29_board_extend.py` (driver) · extension `MODE_FLAGS`/`METRIC_NAME`/`RAM_WEIGHTS` de `run_s29_board_int8.py` · 15 nouveaux `experiments/exp_S29_board_int8/results_*.json` |

---

## Contexte

Le board ne couvre que **5 des 20** couples (modèle, dataset) du benchmark INT8 (S2904/S2905) :
EWC×{cwru,pronostia}, HDC×{cmapss,monitoring}, TinyOL×cwru. Le PC (Sprint 28) en couvre **20**.
S2913 complète les **15 cellules manquantes** par mesures réelles sur la NUCLEO-F439ZI, afin
d'obtenir une grille board 4×5 directement comparable au PC.

**Règle absolue (CLAUDE.md)** : aucun chiffre inventé — chaque JSON est écrit **uniquement
après** que la carte a réellement streamé la cellule.

---

## Matrice des 15 cellules manquantes

| Modèle | Datasets à mesurer | Procédure board |
|--------|--------------------|------------------|
| **EWC** | cmapss, monitoring, paderborn | même binaire (apprentissage en ligne) → stream direct |
| **HDC** | cwru, pronostia, paderborn | même binaire (projection LCG déterministe) → stream direct |
| **TinyOL** | cmapss, monitoring, pronostia, paderborn | export encodeur par dataset → `make && make flash` → stream |
| **Mahalanobis** | cmapss, cwru, monitoring, pronostia, paderborn | export poids INT8 → build `-DMAHA_INT8` → flash → stream ; FP32 = binaire défaut |

---

## Driver `scripts/run_s29_board_extend.py`

Sur le modèle de `run_board_threshold_sweep.py` / `run_feature_condition_board.py` (boucle
train→export→build→flash→stream). Pour chaque cellule de la matrice (option `--only model,dataset`) :

1. **Export poids si requis** : TinyOL (`export_weights_tinyol.py --train-dataset <ds>`) ;
   Mahalanobis (`export_weights_c.py --maha-int8` depuis un détecteur entraîné, config
   `configs/mahalanobis_int8_<ds>.yaml`). EWC/HDC : aucun export.
2. **Build/flash si requis** : TinyOL et Mahalanobis re-flashent (`make` avec `-DMAHA_INT8`
   pour Maha INT8, dims via `-D` si besoin) ; EWC/HDC réutilisent le binaire courant.
3. **Mesure** : réutilise la logique de `run_s29_board_int8.py` (FP32 puis INT8, reset DTR
   entre runs) → JSON schéma S2904 dans `experiments/exp_S29_board_int8/`.

### Extensions à `run_s29_board_int8.py`

- `MODE_FLAGS["mahalanobis"]` : FP32 = défaut (binaire FP32, flag 0x00) ; INT8 = binaire
  `-DMAHA_INT8` (flag 0x00 aussi → 2 builds distincts, gérés par le driver).
- `METRIC_NAME["mahalanobis"] = "auroc"`.
- `RAM_WEIGHTS["mahalanobis"]` : empreinte analytique mu + sigma_inv FP32 vs INT8 (cohérente
  archi, indépendante du dataset).

---

## Gestion N/A honnête

Combos attendus dégénérés (alignés sur le comportement PC) :
- **tout ×Paderborn** si les tâches de test sont **mono-classe** → AUROC non défini.
- **HDC×paderborn** si `feature_bounds` non calibrés.

Dans ces cas, le driver écrit `"metric_value": null` + `"na_reason": "<raison>"` (mono-classe,
feature_bounds…) **au lieu de forcer un chiffre**. La latence/RAM restent mesurées et écrites
(elles sont valides indépendamment de la métrique).

---

## Protocole de mesure (je pilote via Bash)

Ordre d'exécution (minimise les re-flashs) :
1. **EWC** cmapss, monitoring, paderborn (binaire courant).
2. **HDC** cwru, pronostia, paderborn (même binaire si déjà EWC/HDC ; sinon rebuild de base).
3. **TinyOL** cmapss, monitoring, pronostia, paderborn (re-flash ×4).
4. **Mahalanobis** ×5 (build FP32 + build `-DMAHA_INT8`, flash ×2 par cellule).

Contrôles par run : `crc_errors == 0`, latence p99 < 100 ms (Gap 2), schéma S2904 valide.

---

## Critères d'acceptation

- 15 nouveaux JSON dans `experiments/exp_S29_board_int8/` (total **20**), schéma identique
  aux 5 existants.
- Cellules N/A explicitement `null` + `na_reason` (jamais de chiffre forcé).
- 0 erreur CRC sur tous les runs ; Gap 2 respecté partout.
- Parité Mahalanobis board↔PC vérifiée sur ≥ 1 dataset non dégénéré (cmapss).
