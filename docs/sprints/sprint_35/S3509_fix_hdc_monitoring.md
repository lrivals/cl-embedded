# S3509 — Correction de l'artefact HDC×monitoring (0.113)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 35 |
| **Priorité** | 🔴 Critique — corrige une valeur fausse affichée en présentation |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 3h |
| **Dépendances** | S3506 (HDC_N_FEATURES configurable), S3508 (streaming board) |
| **Fichiers cibles** | `experiments/exp_S35_board_5feat_hdc_monitoring/`, `experiments/comparison_sprint23.json` |
| **Références** | `experiments/exp_S33_board_gap1/results_hdc_monitoring.json` (`note_feature: "monitoring zéro-paddé 4→5 feat (5ᵉ synthétique nulle)"`, `online_accuracy=0.1133`) ; valeur PC légitime `exp_S33_PC_hdc_monitoring` = 0.8498 |

---

## Contexte

`monitoring` est natif **4 features**. Le firmware 5-feat le zéro-padde à 5 (5ᵉ feature synthétique
nulle), ce qui **casse la projection HDC embarquée** → `acc_final=0.1133` (dégénéré, ≈ hasard).
Cette valeur fausse apparaît dans `comparison_sprint23.json` et la heatmap de présentation.

## Spec

Cause racine : zéro-padding 4→5 incompatible avec HDC (la 5ᵉ dim nulle pollue le hypervecteur).
Correction = **re-run board HDC×monitoring sans padding dégénéré** :

- Builder un firmware HDC à `HDC_N_FEATURES=4` (dims natives monitoring, via S3506) — OU appliquer
  la condition `all`/`best` monitoring (4 features) où le padding n'existe pas.
- Streamer HDC×monitoring sur board → `experiments/exp_S35_board_5feat_hdc_monitoring/results.json`
  avec la **vraie valeur** (attendue cohérente avec le PC ≈ 0.85, à mesurer, pas à inventer).
- Mettre à jour la cellule `comparison_sprint23.json["results"]["monitoring"]["hdc"]["nucleo_f439zi"]` :
  remplacer `0.1133` + la note artefact par la valeur mesurée et une note explicite.

**Règle** : valeur **mesurée** sur board, jamais recopiée du PC ni inventée. Si la board n'a pas
encore tourné, champ « à mesurer » + note expliquant l'artefact d'origine.

## Implémentation (✅)

- **Re-run board corrigé** via la condition `all` (choix utilisateur) : firmware buildé à
  `HDC_N_FEATURES=4` (monitoring natif, **sans zéro-padding**) → cellule réelle
  `exp_S35_board_all_hdc_monitoring`. Valeur **mesurée board = `online_accuracy=0.8788`**
  (cohérente avec le PC `0.8498`, vs artefact `0.1133`).
- `generate_comparison_sprint23.py::_apply_s3509_override` remplace
  `comparison_sprint23.json["results"]["monitoring"]["hdc"]["nucleo_f439zi"]["acc_final"]`
  par `0.8788` + note explicite (l'artefact 0.1133 n'apparaît plus). **Jamais édité à la main**
  (régénéré par le script). La heatmap board (S3510) affiche désormais la valeur corrigée.
- Le re-run `5feat` (`exp_S35_board_5feat_hdc_monitoring`, monitoring 5feat≡4-feat) produit la
  même valeur non dégénérée par le balayage S3508.

## Vérification

```bash
python scripts/run_feature_condition_board.py --port /dev/ttyACM0 --model hdc --dataset monitoring
python -c "import json; c=json.load(open('experiments/comparison_sprint23.json')); \
v=c['results']['monitoring']['hdc']['nucleo_f439zi']['acc_final']; assert v != 0.1133, 'artefact non corrigé'"
```
