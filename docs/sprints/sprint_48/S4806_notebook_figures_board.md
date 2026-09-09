# S4806 — Notebook + figures board (heatmaps symétriques au PC)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 48 |
| **Priorité** | 🟠 Importante — restitue les mesures board face au PC (théorie↔matériel). |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 2h |
| **Dépendances** | S4805 (`exp_S48_summary.json`) · S4706 (catalogue PC symétrique) |
| **Fichiers cibles** | `notebooks/cl_eval/quant_depth_board/comparison.ipynb`, `docs/figures/quant_depth_board/` |
| **Références** | `src/figures/` registre (S4201), notebook board S29/S45 |

---

## Contexte

Restitue les mesures board (S4805) en figures **symétriques au PC** (S4706), pour lire d'un coup l'écart
théorie↔matériel : la RAM théorique (S47) vs le `.bss` réel (S48), et la latence que l'émulateur ne donnait pas.

## Spec

### Figures `docs/figures/quant_depth_board/`

| Figure | Contenu | Source |
|--------|---------|--------|
| `board_auroc_vs_bits.png` | AUROC board vs `weight_bits` par granularité, par dataset (symétrique à `auroc_vs_bits.png` PC) | `exp_S48_summary.json` |
| `bss_packed_vs_unpacked.png` | `.bss` mesuré : non-packé (≈ INT8) vs packé (÷2/÷4) vs RAM théorique S47 — **le résultat clé** | `exp_S48_summary.json` + `exp_S47_depth/` |
| `latency_vs_bits.png` | Latence DWT P50/P99 vs `weight_bits` (coût du dépacking), ligne Gap 2 (100 ms) | `exp_S48_summary.json` |
| `parity_board_pc.png` | Parité pred board↔PC par cellule (attendu 1.000) | `exp_S48_parity_*.json` |
| `heatmap_board_bits_gran.png` | Heatmap AUROC board (bits × granularité) ; N/A gris | `exp_S48_summary.json` |

Notebook `notebooks/cl_eval/quant_depth_board/comparison.ipynb` : galerie FR, valeurs rechargées par cellule
(0 chiffre en dur), tableau de synthèse théorie↔matériel + reco, nbconvert sans erreur.

## Contraintes

- **0 chiffre en dur** (garde AST) ; N/A gris ; badges plateforme « mesuré board » vs « théorique PC ».
- Nbconvert du notebook sans erreur.

## Vérification

```bash
python scripts/generate_figures.py --catalog quant_depth_board
ls docs/figures/quant_depth_board/*.png | wc -l
jupyter nbconvert --to notebook --execute notebooks/cl_eval/quant_depth_board/comparison.ipynb --stdout > /dev/null
```

---

## Résolution (implémentée)

**Catalogue** `src/figures/catalogs/quant_depth_board.py` (`@register_catalog("quant_depth_board")`, `OUT_SUBDIR="quant_depth_board"`), enregistré dans `catalogs/__init__.py`. Charge l'unique source `exp_S48_summary.json` via `load_experiment` ; cellules absentes → gris/NaN (jamais 0) ; badges **mesuré board** vs **théorique PC** ; **0 chiffre en dur** (garde AST, seuls des flottants de la liste blanche layout). `build()` skip gracieux si l'agrégat manque. **5 PNG** dans `docs/figures/quant_depth_board/` :

- `bss_packed_vs_unpacked.png` (résultat clé : `.bss` non-packé ≈ INT8 vs packé) ;
- `board_auroc_vs_bits.png` (AUROC board ∥ émulateur PC) ;
- `latency_vs_bits.png` (DWT P50/P99 packé, budget Gap 2) ;
- `parity_board_pc.png` (parité par cellule, 1.000) ;
- `heatmap_board_bits_dataset.png` (AUROC board, N/A gris).

**Notebook** `notebooks/cl_eval/quant_depth_board/comparison.ipynb` : génère les figures via le catalogue, galerie FR, tableau de synthèse théorie↔matériel rechargé depuis `exp_S48_summary.json` (0 chiffre en dur), lecture (packé < non-packé ; latence ≪ 100 ms ; parité 1.000). **nbconvert sans erreur**.
