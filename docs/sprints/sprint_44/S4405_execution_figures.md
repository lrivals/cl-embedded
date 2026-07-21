# S4405 — Exécution de la grille PC + figures d'impact

| Champ | Valeur |
|-------|--------|
| **Sprint** | 44 |
| **Priorité** | 🟠 Haute — produit les chiffres et visuels comparant tous les détecteurs sur tous les datasets. |
| **Statut** | ✅ Implémenté — `scripts/run_sprint44_pc.py` (grille 36/36) + `src/figures/catalogs/drift_detection_pc.py` (5 PNG). |
| **Durée estimée** | 6h |
| **Dépendances** | S4402 ✅ + S4403 ✅ (détecteurs) · S4404 ✅ (harnais) · S4302 ✅ (loaders) · `src/evaluation/plots.py` ✅ |
| **Fichiers cibles** | `scripts/run_sprint44_pc.py`, `experiments/exp_S44_PC_{detector}_{dataset}/results.json`, `docs/figures/drift_detection_pc/*.png` |
| **Références** | Pattern d'exécution Sprint 38 S3802 (`run_sprint38_pc.py`) · règle « `null` tant que non exécuté » |

---

## Contexte

Rassemble détecteurs (S4402/S4403) + baseline + harnais (S4404) sur les datasets S43 en une **grille
reproductible**, et produit les figures d'impact qui répondent à « quel détecteur, à quel coût, pour quel
délai ». C'est la matière première de la reco MCU (S4406) et des slides.

## Spec

### 1. Driver — `scripts/run_sprint44_pc.py`

CLI `--detector <nom> --dataset <nom>` (et `--all`). Pour chaque cellule :
1. Charge le `DriftDataset` (S4302) ; segment d'enrôlement → `set_params_from_reference`.
2. **Streaming séquentiel** : rejoue le flux échantillon par échantillon.
   - Détecteurs **non-supervisés** : `update(feature)` (feature/agrégation ← config).
   - Détecteurs **supervisés** : `update(error_t)` où `error_t` vient du modèle de faute
     (`error_stream`, S4402) — le vrai label alimente le flux d'erreur, jamais le détecteur directement.
3. Collecte les `verdicts`, applique le harnais S4404 → métriques de détection + coût (proxies).
4. Écrit `experiments/exp_S44_PC_{detector}_{dataset}/results.json` : verdicts, `drift_metrics`,
   `cost` (proxies PC, `_proxy:true`), `config_snapshot`, `requires_label`, `viabilite_mcu`.
5. Grille = (≈ 9 détecteurs) × (datasets S43) × (axe supervisé/non-supervisé selon `requires_label`).

**Règles** : `null` tant que non exécuté (aucun chiffre inventé) ; seed 42 ; même loader/segment que S45.

### 2. Figures d'impact — `docs/figures/drift_detection_pc/`

Chargées depuis les `results.json` :
- **Délai vs FAR** (scatter par détecteur) — le compromis central ; famille supervisée vs non-supervisée
  distinguées par couleur.
- **Courbe statistique vs points de drift** : la statistique interne du détecteur dans le temps,
  `drift_points` en verticales, alarmes marquées → lisibilité du déclenchement.
- **RAM (state_bytes) / latence par détecteur** — barres, annotées viabilité MCU (prépare S45).
- **Heatmap détecteur × dataset** : F1 de détection (et délai) — vue d'ensemble.
- **Supervisé ∥ non-supervisé** : figure de synthèse de l'axe d'étude (précision vs autonomie/coût).

## Contraintes

- Aucune donnée brute committée ; figures régénérables depuis JSON (0 valeur inline).
- Cellules non calculables (délai sur Electricity/NOAA) → `null`/gris, honnête.
- Distinguer visuellement **proxy-PC** (RAM/latence ici) de la mesure board (S45) dans les légendes.
- Réutiliser `plots.py` (backend Agg) — style commun, pas de duplication.

## Vérification

```bash
python scripts/run_sprint44_pc.py --detector page_hinkley --dataset synthetic   # → exp_S44_PC_page_hinkley_synthetic/
python scripts/run_sprint44_pc.py --all
ls docs/figures/drift_detection_pc/
```
- Sur le **synthétique** (points exacts), les détecteurs viables ont un `mean_detection_delay` fini et un
  `missed_detection_rate` faible → valide la chaîne complète.
- Ordre attendu du coût : Page-Hinkley/DDM (O(1)) < PSI (O(bins)) < KSWIN/KS/MMD (O(W)) — vérifiable sur
  la figure RAM.
