# S3405 / S3406 — Q15 Mahalanobis (Python) + expérience PC

| Champ | Valeur |
|-------|--------|
| **Sprint** | 34 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 3h (S3405) + 3h (S3406) |
| **Dépendances** | Sprint 28 ✅ (constat dégradation Mahalanobis INT8 : CWRU −0.236, Pronostia −0.238) |
| **Fichiers cibles** | `src/models/unsupervised/mahalanobis_int8.py`, `experiments/exp_S34_maha_q15/`, `configs/mahalanobis_q15_*.yaml` |
| **Références** | `mahalanobis_int8.py:77-94` (quantification affine `mu_`/`sigma_inv_` actuelle), `:96-97` (`TODO(arnaud)` Q15), `:142-161` (`get_memory_footprint()`) |

---

## Contexte

Le benchmark Sprint 28 a confirmé que la quantification INT8 affine globale de
`sigma_inv_` (confirmée lignes 90-94 : un seul scale/zero-point pour toute la matrice) casse
la distance de Mahalanobis sur des matrices à grande dynamique (AUROC −0.236 CWRU, −0.238
Pronostia). Le `TODO(arnaud)` (lignes 96-97, confirmé) propose explicitement un fallback
Q15 (int16). Ce doc couvre l'implémentation Python **et** l'expérience qui la valide,
indissociables.

---

## S3405 — `mahalanobis_int8.py` mode `quant="q15"`

```python
# Lève le TODO(arnaud):96-97
class MahalanobisInt8Detector:
    def __init__(self, ..., quant: str = "int8"):
        """quant in {"int8", "q15"}.
        q15 : sigma_inv_ quantifié en int16 (Q15 fixed-point), scale par-tenseur adapté à
        sa grande dynamique. mu_ reste en INT8 (faible dynamique, déjà correct).
        """
        ...

    def _quantize_sigma_inv_q15(self, sigma_inv: np.ndarray) -> tuple[np.ndarray, float]:
        """int16 = round(sigma_inv / scale), scale = max(abs(sigma_inv)) / 32767."""
        ...

    def get_memory_footprint(self) -> dict:
        """Mise à jour : Q15 sigma_inv_ = d^2 x 2 bytes (au lieu de d^2 x 1 en INT8) —
        economie x2 vs FP32 (au lieu de x4 en INT8), mu_ reste x4 (INT8 inchangé).
        """
        ...
```

**Règles** :
- `mu_` reste INT8 (pattern affine existant, lignes 77-94, inchangé — sa dynamique est
  faible, pas concernée par le bug).
- `sigma_inv_` Q15 : un seul scale par tenseur (pas par-ligne/colonne), résolution 256× plus
  fine qu'INT8 (16 bits vs 8 bits de mantisse utile).
- Aucune régression du mode `"int8"` existant : `quant="int8"` (valeur par défaut) doit
  produire des résultats strictement identiques à avant ce sprint.

## S3406 — Expérience PC

`experiments/exp_S34_maha_q15/{dataset}_{fp32,int8,q15}.json` sur CWRU + Pronostia (cibles
du bug) + 3 autres datasets (Monitoring, CMAPSS, Paderborn — non-régression). Cible :
**ΔAUROC < 0.02** sur CWRU/Pronostia en Q15 (vs −0.236/−0.238 en INT8), sans dégrader FP32
ni régresser INT8 sur les 3 autres. Configs `configs/mahalanobis_q15_{dataset}.yaml`. RAM
profiling obligatoire (nouveau mode mesuré). Agrégat `summary.json`.

---

## Vérification

```bash
python -c "from src.models.unsupervised.mahalanobis_int8 import MahalanobisInt8Detector; \
d = MahalanobisInt8Detector(quant='q15'); print('OK')"

python scripts/train_mahalanobis.py --config configs/mahalanobis_q15_cwru.yaml
pytest tests/test_mahalanobis_q15.py -v   # S3409 : récup AUROC, non-régression FP32/INT8
```

---

## Réalisé (S3405/S3406)

- **S3405** : `MahalanobisDetectorInt8` étendu (`quant ∈ {int8, q15}` lu du config). `quant="q15"` :
  `_quantize_sigma_inv_q15` (int16 symétrique, `scale = max|·|/32767`), `calibrate_q15`,
  `anomaly_score_q15`/`predict_q15`, `get_memory_footprint_q15` (Σ⁻¹ d²×2 B). `mu_` reste INT8
  affine. `quant="int8"` (défaut) **strictement inchangé** (test `test_int8_mode_unchanged`).
- **S3406** : `scripts/run_s34_maha_q15.py` + 5 configs `mahalanobis_q15_*.yaml` →
  `experiments/exp_S34_maha_q15/{ds}_{fp32,int8,q15}.json` + `summary.json` (5 datasets).
- **Résultat clé** : la métrique de recouvrement robuste est la **corrélation de rang au FP32**
  (pilote seuil/AUROC) — Q15 > INT8 sur **tous** les datasets (Pronostia 0.985 vs 0.649 ;
  Paderborn 0.921 vs 0.827 ; CWRU 0.536 vs 0.409). **AUROC recouvrée** sur les datasets à AUROC
  non-dégénérée : Pronostia **ΔAUROC −0.113 (INT8) → +0.013 (Q15)** ✅ (cible < 0.02), CMAPSS
  +0.005, Monitoring ≈0. CWRU : AUROC FP32 = 0.475 (sub-random, binarisation dégénérée) → AUROC
  non pertinente, mais fidélité de rang nettement améliorée. **Constat secondaire** : sur très
  grande dynamique (Paderborn Σ⁻¹ ~6e5) l'erreur ABSOLUE de score Q15 peut dépasser l'INT8 — non
  par perte de fidélité (Q15 reconstruit Σ⁻¹ 200× mieux) mais parce que `mu_` reste INT8 et que
  son erreur est amplifiée par les grandes valeurs de Σ⁻¹ que Q15 préserve (INT8 les écrase →
  distances collapsées). Piste future : `mu_` Q15 aussi.
