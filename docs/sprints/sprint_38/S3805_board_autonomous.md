# S3805 — Board P2/P3 : mise à jour autonome + mesure d'économie

| Champ | Valeur |
|-------|--------|
| **Sprint** | 38 |
| **Priorité** | 🔴 Critique — démontre l'autonomie réelle (carte sans PC) et produit les chiffres d'économie. |
| **Statut** | ✅ Implémenté — board réelle NUCLEO-F439ZI, 8 cellules gated (P2/P3 × 2 ds × 2 init) + refs scratch complétées. |
| **Durée estimée** | 6h |
| **Dépendances** | S3803 ✅ (`-DEWC_AUTO_UPDATE`, `drift_thresholds.h`) · S3804 ✅ (driver + bornes) · `scripts/export_weights_c.py` (`--drift-thresholds`) |
| **Fichiers cibles** | `scripts/run_sprint38_board.py` (`--policy gated_truelabel\|gated_pseudolabel`), `experiments/exp_S38_board_{gated_truelabel,gated_pseudolabel}_{dataset}/results.json` |
| **Références** | S3805 = chemin autonome ; le gate décide à bord (cf. S3803) |

---

## Contexte

P2/P3 sont les politiques **autonomes** : c'est le gate embarqué (maha + fenêtre glissante) qui décide
des mises à jour, **sans intervention de l'hôte**. On streame **sans `--update`** : le bit UART n'a plus
d'effet, le firmware (compilé `-DEWC_AUTO_UPDATE`) tranche seul. Le vrai label reste transmis mais sert
seulement au scoring/parité (jamais au SGD en P3).

## Spec

Par dataset :
1. `export_weights_c.py --mahal --ewc-head --drift-thresholds` (seuils ← `drift_thresholds.json` S3802).
2. Build :
   - P2 : `make EXTRA_CFLAGS="-DEWC_AUTO_UPDATE" EWC_IN=5 MAHA_DIM=5 all`.
   - P3 : `make EXTRA_CFLAGS="-DEWC_AUTO_UPDATE -DGATE_PSEUDO_LABEL" EWC_IN=5 MAHA_DIM=5 all`.
   Lire `.bss` (delta vs défaut = coût RAM du gate). `make flash`.
3. Stream **sans `--update`**, split complet.

Métriques stockées → `results.json` :
- **`n_updates` réels** (compteur firmware renvoyé) + **`update_rate`** = `n_updates / n_samples`.
- **`gate_overhead_us`** : surcoût par échantillon (maha_score + drift_update) vs frozen S3804.
- **`mean_latency_us`** : latence moyenne effective (gate sur tous les échantillons + SGD seulement sur flags).
- **`bss_bytes`** + **`bss_delta_vs_default`** (coût RAM gate : drift detector + ring buffer).
- **parité verdicts board↔PC** (taux d'accord NORMAL/DRIFT/FAULT) ; prédictions board↔PC.
- `acc`/`f1`/`af`, `gap2_ok` (< 100 ms).

## Vérification

```bash
python scripts/run_sprint38_board.py --policy gated_truelabel    --dataset monitoring --port /dev/ttyACM0
python scripts/run_sprint38_board.py --policy gated_pseudolabel  --dataset pronostia  --port /dev/ttyACM0
```
- `update_rate` P2/P3 < 1 (le gate filtre) → **latence moyenne < `always`** (économie de SGD).
- Parité verdicts board↔PC élevée (mêmes seuils exportés → mêmes décisions, aux arrondis float près).
- `.bss` delta modeste (~quelques centaines d'octets : `g_drift`).
- Toutes latences ≪ 100 ms (Gap 2).

## Résultats d'implémentation

**Verrou levé** : la réponse UART V3 ne transporte ni `n_updates` ni le verdict. Sous
`-DEWC_AUTO_UPDATE` **uniquement** (build par défaut strictement inchangé → 0 régression), on
**réinterprète 2 champs du snapshot** dans `pipeline.c` (chemin EWC) — `snap.auroc ←
(float)g_last_verdict` (0=NORMAL,1=FAULT,2=DRIFT) et `snap.forgetting ← (float)g_n_updates` (compteur
cumulé) — `snap.accuracy` inchangé. **Wire format V3 identique** → aucune modif de `sensor_stream.py`
(les champs `auroc`/`forgetting` sont déjà exposés par échantillon ; seule leur sémantique change,
documentée). Le driver `run_sprint38_board.py` décode `n_updates = last.forgetting`,
`verdict_board = round(auroc)`.

**Build** : `.bss` défaut **105 036 B** invariant ; gate **105 336 B (+300 B** : `g_drift`+`g_n_updates`
+`g_last_verdict`). P2/P3 compilent sans warning.

**Driver** : `build_and_flash_gated` flashe la **Maha d'enrôlement** (welford, P95 sur les `n_enr=500`
premiers sains = miroir exact PC, ≠ `train_maha_board`) + tête EWC PC + seuils (`--drift-thresholds`).
Stream **sans `--update`** ; le vrai label reste transmis (SGD P2 + scoring). `_pc_gate_replay`
reconstruit `verdict_pc` sur l'ordre board (avec `maha.partial_fit` sur DRIFT en P3) → parité.

**Board réelle NUCLEO-F439ZI — 8 cellules gated (0 CRC, Gap 2 ✅)** :

| cellule | n_updates | update_rate | verdict_parity | gate_ovh | F1_faulty |
|---------|-----------|-------------|----------------|----------|-----------|
| gated_truelabel × monitoring × {pretrained,scratch} | 196/7671 | 0.0256 | **1.000** | ~27 µs | 0.919 / 0.182 |
| gated_pseudolabel × monitoring × {pretrained,scratch} | 196/7671 | 0.0256 | **1.000** | ~27 µs | 0.919 / 0.182 |
| gated_truelabel × pronostia × {pretrained,scratch} | 186/7533 | 0.0247 | **1.000** | ~28 µs | 0.889 / 0.613 |
| gated_pseudolabel × pronostia × {pretrained,scratch} | 186/7533 | 0.0247 | **1.000** | ~28 µs | 0.504 / 0.183 |

`update_rate` strictement ordonné **frozen=0 < gated≈0.025 < always=1** ; `mean_latency` gated ≈ 79–82 µs
≪ always 238–251 µs (économie du SGD filtré) ≪ 100 ms ; `.bss` Δ=+300 B (gate). **Parité verdict
board↔PC = 1.000 sur les 8 cellules** (mêmes seuils exportés ⇒ décision d'update identique).
Refs `scratch` frozen/always complétées pour une economy_table comparable.
