# S4002 — Validation board NUCLEO-F439ZI du kernel v2 (récupération INT8 réelle)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 40 |
| **Priorité** | 🔴 Critique — **board réelle requise** (lève l'axe « émulé PC » de l'article) |
| **Statut** | 📝 Doc — spec prête ; **différé si carte indisponible** (reprend S3915/S3916/S3919) |
| **Durée estimée** | ~10h (quand carte branchée) |
| **Dépendances** | S4001 (kernel v2 + export + tests host) · `make flash` + `/dev/ttyACM0` · exp_S36 (conditions de référence) |
| **Fichiers cibles** | `scripts/run_s40_board_v2.py` → `experiments/exp_S40_board_v2/` |
| **Références** | `scripts/run_sprint36_board.py` (squelette apparié) · `scripts/board_pc_parity.py` · S3915/S3916/S3919 |

## Contexte

L'article affirme que la PTQ INT8 « legacy » s'effondre **et** qu'un kernel calibré la récupère. Le premier
point est mesuré board (Sprint 36) ; le second n'est aujourd'hui qu'**émulé PC** (Sprint 39). Ce ticket
apporte la **preuve matérielle** de la récupération : flasher le kernel v2 (S4001) et mesurer, **dans des
conditions strictement identiques à exp_S36**, la F1 board, l'accord INT8↔FP32 et la parité board↔PC.

## Spec — driver `run_s40_board_v2.py`

Réutilise `run_sprint36_board.py` comme squelette (mêmes séquences / seed / ordre de streaming ⇒ résultats
**comparables** à exp_S36 et exp_S39). Par cellule : train réf → export `--int8-v2` → build → flash →
stream **sans `--update`** (frozen) puis **avec** (online).

- **Grille** : schémas {`per_channel`, `q15`} × datasets {`pronostia`, `monitoring`} × protocoles
  {`frozen`, `online`} → `experiments/exp_S40_board_v2/results_{scheme}_{dataset}_{proto}.json`.
- **A/B v1 vs v2** : flasher aussi le v1 (INT8 legacy) sur le même stream → confirme la récupération F1
  board (v1 0.07–0.15 → v2 ≈ FP32).

| Métrique | Source | Critère |
|----------|--------|---------|
| Latence inférence (+MAJ) | DWT P50/P99 | ≪ 100 ms (**Gap 2**) |
| `.bss` total | `make size` | < 256 Ko |
| F1 board v2 | stream étiqueté | ≈ FP32 (Δ≤0.02) ; comparer émulateur S3904 |
| Accord INT8-v2 ↔ FP32 | stream apparié | ≥ 0.95 (per-channel/q15) |
| Parité board↔PC | `board_pc_parity.py` | frozen = 1.000 exact ; online ≈ documenté |
| CRC | UART | 0 erreur |
| RAM poids v2 | export | INT8 ÷4 / Q15 ÷2 vs FP32 (**Gap 3**) |

## Vérification

```bash
arm-none-eabi-gcc --version              # toolchain ARM
ls /dev/ttyACM0                          # carte branchée
python scripts/run_s40_board_v2.py --scheme per_channel --dataset pronostia --proto frozen
python scripts/run_s40_board_v2.py --scheme q15 --dataset monitoring --proto online
python scripts/board_pc_parity.py --exp exp_S40_board_v2
```

> **Honnêteté** : aucun chiffre board écrit tant que la carte n'a pas streamé. `experiments/exp_S40_board_v2/`
> reste absent jusqu'à exécution réelle ; les cellules correspondantes de l'article portent `"à mesurer"`.
>
> `FIXME(gap3)` : si la récupération board est confirmée (per-channel/Q15 F1 ≈ FP32, RAM ÷4/÷2), le Gap 3
> passe de « partiel » à **contribution positive** (RAM réduite SANS perte de métrique sur MCU réel).
