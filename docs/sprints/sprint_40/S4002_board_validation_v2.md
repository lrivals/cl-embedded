# S4002 — Validation board NUCLEO-F439ZI du kernel v2 (récupération INT8 réelle)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 40 |
| **Priorité** | 🔴 Critique — **board réelle requise** (lève l'axe « émulé PC » de l'article) |
| **Statut** | 🟡 Partiellement mesuré — **4 cellules `per_channel` streamées carte réelle** (2 jeux × frozen/online) ; `q15` (×4) et A/B `int8_legacy` v2 (×4) restent au banc (cf. `S4010`) |
| **Durée estimée** | ~10h (quand carte branchée) |
| **Dépendances** | S4001 (kernel v2 + export + tests host) · `make flash` + `/dev/ttyACM0` · exp_S36 (conditions de référence) |
| **Fichiers cibles** | `scripts/run_s40_board_v2.py` → `experiments/exp_S40_board_v2/` |
| **Références** | `scripts/run_sprint36_board.py` (squelette apparié) · `scripts/board_pc_parity.py` · S3915/S3916/S3919 |

## Contexte

L'article affirme que la PTQ INT8 « legacy » s'effondre **et** qu'un kernel calibré la récupère. Le premier
point est mesuré board (Sprint 36) ; le second n'était qu'**émulé PC** (Sprint 39). Ce ticket
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


## Résultats mesurés (carte réelle NUCLEO-F439ZI)

Les 4 cellules `per_channel` du kernel v2 ont été flashées et streamées, **0 erreur CRC** :

| Jeu | Protocole | F1 | Parité vs émulateur | Accord INT8↔FP32 | Latence P50 | `.bss` |
|-----|-----------|----|---------------------|------------------|-------------|--------|
| Monitoring | gelé | 0.9173 | 1.000 | 0.9996 | 65 µs | 101 236 B |
| Monitoring | en ligne | 0.9016 | 0.9885 | 0.8337 | 577 µs | 101 236 B |
| Pronostia | gelé | 0.8995 | 1.000 | 0.9951 | 68 µs | 106 152 B |
| Pronostia | en ligne | 0.9212 | 0.9736 | 0.8100 | 606 µs | 106 152 B |

Lecture : le F1 revient au niveau FP32 (0.9194 / 0.9164) et la **parité gelée est exacte** — le portage
du kernel calibré se fait sans perte. L'accord INT8↔FP32 chute en ligne parce que les deux trajectoires de
poids divergent sous mise à jour continue, sans que la métrique finale en souffre.

**Reste au banc** (8 cellules sur 12) : `q15` × {2 jeux} × {gelé, en ligne} et la reprise A/B de
`int8_legacy` sous le kernel v2, plus les fichiers `parity_{scheme}_frozen_{ds}.json` attendus par
`TestS40Parity`. Ces cellules portent `"à mesurer"` partout — voir `S4010_mesures_manquantes.md`.
