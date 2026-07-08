# S3915–S3917 — Validation board NUCLEO-F439ZI

| Champ | Valeur |
|-------|--------|
| **Sprint** | 39 |
| **Priorité** | 🔴 Critique (S3915/S3916) · 🟢 (S3917) — **board réelle requise** |
| **Statut** | ✅ S3915/S3916 implémentés (board réelle, 1er juil. 2026) · ⬜ S3917 différé (`TODO(dorra)` CMSIS-NN) |
| **Durée estimée** | ~10h (quand carte disponible) |
| **Dépendances** | S3907 ✅ (kernel v2) · S3908 ✅ (header généré) · S3910 (spec SIMD) · `make flash` + `/dev/ttyACM0` |
| **Fichiers cibles** | `scripts/run_s39_board.py`, `experiments/exp_S39_board/`, `firmware/.../ewc_head_int8_v2_simd.c` |
| **Références** | `scripts/run_sprint36_board.py` · `scripts/board_pc_parity.py` · `scripts/run_s29_board_int8.py` |

---

## ✅ Réalisé (board réelle NUCLEO-F439ZI, 1er juillet 2026)

**Câblage firmware (verrou levé).** Le kernel v2 (`ewc_head_int8_v2.c/.h`, S3907) n'était pas
branché sur le pipeline. Sélection **à la compilation** `-DEWC_INT8_V2` (nibble protocole saturé
→ mirroir `-DMAHA_INT8`, S2912) : dans `pipeline.c`, le chemin `0x40` (`PROTO_FLAG_INT8_MODE`)
route vers `ewc_int8_v2_forward` au lieu du v1. **Wire format UART inchangé** (`sensor_stream.py`
intact, `--model ewc-int8`). `g_ewc_int8_v2` quantifié au boot via
`ewc_int8_v2_from_fp32_calib(&g_ewc_int8_v2, &g_ewc_head, EWC_V2_ACT_MAX)` — `act_max` ajouté au
header généré (`export_weights_c.py --int8-v2` → `EWC_V2_ACT_MAX[3]`) pour choisir le bon QMAX
par variante (int8/q15/mixed). **Build v1 défaut `.bss=105 036 B` invariant → 0 régression**
(`make test` 127/2 TinyOL préexistants) ; v2 compile per_channel/q15/mixed sans erreur.

**Driver** `scripts/run_s39_board.py` (réutilise le checkpoint apparié `exp_S39_matched/` → poids
identiques PC↔board ; train→export `--ewc-head`+`--int8-v2`→build flags du schéma→flash→stream
**sans `--update`**→parité gelée vs émulateur sur les features réellement streamées).

**5 cellules board réelle mesurées (0 CRC, Gap 2 ✅, parité gelée bit-exacte 1.000 / 0 mismatch) :**

| Schéma | Dataset | F1 board | F1 émul (S3918) | `.bss` | Lat P50 | Parité |
|--------|---------|:--------:|:---------------:|:------:|:-------:|:------:|
| legacy_c (v1) | pronostia | 0.078 | 0.066 | 105 036 B | 53 µs | 1.000 |
| per_channel_int8 (v2) | pronostia | **0.928** | 0.943 | 106 152 B | 68 µs | 1.000 |
| q15 (v2) | pronostia | **0.970** | 0.962 | 106 856 B | 75 µs | 1.000 |
| legacy_c (v1) | cmapss | 0.133 | 0.227 | 105 036 B | 53 µs | 1.000 |
| per_channel_int8 (v2) | cmapss | **0.400** | 0.448 | 106 152 B | 67 µs | 1.000 |

**S3916 A/B v1 vs v2 (board réelle)** : le kernel v2 **récupère la F1 sur silicium réel**
(pronostia 0.078→0.928, cmapss 0.133→0.400), confirmant le diagnostic PC (émulateur) sur le
matériel. Le coût : `.bss` +1.1–1.8 Ko (deuxième tête), latence +14–22 µs (déquant→FP32 sur FPU,
cohérent S29) — **toutes ≪ 100 ms (Gap 2 ✅)**. Écart F1 board↔émulateur = sous-échantillon
streamé (300 éch. aléatoires) ≠ split complet ; la **parité gelée bit-exacte (0 mismatch)**
prouve que board et émulateur calculent la même chose échantillon par échantillon.

**S3917 (SIMD CMSIS-NN)** : **différé** (`TODO(dorra)`) — sources CMSIS-NN présentes
(`stm32f4_cubemx/Drivers/CMSIS/NN/`) mais non intégrées au build `stm32f4_blink` ; sémantique
d'accumulation `arm_fully_connected_s8` distincte du kernel v2 (déquant→FP32) → investigation
séparée, pas de parité bit-à-bit. Ne bloque pas la validation S3915/S3916.

---

## Contexte

Toute la méthodologie (Partie A) est faite au PC via l'émulateur. Restent les mesures qui **exigent
physiquement la carte** : latence DWT réelle, `.bss` cible, parité streaming board↔PC, et le bench SIMD.
Ces tâches sont **documentées maintenant, exécutées plus tard** ; les drivers et specs sont prêts dès la
Partie A pour qu'il suffise de flasher.

## S3915 — Mesures board du kernel v2

`run_s39_board.py` (train réf → export `--int8-v2` → build → flash → stream **sans `--update`**) pour
chaque schéma (per-channel, q15, mixte) × datasets board :

| Métrique | Source | Critère |
|----------|--------|---------|
| Latence inférence | DWT P50/P99 | ≪ 100 ms (Gap 2) |
| `.bss` total | `make size` | < 256 Ko |
| F1 board | stream étiqueté | comparer à émulateur (S3904) |
| Parité board↔PC | `board_pc_parity.py` | accord ≥ 0.95 (per-channel/q15) |
| CRC | UART | 0 erreur |

→ `experiments/exp_S39_board/results_{scheme}_{dataset}.json`.

## S3916 — A/B v1 vs v2 (board réelle)

Confirmer sur carte que le kernel v2 **récupère la F1** vs v1 (qui donnait 0.07–0.15, Sprint 36) :
flasher les deux builds, streamer le même jeu, comparer F1 et accord. Valide que le diagnostic PC
(émulateur) tient sur le matériel réel.

## S3917 — Bench SIMD CMSIS (lève S2908)

Si la toolchain CMSIS est débloquée (`TODO(dorra)`) : build `-DUSE_CMSIS_NN`, bench DWT scalaire v2 vs
`arm_fully_connected_s8` → `results_simd.json`. Tranche la question latence INT8 (cf. S3910).

## Vérification (quand carte branchée)

```bash
arm-none-eabi-gcc --version          # toolchain ARM
ls /dev/ttyACM0                      # carte branchée
python scripts/run_s39_board.py --scheme q15 --dataset pronostia
python scripts/board_pc_parity.py --exp exp_S39_board
```

> **Honnêteté** : aucun chiffre board n'est écrit tant que la carte n'a pas streamé (règle « pas de
> résultat inventé »). Les JSON `exp_S39_board/` restent absents jusqu'à exécution réelle.
