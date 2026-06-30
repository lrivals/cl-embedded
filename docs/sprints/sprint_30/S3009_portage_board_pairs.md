# S3009 / S3010 — Portage board : paires arbitraires (généralisation DUAL_MODE)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 30 |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Implémenté (board réelle) |
| **Durée estimée** | 3h (S3009) + 2h (S3010) |
| **Dépendances** | Sprint 27 ✅ (DUAL_MODE `pipeline.c`) · S3006 (paires PC validées) · `firmware/.../mahalanobis.c` ✅ |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/pipeline.c`, `firmware/stm32f4_blink/inc/pipeline.h`, `scripts/sensor_stream.py`, `experiments/exp_S30_board_*/` |
| **Références** | `firmware/stm32f4_blink/src/pipeline.c` (l.430-478 DUAL_MODE), `scripts/board_experiment_recorder.py` |

---

## Contexte

Le Sprint 27 a câblé DUAL_MODE en dur pour EWC_REG + EWC_MC. S3009 le **généralise** pour exécuter une paire **Mahalanobis + {HDC, EWC, TinyOL}** sélectionnée par FLAGS, en réutilisant l'infrastructure de réponse combinée. Mahalanobis (128 B) tourne toujours comme détecteur ; le 2ᵉ slot est paramétrable.

## S3009 — pipeline.c + sensor_stream.py

- Étendre la branche DUAL_MODE : `g_detector` (Mahalanobis) + un modèle supervisé sélectionné par bits FLAGS.
- Réponse = `[pred_maha, score_maha, pred_sup, conf_sup, lat_us, ...]`.
- **MAJ `sensor_stream.py` en parallèle** (`parse_response`, format de trame) — règle CLAUDE.md : protocole UART ↔ sensor_stream jamais désynchronisés.
- Vérifier l'absence de collision de bits FLAGS (byte saturé, TODO dorra S2600).

## S3010 — Expériences board (RAM profiling obligatoire)

- ≥1 paire via `board_experiment_recorder.py` : latences **séparées** (Maha seul, supervisé seul) et **combinée**, `.bss` total, métriques en ligne (AUROC/F1).
- Vérifier latence combinée << 100 ms (Gap 2).

---

## Vérification

```bash
cd firmware/stm32f4_blink && make all && arm-none-eabi-size build/stm32f4_blink.elf  # .bss < 256 Ko
make test    # 0 nouvelles régressions
python scripts/board_pair_recorder.py --pair maha-ewc --dataset cwru --dry-run --output /tmp/exp_pair
```

---

## Bilan d'implémentation (S3009 + S3010)

**Constat FLAGS** : le byte est saturé en bits individuels, mais le **nibble haut**
(sélecteur de mode) a des valeurs libres. On y place 3 modes paire (aucune collision avec
EWC 0x10…TINYOL_INT8 0xC0) : `PROTO_FLAG_PAIR_MAHA_EWC=0x90`, `..._HDC=0xA0`, `..._TINYOL=0xB0`
(masque `PROTO_PAIR_MODE_MASK=0xF0`). Dispatch placé **avant** DUAL/MULTICLASS (règle exact-match avant subset).

**Firmware** (`pipeline.c`/`pipeline.h`) : bloc `PAIR_MODE` co-exécutant `g_detector` (Mahalanobis
sur copie z-scorée — `raw` non clobbé) + supervisé {EWC/HDC/TinyOL} sur `raw` brut (parité board↔PC).
Nouvelle réponse `uart_send_response_pair` **22 B** : `[pred_maha:u8][score_maha:f32][pred_sup:u8][conf_sup:f32][lat_us:u32][auroc_maha:f32][f1_sup:f32]`.
`sensor_stream.py` mis à jour en parallèle (constantes miroir, `RESPONSE_PAIR_FMT`, `parse_response`,
choix `--model pair-maha-{ewc,hdc,tinyol}`, dispatch taille réponse).

**Build/tests** : `make all` OK, `.bss=104 576 B` (39.9 % de 256 Ko / 53.4 % de la SRAM 192 Ko) ;
`make test` = 92 tests, **+3 PAIR PASS** (T80–T82), 2 échecs TinyOL pré-existants hors périmètre.

**S3010 — capture board réelle** (NUCLEO-F439ZI, `/dev/ttyACM0`, CWRU, 300 samples, `--update`) via
`scripts/board_pair_recorder.py` (latences séparées + combinée + `.bss` + métriques en ligne) :

| Paire | Maha seul | Supervisé seul | **Combinée** | Overhead | AUROC Maha | F1 sup | Gap 2 |
|-------|:---------:|:--------------:|:------------:|:--------:|:----------:|:------:|:-----:|
| `exp_S30_board_maha_ewc` | 5 µs | 251 µs | **256 µs** | ~0 µs | 1.000 | 0.471 | ✅ << 100 ms |
| `exp_S30_board_maha_hdc` | 5 µs | 647 µs | **651 µs** | ~0 µs | 0.998 | 0.473 | ✅ << 100 ms |

`.bss` total = **104 576 B** (39.9 % de 256 Ko). Latence combinée ≈ somme des deux modèles
(overhead négligeable, confirme la co-exécution séquentielle propre du DUAL_MODE généralisé).
