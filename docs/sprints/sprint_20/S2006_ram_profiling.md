# S2006 — RAM profiling 3 modèles simultanés → tableau Gap 2 formel

| Champ | Valeur |
|-------|--------|
| **Sprint** | 20 |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Terminé |
| **Durée estimée** | 3h |
| **Dépendances** | S2001, S2003 (tous modèles compilés ensemble) |
| **Fichiers cibles** | `scripts/parse_map_file.py`, `experiments/exp_S19_02/gap2_table.json` |
| **Référence** | `docs/sprints/sprint_19/S1913_ram_profiling_static.md` |

---

## Contexte

`parse_map_file.py` existe (Sprint 19, S1913). Il faut l'exécuter sur un binaire qui active les 3 modèles simultanément (Mahalanobis + EWC + TinyOL + métriques) et produire le tableau Gap 2 exploitable pour le manuscrit.

---

## Étapes

### 1. Compiler avec les 3 modèles actifs

Vérifier que `main.c` et `pipeline.c` instancient les 3 structs en globale :

```c
static MahalanobisDetector g_maha;   /* ~200 B */
static EWCHead             g_ewc;    /* ~9.5 Ko */
static TinyOLEncoder       g_enc;    /* ~40 B SRAM + ~5.6 Ko Flash */
static OnlineAccuracy      g_acc;    /* 8 B */
static OnlineAUROC         g_auroc;  /* 258 B */
static ForgettingTracker   g_fgt;    /* 36 B */
static ProfilingState      g_prof;   /* ~20 B */
```

```bash
make -C firmware/stm32f4_blink/ all CFLAGS="-Wl,-Map=build/firmware.map"
```

### 2. Exécuter `parse_map_file.py`

```bash
python scripts/parse_map_file.py \
    firmware/stm32f4_blink/build/firmware.map \
    --budget 65536 \
    --output experiments/exp_S19_02/gap2_table.json
```

### 3. Format de sortie attendu

```json
{
  "platform": "nucleo_f439zi",
  "total_bss_bytes": 10062,
  "total_data_bytes": 0,
  "total_flash_bytes": 18432,
  "gap2_budget_bytes": 65536,
  "gap2_margin_bytes": 55474,
  "gap2_compliant": true,
  "breakdown": {
    "g_maha (.bss)":  200,
    "g_ewc (.bss)":   9500,
    "g_enc (.bss)":   40,
    "g_acc (.bss)":   8,
    "g_auroc (.bss)": 258,
    "g_fgt (.bss)":   36,
    "g_prof (.bss)":  20,
    "TINYOL_ENC_W1 (.rodata)": 5632
  },
  "fixme": "NUCLEO-F439ZI indicatif (192 Ko SRAM). Validation formelle STM32N6 (64 Ko) bloquée."
}
```

---

## Tableau manuscrit (à inclure dans chapitre 4)

| Modèle | SRAM .bss | Flash .rodata | Total SRAM | Marge / 64 Ko |
|--------|:---------:|:-------------:|:----------:|:-------------:|
| Mahalanobis | 200 B | — | **200 B** | ✅ 63.8 Ko |
| EWC head | 9 500 B | — | **9.5 Ko** | ✅ 54.5 Ko |
| TinyOL encoder | 40 B | 5 632 B | **40 B SRAM** | ✅ 64.0 Ko |
| Métriques | 302 B | — | **302 B** | ✅ 63.7 Ko |
| **Tous simultanés** | **~10.1 Ko** | **5.6 Ko** | **~10.1 Ko** | **✅ ~54 Ko free** |

> `FIXME(gap2)` : mesure NUCLEO-F439ZI (192 Ko). STM32N6 (64 Ko Cortex-M55) requis pour validation formelle.

---

## Vérification

- [ ] `parse_map_file.py` s'exécute sans erreur sur le .map 3 modèles
- [ ] `gap2_compliant: true` dans le JSON de sortie
- [ ] Tableau Markdown généré lisible dans le rapport
- [ ] `FIXME(gap2)` mentionné dans le JSON pour traçabilité

---

## Questions ouvertes

- `FIXME(gap2)` : Validation formelle STM32N6 bloquée — à noter explicitement dans le manuscrit
- `TODO(arnaud)` : Inclure la stack peak estimée (~1 Ko) dans le tableau, ou SRAM statique seul ?
