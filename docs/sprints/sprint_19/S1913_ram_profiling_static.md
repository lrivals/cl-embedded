# S1913 — RAM profiling statique : `-Wl,-Map` + parser map file → valide < 64 Ko

| Champ | Valeur |
|-------|--------|
| **Sprint** | 19 |
| **Priorité** | 🟢 Nice-to-have |
| **Statut** | ✅ Script existant — procédure à documenter |
| **Durée estimée** | 2h |
| **Dépendances** | S1902, S1903 (modèles compilés) |
| **Fichiers cibles** | `scripts/parse_map_file.py`, `firmware/stm32f4_blink/Makefile` |

---

## Contexte

La contrainte RAM 64 Ko (STM32N6) est le **Gap 2** du projet — jamais formellement validé dans la littérature pour ce type de modèles. Pour combler ce gap, il faut des mesures précises, reproductibles, et tracées dans les expériences.

Le profiling statique via le fichier `.map` linker donne la RAM statique (`.bss` + `.data`) par symbole, sans nécessiter de board. C'est une borne inférieure de la RAM totale (la RAM dynamique stack/heap s'ajoute à l'exécution).

---

## Objectif

Générer un fichier `.map` via le linker ARM GCC, le parser avec `parse_map_file.py`, et valider que le total `.bss` + `.data` de tous les modèles reste sous 64 Ko.

---

## État actuel

**`scripts/parse_map_file.py`** — existe (créé Sprint 18/19). Parse les sections `.bss` et `.data` d'un fichier `.map` GCC et retourne un breakdown par symbole avec totaux.

---

## Procédure

### 1. Générer le fichier `.map`

#### Option A — Via Makefile existant

Vérifier si `firmware/stm32f4_blink/Makefile` ou `CMakeLists.txt` a déjà un flag `-Map` :

```bash
grep -r "Map" firmware/stm32f4_blink/
```

Si absent, ajouter dans `Makefile` ou `CMakeLists.txt` :

```makefile
# Dans Makefile (variable LDFLAGS)
LDFLAGS += -Wl,-Map=build/firmware.map,--cref
```

```cmake
# Dans CMakeLists.txt
target_link_options(firmware PRIVATE -Wl,-Map=build/firmware.map,--cref)
```

#### Option B — EXTRA_LDFLAGS à la compilation

```bash
make -C firmware/stm32f4_blink/ \
    EXTRA_LDFLAGS="-Wl,-Map=firmware/stm32f4_blink/build/firmware.map,--cref" \
    all
```

### 2. Parser le fichier `.map`

```bash
python scripts/parse_map_file.py \
    firmware/stm32f4_blink/build/firmware.map \
    --verbose \
    --budget 65536
```

Sortie attendue :
```
=== RAM Profiling — firmware.map ===
Section .bss:
  g_detector       (mahalanobis.o)   :   200 B
  g_ewc_head       (ewc_head.o)      :  9728 B
  g_tinyol_enc     (tinyol.o)        :  2816 B
  g_tinyol_dec     (tinyol.o)        :  2816 B
  g_acc            (pipeline.o)      :     8 B
  g_auroc          (pipeline.o)      :   258 B
  g_fgt            (pipeline.o)      :    36 B
  [stack HAL...]                     :  4096 B

Section .data:
  [initialized vars]                 :    ~0 B

TOTAL RAM statique (.bss + .data)    : ~20 Ko
Budget STM32N6                       : 64 Ko
MARGE                                : ~44 Ko ✅
```

### 3. Valider la contrainte

`parse_map_file.py` doit retourner exit code 1 si le total dépasse `--budget` :

```python
if total_ram > budget:
    sys.exit(1)  # CI échoue
```

### 4. Intégrer dans CI GitHub Actions

Dans `.github/workflows/firmware.yml`, ajouter une étape après la compilation :

```yaml
- name: RAM profiling statique
  run: |
    make -C firmware/stm32f4_blink/ \
        EXTRA_LDFLAGS="-Wl,-Map=firmware/stm32f4_blink/build/firmware.map" \
        all
    python scripts/parse_map_file.py \
        firmware/stm32f4_blink/build/firmware.map \
        --budget 65536
```

---

## Interprétation des sections

| Section | Type | Contenu |
|---------|------|---------|
| `.bss` | RAM statique, initialisée à 0 au boot | Structs globales (détecteurs, métriques) |
| `.data` | RAM statique, initialisée avec valeurs | Variables globales initialisées |
| `.rodata` | Flash (read-only) | Tableaux `const` : poids, ZSCORE_MEAN/STD |
| `.text` | Flash (code) | Instructions machine |
| **Stack** | RAM dynamique (non visible dans .map) | Variables locales — mesurer via DWT |

> **Important** : `.rodata` (poids Flash) ne compte **pas** dans la RAM. Les tableaux `const float TINYOL_W_ENC1[]` vont en Flash, pas en SRAM.

---

## Budget RAM récapitulatif attendu (3 modèles simultanés)

| Symbole | Section | Taille |
|---------|---------|--------|
| `g_detector` (Mahalanobis) | .bss | ~200 B |
| `g_ewc_head` (EWC) | .bss | ~9 728 B |
| `g_tinyol_enc` + `g_tinyol_dec` (TinyOL) | .bss | ~5 600 B |
| `g_acc` + `g_auroc` + `g_fgt` | .bss | ~302 B |
| HAL + System | .bss/.data | ~8–15 Ko |
| **Total RAM statique** | | **~20–25 Ko** |
| **Marge / 64 Ko** | | **~39–44 Ko ✅** |

> Le stack (~4 Ko typique) est en plus — non visible dans `.map`. La somme totale reste largement sous 64 Ko.

---

## Fichiers cibles

| Fichier | Action |
|---------|--------|
| `scripts/parse_map_file.py` | Vérifier exit code 1 si budget dépassé |
| `firmware/stm32f4_blink/Makefile` | Ajouter `-Wl,-Map=...` si absent |
| `firmware/stm32f4_blink/CMakeLists.txt` | Alternative CMake |
| `.github/workflows/firmware.yml` | Ajouter étape profiling |

---

## Vérification

- [ ] `make -C firmware/stm32f4_blink/ all` génère `build/firmware.map`
- [ ] `python scripts/parse_map_file.py build/firmware.map --budget 65536` → exit 0
- [ ] Total `.bss` + `.data` < 25 Ko (hors HAL)
- [ ] CI GitHub Actions intègre l'étape de profiling

---

## Questions ouvertes

- `FIXME(gap2)` : Ce profiling (NUCLEO, ARM GCC Cortex-M4) est **indicatif**. Le budget réel STM32N6 dépend du compilateur STM32CubeIDE, des options d'optimisation (`-O2` vs `-Os`), et des sections HAL N6 (plus lourdes). Une re-mesure sur toolchain STM32N6 est requise pour valider formellement Gap 2.
- `TODO(dorra)` : Y a-t-il un outil STM32 officiel pour profiler la RAM statique sur STM32N6 (STM32CubeMonitor-Power, STM32CubeIDE Map file viewer) ?
