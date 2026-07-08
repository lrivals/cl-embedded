# Mesure de la RAM sur NUCLEO-F439ZI — contiguïté `.bss` et pic réel

> Source de vérité pour répondre à : « le chiffre `.bss` est-il sûr ? l'espace
> `[_sbss, _ebss]` est-il contigu ? quelle est la vraie RAM consommée ? »

## 1. La zone `[_sbss, _ebss]` est contiguë par construction

Le firmware calcule son empreinte RAM comme `_ebss - _sbss` (au boot dans
[`profiling.c`](../../firmware/stm32f4_blink/src/profiling.c), exposé par
[`hw_info.c`](../../firmware/stm32f4_blink/src/hw_info.c) sous `ram_used = .data + .bss`).

Le doute « y a-t-il autre chose entre `_sbss` et `_ebss` ? » n'a pas lieu d'être :
dans le linker script
[`STM32F439ZITx_FLASH.ld`](../../firmware/stm32f4_blink/linker/STM32F439ZITx_FLASH.ld),
`_sbss` et `_ebss` encadrent **une seule section de sortie** :

```ld
.bss : {
  . = ALIGN(4);
  _sbss = .;
  *(.bss) *(.bss*) *(COMMON)
  . = ALIGN(4);
  _ebss = .;
} >RAM
```

Une section de sortie du linker est, par définition, un **bloc d'adresses
contiguës**. Aucune autre section ne peut s'y intercaler. `_ebss - _sbss` est donc
**exactement** la taille `.bss`, sans trou ni intrus.

### Vérification croisée (read-only, reproductible)

Depuis `firmware/stm32f4_blink/` après `make all`. Les quatre sources doivent
donner la **même** taille `.bss` :

```bash
ELF=build/stm32f4_blink.elf

# 1. Section headers de l'ELF (outil indépendant)
arm-none-eabi-size    $ELF        # colonne bss
arm-none-eabi-objdump -h $ELF     # .bss : VMA + Size ; .data finit pile à _sbss

# 2. Aucun symbole non-BSS entre les bornes (hors labels frontière _edata/_sbss)
arm-none-eabi-nm -n $ELF | grep -iE ' [bB] | _sbss| _ebss'

# 3. Le .map (déjà parsé par scripts/parse_map_file.py)
python ../../scripts/parse_map_file.py build/stm32f4_blink.map

# 4. Cohérence runtime : le boot imprime "BSS+data" (hw_info_print) == size/map
```

Résultat mesuré (build courant) — les quatre concordent :

| Source | `.bss` | `.data` |
|---|---|---|
| `arm-none-eabi-size` | 105 036 B | 460 B |
| `objdump -h` (Size) | 0x19a4c = 105 036 B | 0x1cc = 460 B |
| `_ebss - _sbss` / `_edata - _sdata` | 105 036 B | 460 B |

`.data` se termine en `0x200001cc`, `.bss` commence en `0x200001cc` → **0 trou,
0 chevauchement**. Le seul symbole non-`b/B` dans l'intervalle est le label
frontière `_edata` (taille nulle, coïncide avec `_sbss`). Contiguïté prouvée.

## 2. `.bss` **n'est pas** le pic de RAM : la pile est exclue

Le vrai risque de soundness n'est pas la contiguïté mais le **périmètre** du
chiffre. `.bss` ne compte que les globales zéro-initialisées. Il **exclut la
pile**, où vivent les gros buffers locaux — p.ex. `float hv[HDC_DIM]` = **4 Ko**
([`hdc.c`](../../firmware/stm32f4_blink/src/hdc.c),
[`pipeline.c`](../../firmware/stm32f4_blink/src/pipeline.c)). Le pic de RAM réel :

```
ram_peak = .data + .bss + pic_de_pile
```

Layout RAM (192 Ko, `0x20000000`–`0x20030000`) : `.data` → `.bss` → (heap 512 B,
dans `.bss`) → **zone pile libre `[_ebss, _estack)`** où la pile croît vers le bas
depuis `_estack`. Aucune section CCM (linker `RAM` seul).

## 3. Mesure du pic de pile (stack high-water mark)

Méthode « stack painting » :

1. **Peinture au boot** — [`startup_stm32f439xx.s`](../../firmware/stm32f4_blink/startup/startup_stm32f439xx.s)
   peint toute la zone libre `[_ebss, _estack)` avec un **canary position-dépendant**
   (chaque mot reçoit **sa propre adresse**, `str r2,[r2]`) juste avant `bl main`.
   À cet instant `SP = _estack` et aucune trame n'est empilée : toute la zone est
   peignable. La peinture n'utilise que des registres et **ne modifie ni `.bss` ni
   `.data`** (vérifié : `.bss`/`.data` inchangés, seul `.text` grossit un peu).
2. **Exécution** d'une charge représentative (stream multi-modèle, pire cas HDC).
3. **Scan** — le plus bas mot dont la valeur ≠ son adresse donne la profondeur :
   `profiling_stack_peak_bytes()` = `_estack - première_adresse_non_intacte`
   (logique pure `profiling_stack_peak_from_region()`, testée host).
   `profiling_ram_peak_bytes()` renvoie `.data + .bss + pic_de_pile`.

**Restitution sans toucher au protocole UART** (cf. règle anti-DEBUG_PRINTF) :
lecture mémoire via OpenOCD avec
[`scripts/measure_stack_watermark.py`](../../scripts/measure_stack_watermark.py) :

```bash
# 1) make flash ; 2) streamer une charge (sensor_stream.py, modes HDC inclus)
openocd -f interface/stlink.cfg -f target/stm32f4x.cfg &   # serveur Tcl RPC 6666
python scripts/measure_stack_watermark.py                  # halt → scan → pic
```

Le script lit `_sbss/_ebss/_sdata/_edata/_estack` depuis l'ELF (aucune valeur en
dur), halt le cœur, scanne le canary (mot ≠ son adresse), et imprime `.data`,
`.bss`, **pic de pile** et **`ram_peak_total`** en % de 256 Ko.

**Caveat** : le seul mot-frontière valant par hasard sa propre adresse
sous-estimerait le pic (~1/2³², négligeable). Le canary position-dépendant élimine
en plus le cas d'un buffer de pile rempli d'une constante répétée.

### Contre-vérification : borne supérieure statique (`-fstack-usage`)

Le pic *mesuré* dépend de la charge streamée : il ne prouve pas à lui seul qu'aucun
chemin d'appel plus profond n'existe. On le confronte donc à une borne **statique**,
calculée sans exécuter la carte, à partir des cadres de pile par fonction émis par
GCC. Le Makefile compile avec **`-fstack-usage`** (fichier annexe `build/<unité>.su`,
**0 impact sur le code généré** — `.text`/`.data`/`.bss` bit-identiques, vérifié par
`size` avant/après).

[`scripts/stack_usage_report.py`](../../scripts/stack_usage_report.py) parse les `.su`
et somme une **chaîne pire-cas déclarée** (auditable) le long de la boucle chaude :

```
main (48 B) → pipeline_run (4 160 B, détient hv[HDC_DIM] = 4 Ko)
            → noyau modèle le plus profond (ewc_mc_sgd_step 560 B)
            = BORNE STATIQUE 4 768 B
```

Le firmware **n'a pas de récursion** et son dispatch modèle est fait par
`switch`/`if` (appels **directs**, pas de pointeurs de fonction) : le graphe d'appels
de la boucle chaude est court et connu → la chaîne déclarée est fiable (c'est une
**contre-vérification**, pas une preuve formelle). Tout gros cadre hors chaîne (p.ex.
`hdc_retrain`, 4 344 B — **éliminé du binaire par `--gc-sections`**, jamais appelé)
est signalé pour ré-audit s'il redevenait atteignable.

```bash
cd firmware/stm32f4_blink && make clean && make all   # génère les .su
python ../../scripts/stack_usage_report.py \
    --measured ../../experiments/exp_S39_ram/ram_ewc.json
# → BORNE STATIQUE 4 768 B ≥ pic mesuré 4 712 B  (marge +56 B) ✅
```

L'inégalité `borne_statique ≥ pic_mesuré` tenant, le pic mesuré n'a pas « raté » de
chemin plus profond : les deux méthodes se corroborent.

## 4. Lien Gap 2

Gap 2 (« < 100 Ko RAM avec chiffres mesurés ») doit s'appuyer sur `ram_peak`
(pile incluse), pas seulement `.bss`. Le `.bss` reste une borne inférieure utile ;
le pic mesuré par la procédure ci-dessus est le chiffre à reporter quand on veut
être **sûr** du résultat.

**Mesure réelle NUCLEO-F439ZI** (dataset Monitoring, entraînement, `experiments/exp_S39_ram/`) :
`.data=460 B` + `.bss=106 152 B` + pic de pile mesuré `4 712 B` (EWC) =
**`ram_peak ≈ 111 324 B` (42,5 % de 256 Ko)**. Le pic de pile est **~identique
(~4,3 Ko) pour les 4 modèles** (EWC 4 712, HDC/Maha/TinyOL 4 336) car le compilateur
réserve **une seule trame pour `pipeline_run()` = le max de ses branches** (dont
`hv[HDC_DIM]` 4 Ko) : le pic de pile est une propriété du firmware entier, non isolable
par modèle. La **borne statique** (§3) le confirme : `4 768 B ≥ 4 712 B` (marge +56 B).
Détail et schémas :
[`notebooks/cl_eval/ram_measurement/ram_explained.ipynb`](../../notebooks/cl_eval/ram_measurement/ram_explained.ipynb).

## 5. Cas à re-mesurer (comparaisons utiles)

Le `.bss` (via `size`) est **exact** et ne demande pas de re-mesure. Ce qui vaut le coup,
c'est le **pic de pile** sur les cas qui grossissent réellement les cadres — pour tester la
thèse « trame partagée » sous stress et confirmer que la borne statique reste ≥ pic :

| Cas | Intérêt | Comment |
|---|---|---|
| **Défaut 5-feat Monitoring** | référence (déjà mesuré) + attacher la borne | `make all && python scripts/run_ram_board.py` |
| **Condition `all` CMAPSS k=21** (S35) | cadres plus gros (dims × 4) → la trame partagée `hv[HDC_DIM]` reste-t-elle dominante ? | `make clean && make all EWC_IN=21 MAHA_DIM=21 TINYOL_IN=21 HDC_N_FEATURES=21 PROTO_MAX_N=21` puis flash + mesure |
| **INT8 board** (S36) | `.bss` poids ÷4 ; la pile change-t-elle ? (`ewc_int8_update` 496 B) | même firmware, streamer avec le flag `FRAME_FLAGS_INT8_MODE` (0x40) |

Procédure d'un cas variant (sortie JSON dédiée, à comparer au cas 5-feat) :

```bash
cd firmware/stm32f4_blink && make clean && make all <overrides>   # + size : vérifier .bss
make flash
# streamer une charge pire-cas (mode HDC + --update), p.ex. via sensor_stream.py
openocd -f interface/stlink.cfg -f target/stm32f4x.cfg &
python ../../scripts/measure_stack_watermark.py \
    --json ../../experiments/exp_S39_ram/ram_<cas>.json --label <cas>
python ../../scripts/stack_usage_report.py \
    --measured ../../experiments/exp_S39_ram/ram_<cas>.json   # borne recalculée sur ce build
```

**Mesuré (NUCLEO-F439ZI réelle, `experiments/exp_S39_ram/`)** — le pic reste ~4,3–4,7 Ko,
dominé par `hv[HDC_DIM]` (4 Ko fixe), **peu sensible aux dims** → confirme la thèse « trame
partagée » et l'argument Gap 2. La borne statique est recalculée depuis les `.su` du build
courant et l'invariant `borne ≥ pic` **tient sur chaque cas** :

| Cas | `.bss` | pic pile mesuré | borne statique | marge |
|---|---|---|---|---|
| 5-feat — EWC | 105 036 B | **4 712 B** | 4 768 B | +56 B ✅ |
| 5-feat — HDC / Maha / TinyOL | 105 036 B | **4 336 B** | 4 768 B | +432 B ✅ |
| INT8 — EWC-INT8 (flag 0x40) | 105 036 B | **4 720 B** | 4 768 B | +48 B ✅ |
| all CMAPSS k=21 — HDC | 184 864 B | **4 392 B** | 4 800 B | +408 B ✅ |
| all CMAPSS k=21 — EWC | 184 864 B | **4 728 B** | 4 800 B | +72 B ✅ |

Constat : passer de k=5 à k=21 fait **exploser `.bss`** (105 → 185 Ko, `proj`/poids/Fisher)
mais laisse le **pic de pile quasi inchangé** (+16…+56 B = buffer `raw[PROTO_MAX_N]` + cadres
EWC légèrement plus gros) ; la borne statique suit (+32 B). INT8 ne réduit pas la pile
(`ewc_int8_update` déquantifie en FP32 sur le FPU, cadre ≈ FP32). RAM pic total k=21 ≈ 190 Ko
(72,4 % de 256 Ko) — toujours sous budget.
