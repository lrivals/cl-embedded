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
   remplit toute la zone libre `[_ebss, _estack)` avec la sentinelle
   `0xDEADBEEF` (`STACK_PAINT_SENTINEL`) juste avant `bl main`. À cet instant
   `SP = _estack` et aucune trame n'est empilée : toute la zone est peignable.
   La peinture n'utilise que des registres et **ne modifie ni `.bss` ni `.data`**
   (vérifié : `.bss`/`.data` inchangés, seul `.text` +16 B).
2. **Exécution** d'une charge représentative (stream multi-modèle, pire cas HDC).
3. **Scan** — le plus bas mot écrasé donne la profondeur atteinte :
   `profiling_stack_peak_bytes()` =
   `_estack - première_adresse_non_sentinelle`
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
dur), halt le cœur, scanne la sentinelle, et imprime `.data`, `.bss`, **pic de
pile** et **`ram_peak_total`** en % de 256 Ko.

**Caveat** : si un mot de pile vaut exactement `0xDEADBEEF`, le pic est
sous-estimé. La sentinelle est choisie pour rendre l'événement négligeable.

## 4. Lien Gap 2

Gap 2 (« < 100 Ko RAM avec chiffres mesurés ») doit s'appuyer sur `ram_peak`
(pile incluse), pas seulement `.bss`. Le `.bss` reste une borne inférieure utile ;
le pic mesuré par la procédure ci-dessus est le chiffre à reporter quand on veut
être **sûr** du résultat. À remplir depuis une exécution réelle du script (pas de
valeur inventée).
