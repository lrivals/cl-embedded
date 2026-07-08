# Mesurer la RAM « pour de vrai » sur NUCLEO-F439ZI

> **Document de présentation** — comment on garantit que le chiffre de RAM annoncé
> (Gap 2) est correct, contigu, et qu'il inclut bien *tout* ce qui consomme de la
> SRAM. Référence technique détaillée : [`context/ram_measurement.md`](context/ram_measurement.md).

---

## 1. La question de départ

> « On rapporte la RAM comme `.bss = _ebss − _sbss`.
> Comment être sûr que l'espace entre ces deux adresses est **contigu**, sans autre
> chose au milieu ? Et si on ne peut pas le vérifier, comment mesurer la RAM
> autrement, d'une façon dont on est **sûr** du résultat ? »

Deux sous-questions, deux réponses distinctes :

1. **La contiguïté** → garantie par construction *et* vérifiable (§2).
2. **La fiabilité du chiffre** → le vrai piège n'est pas la contiguïté, c'est que
   `.bss` **oublie la pile** (§3). On corrige avec une mesure de pic réel (§4).

---

## 2. « Y a-t-il un trou entre `_sbss` et `_ebss` ? » → Non, par construction

### 2.1 Ce que dit le linker

Le firmware calcule son empreinte au boot :

```c
g_profiling.bss_bytes = (uint16_t)((uintptr_t)&_ebss - (uintptr_t)&_sbss);
```

`_sbss` et `_ebss` ne sont pas deux adresses arbitraires : ce sont les bornes
d'**une seule section de sortie** déclarée dans le linker script
(`STM32F439ZITx_FLASH.ld`) :

```ld
.bss : {
  . = ALIGN(4);
  _sbss = .;          /* ← borne basse */
  *(.bss) *(.bss*) *(COMMON)
  . = ALIGN(4);
  _ebss = .;          /* ← borne haute */
} >RAM
```

**Une section de sortie du linker est, par définition, un bloc d'adresses
contiguës.** Le linker pose les variables les unes après les autres dans cet
intervalle ; il ne peut rien y insérer d'étranger. Donc `_ebss − _sbss` est
*exactement* la taille de `.bss`, sans trou ni intrus.

### 2.2 Plan mémoire (RAM 192 Ko : `0x20000000` → `0x20030000`)

```
0x20000000  ┌────────────────────┐  _sdata
            │  .data (460 B)     │  globales initialisées
0x200001cc  ├────────────────────┤  _edata == _sbss   ← jointif, 0 trou
            │                    │
            │  .bss (105 036 B)  │  globales à zéro + heap statique 512 B
            │                    │
0x20019c18  ├────────────────────┤  _ebss
            │                    │
            │  zone pile libre   │  ← la PILE vit ici (croît vers le bas ↓)
            │   (91 112 B)       │     ET N'EST PAS DANS .bss !
            │                    │
0x20030000  └────────────────────┘  _estack
```

### 2.3 Comment le *vérifier* (4 sources indépendantes)

Depuis `firmware/stm32f4_blink/`, après `make all` :

```bash
ELF=build/stm32f4_blink.elf
arm-none-eabi-size    $ELF      # 1. colonne bss (lit l'en-tête ELF)
arm-none-eabi-objdump -h $ELF   # 2. .bss VMA+Size ; .data finit pile à _sbss
arm-none-eabi-nm   -n $ELF      # 3. types : tout symbole entre les bornes est 'b'/'B'
python scripts/parse_map_file.py build/stm32f4_blink.map  # 4. le .map
```

Résultat mesuré sur le build courant — **les 4 concordent** :

| Source | `.bss` | `.data` |
|---|---:|---:|
| `arm-none-eabi-size` | 105 036 B | 460 B |
| `objdump -h` (Size) | 0x19a4c = 105 036 B | 0x1cc = 460 B |
| `_ebss − _sbss` / `_edata − _sdata` | 105 036 B | 460 B |

- `.data` se termine en `0x200001cc`, `.bss` commence en `0x200001cc`
  → **jointif, aucun trou, aucun chevauchement**.
- Le seul symbole non-`b/B` dans l'intervalle est le label frontière `_edata`
  (taille nulle, qui coïncide avec `_sbss`). Aucune vraie donnée étrangère.

> ✅ **Conclusion §2** : la contiguïté est garantie ET prouvée. Le chiffre `.bss`
> est exact.

---

## 3. Le vrai piège : `.bss` n'est PAS le pic de RAM

`.bss` ne compte que les **variables globales** mises à zéro. Il **ignore la
pile**, où vivent les gros tableaux locaux. Exemple concret dans le chemin HDC :

```c
float hv[HDC_DIM];   /* HDC_DIM = 1000 → 4 000 octets SUR LA PILE */
```

Ces 4 Ko n'apparaissent **nulle part** dans `.bss`. Donc :

> ⚠️ Le chiffre `.bss` est une **borne inférieure**, pas le pic réel.
> Le pic honnête est : **`ram_peak = .data + .bss + pic_de_pile`**.

C'est ce qu'il faut reporter quand on veut être *sûr* du résultat (Gap 2).

---

## 4. Mesurer le pic de pile : « stack painting » (high-water mark)

Technique standard en embarqué, en 3 temps :

Canary **position-dépendant** : chaque mot est peint avec **sa propre adresse**
(un buffer de pile rempli d'une constante répétée ne peut donc pas masquer une zone).

```
   AU BOOT (startup.s)              APRÈS UNE CHARGE              ON SCANNE
   ┌──────────────┐ _estack        ┌──────────────┐ _estack     ┌──────────────┐
   │ mot = @mot   │                 │  pile        │ ← utilisée  │  pile        │
   │ mot = @mot   │  on peint       │  utilisée    │             │  utilisée    │
   │ mot = @mot   │  chaque mot     │ mot = @mot   │             ├──────────────┤ ← 1er mot
   │ mot = @mot   │  avec sa propre │ mot = @mot   │             │ mot = @mot   │   ≠ @mot
   │ mot = @mot   │  adresse        │ mot = @mot   │             │ mot = @mot   │
   └──────────────┘ _ebss          └──────────────┘ _ebss       └──────────────┘
                                                            pic = _estack − (1er mot ≠ @mot)
```

1. **Peinture au démarrage** — dans `startup_stm32f439xx.s`, juste avant `bl main`
   (à cet instant `SP = _estack`, aucune trame empilée → toute la zone
   `[_ebss, _estack)` est libre) :

   ```asm
   ldr r2, =_ebss
   ldr r3, =_estack
   PaintStack:
     str r2, [r2]        @ canary = adresse (Rt==Rn sans writeback : OK)
     add r2, r2, #4
   LoopPaintStack:
     cmp r2, r3
     bcc PaintStack
   ```

   La boucle n'utilise que des registres → **ne modifie ni `.bss` ni `.data`**
   (vérifié : tailles inchangées, seul `.text` grossit un peu).

2. **Exécution** d'une charge représentative (stream multi-modèle, pire cas HDC).

3. **Scan** — le plus bas mot dont la valeur ≠ son adresse marque la profondeur atteinte :

   ```c
   uint32_t profiling_stack_peak_bytes(void);   // _estack − 1er_mot_écrasé
   uint32_t profiling_ram_peak_bytes(void);     // .data + .bss + pic_de_pile
   ```

   La logique de scan est isolée dans une fonction pure
   `profiling_stack_peak_from_region()` → **testée sur host** (Unity).

### 4.1 Lire le résultat sans toucher au protocole UART

Le flux UART est binaire ; y injecter du texte le casserait (règle anti-`DEBUG_PRINTF`).
On lit donc la mémoire **via OpenOCD** après coup, avec
[`scripts/measure_stack_watermark.py`](../scripts/measure_stack_watermark.py) :

```bash
cd firmware/stm32f4_blink && make flash
# streamer une charge (sensor_stream.py, modes HDC inclus), laisser tourner
openocd -f interface/stlink.cfg -f target/stm32f4x.cfg &   # serveur Tcl RPC 6666
python scripts/measure_stack_watermark.py
```

Le script lit `_sbss/_ebss/_sdata/_edata/_estack` **depuis l'ELF** (zéro valeur en
dur), halt le cœur, scanne la sentinelle, et affiche. **Mesure réelle NUCLEO-F439ZI**
(dataset Monitoring, mode entraînement, `experiments/exp_S39_ram/`) :

```
  .data        :      460 B
  .bss         :  106 152 B   (chiffre habituellement rapporté)
  pic de pile  :    4 712 B   (EWC entraînement — mesuré, high-water mark)
  ─────────────────────────────
  RAM pic total:  111 324 B   (42.5 % de 256 Ko)
```

> **Caveat honnête** : le seul mot-frontière valant par hasard sa propre adresse
> sous-estimerait le pic (~1/2³², négligeable) ; le canary position-dépendant élimine
> le cas d'un buffer rempli d'une constante répétée.
>
> **Nuance mesurée** : le pic de pile est **~identique (~4,3 Ko) pour tous les modèles**
> (EWC 4 712 B, HDC/Maha/TinyOL 4 336 B) car le compilateur réserve **une seule trame pour
> `pipeline_run()` = le max de toutes ses branches** (dont `hv[HDC_DIM]` 4 Ko). Le pic de
> pile est donc une propriété du firmware entier, pas isolable par modèle.

---

## 5. Ce que ça change pour le Gap 2

| Avant | Après |
|---|---|
| RAM = `.bss` seul (borne inférieure) | RAM = `.data + .bss + pic_de_pile` (pic réel) |
| Pile (ex. `hv[HDC_DIM]` 4 Ko) invisible | Pile mesurée et incluse |
| « on suppose » la contiguïté | contiguïté **prouvée** (4 sources croisées) |

Le `.bss` reste un indicateur utile, mais le chiffre **à présenter** quand on veut
être sûr est le pic mesuré. À remplir depuis une exécution réelle du script —
**aucune valeur inventée**.

---

## 6. Où vit quoi, par modèle — et exemple chiffré (Monitoring)

Le notebook [`notebooks/cl_eval/ram_measurement/ram_explained.ipynb`](../notebooks/cl_eval/ram_measurement/ram_explained.ipynb)
détaille, pour chaque modèle, la part **statique** (poids figés, Flash → copie RAM) vs
**modulable** (état de *continual learning* mis à jour à bord, `.bss` in-place), plus la
**pile**. Les tailles `.bss` sont lues réellement de l'ELF ; le split est calculé et
vérifié par [`scripts/ram_breakdown.py`](../scripts/ram_breakdown.py).

Sur Monitoring (5-feat), split validé au niveau octet contre `nm` :

| Modèle | statique (poids) | modulable (état CL) | `.bss` total | pile inf. | pile entr. |
|---|---:|---:|---:|---:|---:|
| EWC | 3 016 B | 5 636 B (Fisher + θ*) | 8 652 B | 200 B | 400 B |
| HDC | 20 000 B (proj) | 8 332 B (mémoire assoc.) | 28 332 B | 4 000 B | 4 300 B |
| Mahalanobis | 108 B (Σ⁻¹) | 20 B (moyenne EMA) | 128 B | 40 B | 40 B |
| TinyOL | 5 716 B (auto-enc. gelé) | 96 B (tête OtO) | 5 812 B | 212 B | 212 B |

**Idée maîtresse** : statique et modulable vivent dans la **même struct pré-allouée en
`.bss`** ; l'entraînement en ligne (Fisher, θ*, mémoire associative, EMA, tête OtO)
réécrit des tableaux **déjà comptés** — il n'ajoute **aucune RAM permanente**, seulement de
la pile transitoire. Donc `.bss` inférence == `.bss` entraînement.

Schémas générés (`docs/figures/ram_measurement/`) :
`fig_memory_map.png`, `fig_stack_painting.png`, `fig_model_maps.png`,
`fig_monitoring_stacked.png`, `fig_infer_vs_train.png`, `fig_shared_frame.png`
(effet « trame partagée `pipeline_run` »).

---

## 7. Récapitulatif des artefacts

| Fichier | Rôle |
|---|---|
| `firmware/.../startup/startup_stm32f439xx.s` | Peinture de la pile au boot |
| `firmware/.../inc/profiling.h`, `src/profiling.c` | `STACK_PAINT_SENTINEL`, getters de pic |
| `firmware/.../tests/test_profiling.c` | 3 tests du scan (PASS) |
| `scripts/measure_stack_watermark.py` | Driver OpenOCD : pic de pile + RAM totale (`--json`) |
| `scripts/ram_breakdown.py` | Split statique/modulable par modèle (vérifié vs nm) |
| `scripts/run_ram_board.py` | Driver carte optionnel : pic réel par modèle Monitoring |
| `notebooks/cl_eval/ram_measurement/ram_explained.ipynb` | Notebook pédagogique + 5 schémas |
| `docs/context/ram_measurement.md` | Référence technique détaillée |
| `docs/presentation_ram_measurement.md` | **Ce document** (présentation) |
