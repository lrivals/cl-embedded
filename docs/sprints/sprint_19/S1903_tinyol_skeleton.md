# S1903 — TinyOL encoder C skeleton : forward pass + poids Flash

| Champ | Valeur |
|-------|--------|
| **Sprint** | 19 |
| **Priorité** | 🔴 Critique |
| **Statut** | ⬜ À faire (skeleton ✅, intégration poids ⬜) |
| **Durée estimée** | 4h |
| **Dépendances** | S1901 (pipeline validé) |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/tinyol.c`, `firmware/stm32f4_blink/inc/tinyol.h`, `firmware/stm32f4_blink/inc/model_weights.h` |
| **Référence** | `Ren2021TinyOL`, `src/models/tinyol/autoencoder.py`, `src/models/tinyol/oto_head.py` |

---

## Contexte

TinyOL est un autoencoder léger conçu pour le continual learning sur MCU. Son architecture — un encodeur qui projette dans un espace embedding puis un décodeur qui reconstruit l'entrée — permet une détection d'anomalies via l'erreur de reconstruction (MSE). Seul le forward pass est porté en C dans ce sprint ; la tête OtO (One-to-One, mise à jour incrémentale) sera portée dans un sprint suivant.

Le skeleton C a été rédigé lors de la mise en place du firmware Sprint 16.

---

## Objectif

Valider que `tinyol_predict()` est fonctionnel avec les poids exportés depuis PyTorch, intégré à `model_weights.h`, et testé via Unity sur `mock_data.h`.

---

## État actuel (code existant)

**`firmware/stm32f4_blink/src/tinyol.c`** — **skeleton complet** :

| Fonction | Signature | État |
|----------|-----------|------|
| `tinyol_encode` | `(enc, x, emb)` | ✅ Linear(5→32)+ReLU → Linear(32→16)+ReLU |
| `tinyol_decode` | `(dec, emb, recon)` | ✅ Linear(16→32)+ReLU → Linear(32→5) |
| `tinyol_reconstruction_error` | `(x, recon, n)` | ✅ MSE = mean((x-recon)²) |
| `tinyol_predict` | `(enc, dec, x, threshold)` | ✅ MSE > threshold → anomalie |

**`firmware/stm32f4_blink/inc/tinyol.h`** — constantes et struct :

```c
#define TINYOL_IN   5   /* features d'entrée */
#define TINYOL_H1  32   /* couche cachée */
#define TINYOL_EMB 16   /* embedding (goulot d'étranglement) */
#define TINYOL_OUT  5   /* reconstruction (= TINYOL_IN) */

typedef struct { float w_enc1[32][5]; float b_enc1[32];
                 float w_enc2[16][32]; float b_enc2[16]; } TinyOLEncoder;
typedef struct { float w_dec1[32][16]; float b_dec1[32];
                 float w_dec2[5][32]; float b_dec2[5]; } TinyOLDecoder;
```

**Budget RAM stack** (local dans `tinyol_predict`) :
- `emb[16]` : 64 B @ FP32
- `recon[5]` : 20 B @ FP32
- Intermédiaires `h1[32]` : 128 B @ FP32

**Budget Flash** (poids statiques) :
- Encodeur : (5×32+32) + (32×16+16) = 160+32+512+16 = 720 floats × 4 = ~2.8 Ko
- Décodeur : (16×32+32) + (32×5+5) = 512+32+160+5 = 709 floats × 4 = ~2.8 Ko
- **Total Flash** : ~5.6 Ko

---

## Ce qui manque / Ce qu'il faut faire

### 1. Exporter les poids PyTorch → `model_weights.h`

Utiliser `scripts/export_weights_c.py` après entraînement TinyOL (exp_XXX Pronostia ou CWRU) :

```bash
python scripts/export_weights_c.py \
    --model tinyol \
    --checkpoint experiments/exp_XXX/tinyol_checkpoint.pt \
    --output firmware/stm32f4_blink/inc/model_weights.h \
    --append   # ajoute aux poids Mahalanobis déjà présents
```

Vérifier que `model_weights.h` contient :
```c
/* TinyOL encoder weights — MEM: 2.8 Ko @ FP32 en Flash */
static const float TINYOL_W_ENC1[TINYOL_H1][TINYOL_IN] = { ... };
static const float TINYOL_B_ENC1[TINYOL_H1] = { ... };
static const float TINYOL_W_ENC2[TINYOL_EMB][TINYOL_H1] = { ... };
static const float TINYOL_B_ENC2[TINYOL_EMB] = { ... };
/* TinyOL decoder weights — MEM: 2.8 Ko @ FP32 en Flash */
static const float TINYOL_W_DEC1[TINYOL_H1][TINYOL_EMB] = { ... };
static const float TINYOL_B_DEC1[TINYOL_H1] = { ... };
static const float TINYOL_W_DEC2[TINYOL_OUT][TINYOL_H1] = { ... };
static const float TINYOL_B_DEC2[TINYOL_OUT] = { ... };
static const float TINYOL_THRESHOLD = 0.05f;  /* depuis board_tinyol.yaml */
```

### 2. Initialiser TinyOLEncoder/Decoder depuis Flash dans `pipeline.c`

Dans `pipeline_init()`, ajouter :
```c
/* Charger poids Flash → structs TinyOL (copie ou pointeur const) */
memcpy(g_tinyol_enc.w_enc1, TINYOL_W_ENC1, sizeof(g_tinyol_enc.w_enc1));
/* ... idem pour chaque tableau ... */
```

> Alternative plus efficace : déclarer les structs directement `const` en Flash si le compilateur le permet (`__attribute__((section(".rodata")))`).

### 3. Périmètre de ce sprint

La **tête OtO** (mise à jour incrémentale des prototypes embeddings) est **hors périmètre Sprint 19**. Seul le forward pass en inférence est validé ici — ce qui simule le comportement NPU (NeuralART Turbo accepte le graphe figé).

---

## Fichiers cibles

| Fichier | Action |
|---------|--------|
| `firmware/stm32f4_blink/src/tinyol.c` | Lecture seule — skeleton validé |
| `firmware/stm32f4_blink/inc/tinyol.h` | Lecture seule — struct + constantes OK |
| `firmware/stm32f4_blink/inc/model_weights.h` | Ajouter poids TinyOL exportés |
| `firmware/stm32f4_blink/src/pipeline.c` | Ajouter init TinyOL |
| `scripts/export_weights_c.py` | Utiliser tel quel (Sprint 16) |
| `configs/board_tinyol.yaml` | Lire threshold à injecter dans model_weights.h |

---

## Budget RAM (indicatif NUCLEO-F439ZI)

| Composant | RAM |
|-----------|-----|
| `TinyOLEncoder` struct (.bss) | ~2.8 Ko @ FP32 |
| `TinyOLDecoder` struct (.bss) | ~2.8 Ko @ FP32 |
| Stack `tinyol_predict` | ~212 B (emb+recon+h1) |
| **Total** | **~5.8 Ko / 192 Ko SRAM** |

Alternative Flash : poids `const` en Flash → RAM structurelle = 0, uniquement le stack.

`FIXME(gap2)` : à valider sur STM32N6 (marge 57 Ko sur 64 Ko budget projet).

---

## Vérification

- [ ] `make -C firmware/stm32f4_blink/ all` — compilation sans warning
- [ ] Tests Unity `test_models.c` :
  - `tinyol_reconstruction_error` sur `MOCK_NORMAL_T0[0]` avec poids zéro → `≈ 0.007` (cf. `MOCK_TINYOL_RECON_ERR_ZERO_WEIGHTS`)
  - `tinyol_predict` avec poids exportés PyTorch → prédit 0 sur samples normaux, 1 sur anomalies (tolérance > 80% accuracy)
- [ ] Aucun `malloc` dans le binaire compilé : `nm firmware.elf | grep malloc` → vide

---

## Questions ouvertes

- `TODO(dorra)` : Le NeuralART Turbo (NPU STM32N6) accepte-t-il le graphe TinyOL encoder comme ONNX opset 17, ou faut-il passer par le format propriétaire `.nef` ? Si `.nef`, la simulation Cortex-M55 SW reste le plan de secours.
- `TODO(arnaud)` : Faut-il implémenter un seuil adaptatif (EMA sur les erreurs normales) ou un seuil fixe depuis `board_tinyol.yaml` est suffisant pour ce sprint ?
