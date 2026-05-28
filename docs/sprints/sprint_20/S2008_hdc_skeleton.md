# S2008 — (Optionnel) HDC C skeleton : hypervecteur encode + recherche AM

| Champ | Valeur |
|-------|--------|
| **Sprint** | 20 |
| **Priorité** | 🟢 Nice-to-have |
| **Statut** | ✅ Terminé |
| **Durée estimée** | 4h |
| **Dépendances** | S2004 (Unity tests infrastructure) |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/hdc.c`, `firmware/stm32f4_blink/inc/hdc.h` |
| **Référence** | `docs/models/hdc_spec.md`, `Benatti2019HDC`, `src/models/hdc/hdc_classifier.py` |

---

## Contexte

HDC (Hyperdimensional Computing) — M3 — n'a pas encore de code firmware.
C'est un modèle non-neuronal avec une empreinte RAM très faible (potentiellement < 1 Ko), ce qui en fait un candidat idéal pour démontrer Gap 2 de manière extrême.

Ce skeleton pose les bases sans objectif d'entraînement incrémental dans ce sprint.

---

## Ce qu'il faut implémenter

### Architecture cible (depuis `hdc_spec.md`)

- Hypervecteur dimension D = 1000 (configurable via `configs/board_hdc.yaml`)
- Encodage : binding spatiale + permutation temporelle (ou random projection FP32)
- Mémoire associative (AM) : K prototypes de classes, distance cosinus

### Structures et fonctions minimales

```c
/* inc/hdc.h */
#define HDC_DIM 1000      /* depuis configs/board_hdc.yaml */
#define HDC_N_CLASSES 2

typedef struct {
    float am[HDC_N_CLASSES][HDC_DIM];  /* MEM: 2*1000*4 = 8 Ko Flash ou SRAM */
    float proj[HDC_DIM][5];            /* MEM: 1000*5*4 = 20 Ko Flash (random proj) */
    int   n_trained;
} HDCClassifier;

void  hdc_init(HDCClassifier *h);
void  hdc_encode(const HDCClassifier *h, const float *x, float *hv_out);
int   hdc_predict(const HDCClassifier *h, const float *hv);
void  hdc_update(HDCClassifier *h, const float *hv, int label);
```

### Budget RAM estimé

| Composant | RAM |
|-----------|-----|
| `am[2][1000]` (Flash const ou .bss) | 8 Ko |
| `proj[1000][5]` (Flash const) | 20 Ko Flash |
| `hv_out` (stack temporaire) | 4 Ko stack |
| **Total SRAM** | **~4 Ko** |

> Note : si `proj` en Flash, SRAM = 8 Ko + 4 Ko stack = **12 Ko** — encore sous 64 Ko.

---

## Vérification

- [x] Compilation sans warnings (`make all` — ARM Cortex-M4, 0 warnings)
- [x] Test Unity `test_hdc_encode_norm` : `|hv_out|² ≈ D` (propriété hypervecteur) — PASS
- [x] Test Unity `test_hdc_predict_label` : classe correcte sur mock_data synthétique — PASS
- [x] RAM firmware total : 16 Ko / 192 Ko (8.21%) — HDC struct non instanciée globalement

## Résultats

```text
make test : 40 Tests 0 Failures 0 Ignored — OK
make all  : RAM 16136 B / 192 Ko (8.21%), FLASH 24504 B / 2 Mo (1.17%)
```

Fichiers créés :

- `configs/board_hdc.yaml`
- `firmware/stm32f4_blink/inc/hdc.h`
- `firmware/stm32f4_blink/src/hdc.c`
- `firmware/stm32f4_blink/tests/test_hdc.c`

---

## Questions ouvertes

- `TODO(arnaud)` : Dimension D=1000 (hdc_spec) ou D=512 pour rester sous 64 Ko avec les 3 autres modèles actifs ?
- `TODO(dorra)` : `proj[]` en Flash (const) ou régénéré depuis seed ? Impact NeuralART Turbo ?
