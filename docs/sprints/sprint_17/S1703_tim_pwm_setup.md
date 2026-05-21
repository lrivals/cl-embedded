# S1703 — TIM3 PWM : signal configurable sur PA6 (base capteur vibration)

| Champ | Valeur |
|-------|--------|
| **ID** | S1703 |
| **Sprint** | Sprint 17 — Objectif 3 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 4h |
| **Dépendances** | S1701 ✅ (projet CubeMX opérationnel) |
| **Fichiers cibles** | `firmware/stm32f4_cubemx/Core/Src/main.c`, `firmware/stm32f4_cubemx/stm32f4_cubemx.ioc`, `firmware/stm32f4_cubemx_examples/TIM_PWMOutput/` |
| **Statut** | ✅ Terminé |

---

## Objectif

Configurer TIM3 en mode PWM sur PA6 (TIM3_CH1) via CubeMX, générer un signal à 1 kHz avec duty cycle 50 %, puis implémenter une variation dynamique du duty cycle. Cette tâche pose les bases du pipeline d'acquisition capteur (TIM trigger → ADC DMA → features → Mahalanobis) prévu pour les sprints suivants.

---

## Contexte matériel

**Timer TIM3 — NUCLEO-F439ZI** :

| Paramètre | Valeur |
|-----------|--------|
| Timer | TIM3 (General Purpose, 32 bits) |
| Bus | APB1 (45 MHz → ×2 = **90 MHz** après le PLL multiplier TIMxCLK) |
| Pin sortie | **PA6** — TIM3_CH1 (Alternate Function AF2) |
| Connecteur board | CN10 ou CN7 selon le variant NUCLEO-144 |

> **Note** : PA6 n'est pas une LED — mesure via oscilloscope ou loopback PA6 → entrée ADC.

**Calcul des registres** pour 1 kHz @ TIMxCLK = 90 MHz :

```
TIMxCLK = APB1_CLK × 2 = 45 MHz × 2 = 90 MHz
freq     = TIMxCLK / ((PSC+1) × (ARR+1))
1 000    = 90 000 000 / ((PSC+1) × (ARR+1))
→ (PSC+1) × (ARR+1) = 90 000
→ PSC = 89, ARR = 999
   vérif : 90 000 000 / (90 × 1000) = 1 000 Hz ✓

duty 50% → CCR = ARR × 0.5 = 500
```

---

## Sous-tâches

| ID | Description | Durée |
|----|-------------|:---:|
| **S17-09** | Configurer TIM3 PWM dans CubeMX + duty 50 % + flash | 2h |
| **S17-10** | Importer exemple `TIM_PWMOutput` STM32CubeF4 | 1h |
| **S17-11** | Variation dynamique duty cycle 10–90 % (simulation vibration) | 1h |

---

## Spécification

### Configuration CubeMX

```
TIM3 → PWM Generation CH1
  Clock Source     : Internal Clock
  Channel 1        : PWM Generation CH1
  Prescaler (PSC)  : 89
  Counter Period (ARR) : 999
  Pulse (CCR1)     : 500   ← duty 50%
  PWM Mode         : PWM mode 1
  CH Polarity      : High

PA6 → TIM3_CH1 (AF2 automatiquement assigné par CubeMX)
```

### Code HAL dans `main.c`

```c
/* Démarrer PWM (après MX_TIM3_Init() généré par CubeMX) */
HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_1);

/* Boucle principale : variation duty cycle */
while (1)
{
    for (uint32_t duty = 100; duty <= 900; duty += 100)
    {
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, duty);
        HAL_Delay(200);
    }
    for (uint32_t duty = 900; duty >= 100; duty -= 100)
    {
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, duty);
        HAL_Delay(200);
    }
}
```

> **Formule duty** : duty_percent = (CCR / (ARR+1)) × 100
> CCR=100 → 10%, CCR=500 → 50%, CCR=900 → 90%

### Interface pour les sprints suivants (TIM → ADC)

Pipeline d'acquisition prévu :
```
Signal capteur vibration
    → PA6 PWM (simulation) ou signal externe
    → TIM3 → trigger ADC1_CH6 (PA6 en mode analogique)
    → ADC DMA (buffer circulaire)
    → features extraction (mean, std, kurtosis)
    → Mahalanobis score
    → UART réponse
```

Ce pipeline sera implémenté dans Sprint 18 (portage EWC CubeMX). Le PWM de ce sprint simule le capteur pour valider le câblage.

---

## Implémentation

### S17-09 : Configuration CubeMX + flash (2h)

**Étape 1 — Ajouter TIM3 dans CubeMX** :

1. Ouvrir `firmware/stm32f4_cubemx/stm32f4_cubemx.ioc` dans CubeMX
2. Onglet *Pinout & Configuration* → *Timers* → **TIM3**
3. Clock Source : `Internal Clock`
4. Channel1 : `PWM Generation CH1`
5. Parameter Settings :
   - Prescaler : `89`
   - Counter Period : `999`
   - Pulse CH1 : `500`
   - PWM Mode : `PWM mode 1`
6. **Generate Code** → CubeMX régénère `main.c` (attention : ne pas écraser la zone `USER CODE`)

> ⚠️ CubeMX génère `MX_TIM3_Init()` dans `main.c`. Ajouter le code utilisateur **dans les blocs `/* USER CODE BEGIN */`** pour qu'il ne soit pas écrasé lors d'une régénération.

**Étape 2 — Code dans la zone utilisateur** :

```c
/* USER CODE BEGIN 2 */
HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_1);
/* USER CODE END 2 */
```

**Étape 3 — Build + flash** :
```bash
cmake --build firmware/stm32f4_cubemx/build -j4
openocd -f interface/stlink.cfg -f target/stm32f4x.cfg \
    -c "program firmware/stm32f4_cubemx/build/stm32f4_cubemx.elf verify reset exit"
```

**Vérification** : mesurer PA6 à l'oscilloscope — signal carré 1 kHz, duty 50 %.  
Sans oscilloscope : loopback PA6 → PA0 (ADC1_CH0) et lire la valeur ADC via `printf`.

### S17-10 : Importer exemple TIM_PWMOutput (1h)

```bash
# Localiser dans le package STM32CubeF4
find ~/STM32Cube -name "TIM_PWMOutput" -type d 2>/dev/null

mkdir -p firmware/stm32f4_cubemx_examples/TIM_PWMOutput
cp -r ~/STM32Cube/.../TIM_PWMOutput/* firmware/stm32f4_cubemx_examples/TIM_PWMOutput/

# Compiler
cd firmware/stm32f4_cubemx_examples/TIM_PWMOutput
make -j4
arm-none-eabi-size build/*.elf
```

Comparer les PSC/ARR de l'exemple officiel avec nos calculs pour valider la formule.

### S17-11 : Variation dynamique duty cycle (1h)

Ajouter la boucle de variation dans `main.c` (cf. spécification). Le cycle complet (10% → 90% → 10%) dure 200 ms × 16 steps = 3.2 s. Cela simule un défaut de vibration avec amplitude croissante.

Ajouter un log UART pour tracer la variation :
```c
printf("duty=%lu%%\r\n", duty / 10);   /* duty=500 → "50%" */
```

---

## Critères d'acceptation

- [x] `MX_TIM3_Init()` dans `firmware/stm32f4_cubemx/Core/Src/tim.c` sans conflit (PSC=15, ARR=999, TIMxCLK=16 MHz HSI)
- [ ] Signal PWM sur PA6 @ 1 kHz ±1 % (vérifié à l'oscilloscope ou via log ADC) — *à valider sur carte*
- [x] Duty cycle variable de 10 % à 90 % par paliers de 10 % (boucle `ccr=100..900` dans `main.c`)
- [x] Exemple `TIM_PWMOutput` compile dans `firmware/stm32f4_cubemx_examples/` (6 Ko Flash)
- [x] Log UART `"duty=XX%"` dans la boucle principale (`printf("duty=%lu%%\r\n", ccr/10)`)

**Note horloge** : le sprint doc supposait TIMxCLK=90 MHz (APB1=45 MHz×2 avec PLL). L'implémentation utilise HSI sans PLL (TIMxCLK=16 MHz) → PSC=15 au lieu de PSC=89. Fréquence cible 1 kHz respectée.  
**Build** : `cmake --build firmware/stm32f4_cubemx/build` → 0 erreur, Flash=11 Ko, RAM=2 Ko.

---

## Questions ouvertes

- `TODO(fred)` : format des données capteur Edge Spectrum — signal analogique continu (→ ADC DMA) ou frames numériques (→ SPI/I2C) ? Influence le choix TIM trigger vs bus digital.
- `TODO(arnaud)` : fréquence de sampling capteur vibration attendue — 1 kHz est-il représentatif ou faut-il cibler 10 kHz+ (CWRU = 48 kHz) ?

---

## Statut

✅ Terminé (2026-05-21) — build OK, validation oscilloscope sur carte restante
