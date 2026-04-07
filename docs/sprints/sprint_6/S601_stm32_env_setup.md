# S6-01 — Setup environnement de développement STM32 (NUCLEO-F439ZI)

| Champ | Valeur |
|-------|--------|
| **ID** | S6-01 |
| **Sprint** | Sprint 6 — Semaine 1 Phase 2 (20–27 mai 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 3h |
| **Dépendances** | Sprint 5 terminé (Phase 1 complète) |
| **Fichiers cibles** | `docs/sprints/sprint_6/S601_stm32_env_setup.md` |

---

## Objectif

Disposer d'un environnement de développement embarqué opérationnel sur la **NUCLEO-F439ZI** via VS Code : compilation ARM GCC, flash OpenOCD, debug Cortex-Debug.

> ⚠️ **Important** : La NUCLEO-F439ZI (STM32F439ZI — Cortex-M4, 256 Ko RAM, **pas de NPU**) est une board de développement intermédiaire. Elle n'est **pas** la cible finale du projet. La cible finale est le **STM32N6** (Cortex-M55, ~64 Ko RAM modèle, NPU inférence-only). Ce sprint valide la chaîne d'outillage (compile → flash → debug) avant d'avoir accès au hardware cible.

**Critère de succès** : une LED clignote sur la NUCLEO-F439ZI, un breakpoint est atteignable depuis VS Code via ST-LINK, le `launch.json` est reproductible et documenté ici.

---

## Carte de développement

**Carte** : NUCLEO-F439ZI  
**Microcontrôleur** : STM32F439ZI (gamme STM32F4)

| Caractéristique | Valeur |
|-----------------|--------|
| Cœur | ARM Cortex-M4 @ 180 MHz |
| FPU + DSP | ✅ |
| Flash | Jusqu'à 2 Mo |
| RAM | 256 Ko |
| NPU | ❌ (pas d'accélérateur IA) |
| Debugger intégré | ST-LINK (pas de sonde externe nécessaire) |
| Connecteurs | Arduino / ST morpho |

---

## Sous-tâches

### 1. Installer la toolchain ARM GCC

```bash
# Ubuntu/Debian
sudo apt install gcc-arm-none-eabi binutils-arm-none-eabi

# Vérifier
arm-none-eabi-gcc --version
```

### 2. Installer OpenOCD

```bash
# Ubuntu/Debian
sudo apt install openocd

# Vérifier
openocd --version
```

### 3. Installer VS Code + extensions

Extensions à installer :

| Extension | Rôle |
|-----------|------|
| C/C++ (Microsoft) | IntelliSense, débogage natif |
| Cortex-Debug | Débogage sur MCU via OpenOCD/GDB |
| STM32 VS Code Extension *(optionnel)* | Intégration CubeMX |
| Makefile Tools **ou** CMake Tools | Build system |

```bash
code --install-extension ms-vscode.cpptools
code --install-extension marus25.cortex-debug
```

### 4. Créer un projet test avec STM32CubeMX

1. Ouvrir STM32CubeMX → New Project → sélectionner `STM32F439ZI`
2. Activer GPIO PA5 (LED verte LD2) en mode `GPIO_Output`
3. Générer le projet en mode **Makefile** (ou CMake)
4. Ouvrir le dossier généré dans VS Code

### 5. Compiler le projet

```bash
make -j4
# Sortie attendue : build/project.elf
```

### 6. Flasher via OpenOCD

```bash
openocd -f interface/stlink.cfg \
        -f target/stm32f4x.cfg \
        -c "program build/project.elf verify reset exit"
```

### 7. Configurer `launch.json` pour le debug Cortex-Debug

Créer `.vscode/launch.json` :

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug STM32 (NUCLEO-F439ZI)",
            "type": "cortex-debug",
            "request": "launch",
            "servertype": "openocd",
            "configFiles": [
                "interface/stlink.cfg",
                "target/stm32f4x.cfg"
            ],
            "cwd": "${workspaceRoot}",
            "executable": "./build/project.elf",
            "runToEntryPoint": "main",
            "showDevDebugOutput": "none"
        }
    ]
}
```

### 8. Valider le debug depuis VS Code

1. Brancher la NUCLEO-F439ZI via USB (ST-LINK)
2. `F5` dans VS Code → le programme s'arrête sur `main()`
3. Poser un breakpoint dans la boucle LED → vérifier l'arrêt

---

## Critères d'acceptation

- [ ] `arm-none-eabi-gcc --version` retourne une version ≥ 10.x
- [ ] `openocd --version` retourne une version ≥ 0.11
- [ ] Le projet blink compile sans erreur (`make -j4`)
- [ ] La LED LD2 (PA5) clignote sur la carte après flash
- [ ] Un breakpoint dans la boucle principale est atteignable depuis VS Code
- [ ] `launch.json` validé et sauvegardé dans ce sprint

---

## Questions ouvertes

- `TODO(arnaud)` : confirmer si on cible bien la NUCLEO-F439ZI ou si une autre board est disponible plus tôt
- `TODO(dorra)` : à quel stade migrer le projet vers STM32N6 — attendre Phase 2 semaine 2 ou commencer directement ?
- `TODO(fred)` : Edge Spectrum a-t-il une board STM32N6 disponible pour les tests ?

---

## Notes

- La NUCLEO-F439ZI dispose d'un ST-LINK intégré → pas besoin de sonde externe (J-Link, etc.)
- Pour le STM32N6 (cible finale), la configuration OpenOCD changera (`target/stm32n6x.cfg`) mais le workflow reste identique
- Garder ce projet blink comme template de référence pour les futurs portages C

**Complété le** : _(à renseigner)_
