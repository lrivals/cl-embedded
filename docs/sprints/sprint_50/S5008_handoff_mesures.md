# S5008 — Passation : instructions pour la session de mesure énergie

> ## ✅ EXÉCUTÉ les 2026-08-04 et 2026-08-05 — voir § 6 « Compte rendu d'exécution »
>
> Les instructions ci-dessous restent le récit de ce qui était **prévu**. Deux constats de
> banc mesurés pendant la session ont invalidé le protocole delta du § 3 et imposé une
> autre méthode. Lire le § 6 **avant** de rejouer quoi que ce soit.

| Champ | Valeur |
|-------|--------|
| **Sprint** | 50 |
| **Objet** | Instructions d'exécution des mesures énergie restantes (S5002, S5003, placeholders S33) |
| **Préparé le** | 2026-08-04 (banc monté, code prêt, aucune mesure d'énergie encore produite) |
| **Prérequis matériel** | NUCLEO-F439ZI + X-NUCLEO-LPM01A câblés (§ Banc), multimètre, accès au cavalier `JP5` |
| **Runbook banc** | [`docs/context/lpm01a_setup.md`](../../context/lpm01a_setup.md) |

> **À lire d'abord.** Ce document dit ce qui est **déjà fait et vérifié**, ce qui est
> **impossible et pourquoi** (contraintes mesurées, pas supposées), et la suite exacte des
> commandes à exécuter. Ne pas re-découvrir ces contraintes : elles ont coûté une session.

---

## 1. État du banc — acquis vérifiés

| Élément | État |
|---|---|
| Sonde LPM01A | ✅ branchée, FW 1.0.1, pilotée sans STM32CubeMonitor-Power par `scripts/lpm01a_probe.py` |
| Port série | ✅ résolu par identifiant USB stable (`/dev/serial/by-id/*PowerShield*`) — **ne jamais coder `/dev/ttyACM0` en dur**, l'ST-LINK occupe ce numéro |
| Décodage du flux | ✅ validé : 100 000/100 000 échantillons à 100 kSPS, contrôlé contre le résumé de la sonde (I_max 9,093 µA ↔ 9,0 µA annoncés) |
| Câblage cible | ✅ `CN14` GND → masse Nucleo ; `CN14` VOUT(+) → côté VDD_MCU de `JP5` retiré |
| Courant moyen relevé | ✅ ~59 mA (mode statique) — mise en service, **pas** une cellule de campagne |
| Firmware veille PHY | ✅ écrit et testé (`src/eth_phy.c`, 145 tests 0 échec, `.bss` défaut invariant 105 036 B) — ❌ **pas encore flashé** |
| Chaîne delta | ✅ écrite et testée (`scripts/run_s50_energy_delta.py`, 13 tests) — ❌ **jamais exécutée sur carte** |
| Énergie mesurée | ❌ **aucune** — tous les JSON restent `"à mesurer"` |

## 2. Contraintes matérielles mesurées — ne pas les recombattre

1. **Aucune voie de synchronisation.** Le flux LPM01A ne transporte que le courant.
   `derive_phase_windows` (colonne PA8) est **inapplicable** — elle lève une exception, à
   raison. Ne pas fabriquer de colonne `sync`.
2. **Mode dynamique inaccessible.** `acqmode dyn` s'interrompt sur
   `error: Overcurrent >59mA, dynamic acquisition stopped` (I_max relevé 75,6 mA). Le
   profilage temporel par inférence à 100 kSPS est hors de portée à ce niveau de courant.
   Seul `acqmode stat` (≤ 200 mA) aboutit, et renvoie **une seule valeur moyennée**.
3. **La sonde n'alimente la carte que pendant une acquisition.** Hors acquisition, la
   F439ZI s'effondre et redémarre en boucle (~100 bannières de boot UART en 6 s) ;
   pendant une acquisition maintenue, elle est stable (0 redémarrage). Toute commande hôte
   (flash, streaming) doit donc tourner **dans** une acquisition — c'est le rôle de
   `--hold-run`.
4. **Le flash sous alimentation sonde est instable.** Même sous maintien, le SWD s'accroche
   par intermittence et le reset échoue (`Unable to reset target`). **Procédure retenue :
   flasher sur alimentation normale (`JP5` en place), puis basculer sur la sonde pour
   mesurer.** C'est une manipulation manuelle de cavalier, assumée.

## 3. Séquence d'exécution

### Étape 0 — vérifier le banc (2 min)

```bash
python scripts/lpm01a_probe.py --selftest
```

Attendu : `Board power supply to target status: ok`, FW `1.0.1`, rails 3,3/3,0/1,8 V proches
des consignes. Si le port n'est pas trouvé, vérifier `ls /dev/serial/by-id/`.

### Étape 1 — flasher la veille du PHY Ethernet (manipulation manuelle)

**Pourquoi** : UM1974 §6.7 impose de neutraliser le PHY pour une mesure de consommation
correcte ; il est alimenté par le rail mesuré alors que le firmware ne l'utilise pas. Et
c'est la seule piste pour repasser sous le plafond de 59 mA du mode dynamique.

1. Débrancher l'USB de la Nucleo, **remettre `JP5`** (alimentation normale par l'USB).
2. Débrancher le fil VOUT de la sonde (évite deux sources en parallèle).
3. Flasher :

```bash
make -C firmware/stm32f4_blink clean
make -C firmware/stm32f4_blink all EXTRA_CFLAGS=-DETH_PHY_POWERDOWN
make -C firmware/stm32f4_blink flash
```

4. Rebrancher VOUT, **retirer `JP5`**, USB des deux côtés.

### Étape 2 — mesurer l'effet du PHY (le chiffre qui décide de la suite)

```bash
python scripts/lpm01a_probe.py --capture --acqmode stat --freq 1k --duration 10 \
    --out captures/idle_phy_off.csv
```

Comparer au repère **~59 mA** de la mise en service (PHY actif).

- **Si le courant passe nettement sous ~50 mA** → retenter le mode dynamique
  (`--acqmode dyn --freq 100k`). S'il aboutit sans overcurrent, **le profilage temporel
  redevient possible** et il faut alors préférer cette voie au protocole delta.
- **Sinon** → rester sur le protocole delta (étape 3), et consigner le constat.

Dans les deux cas : écrire les deux courants (avec/sans PHY) dans
[`lpm01a_setup.md`](../../context/lpm01a_setup.md) §4bis. C'est une mesure, elle a sa place.

### Étape 3 — campagne delta, 8 cellules

Principe : deux fenêtres de **même durée**, l'une au repos, l'autre pendant N inférences ;
`µJ/inférence = (I_actif − I_repos) × V × T / N`. Le pilote fait les deux fenêtres et écrit
la cellule JSON.

```bash
python scripts/run_s50_energy_delta.py --model ewc --encoding int8 \
    --n-inference 500 --window 10 \
    --stream-cmd "python scripts/sensor_stream.py --dataset monitoring --n 500" \
    --out experiments/exp_S50_energy/
```

À répéter pour les 8 couples `{ewc,hdc,tinyol,maha} × {fp32,int8}` (drapeaux UART du
modèle : cf. `sensor_stream.py`). **Vérifier à chaque cellule** que `delta_a > 0` : sinon la
cellule sort honnêtement en `"à mesurer"` avec `na_reason`, et il faut augmenter `N` ou la
durée de fenêtre plutôt que d'accepter le résultat.

> **Point d'attention** : `--n-inference` doit être le nombre d'inférences **réellement**
> exécutées dans la fenêtre, pas la consigne. Si le stream n'en exécute que 380 sur 500
> demandées, c'est 380 qu'il faut passer — sinon les µJ/inférence sont faux.

### Étape 4 — autonomie (S5003) et agrégation

```bash
python scripts/run_s50_energy.py --manifest configs/energy_campaign_s50.yaml \
    --out experiments/exp_S50_energy/
```

`src/evaluation/autonomy.py` calcule `Autonomie_h = Capacité / I_moy` à partir des capacités
de `configs/hw_profile_f439zi.yaml`. L'I_moy vient désormais d'une mesure réelle.

### Étape 5 — figures, docs, tests

- Régénérer le catalogue : `python scripts/generate_figures.py --catalog energy_real`
  → les figures E1–E3 aujourd'hui grises « à mesurer » doivent afficher des valeurs
  **chargées via `load_experiment`** (la garde AST interdit tout chiffre en dur).
- Mettre à jour `S5002`, `S5003`, `docs/roadmap_phase2.md`, `docs/triple_gap.md`, `CLAUDE.md`.
- `python -m pytest tests/test_s50_energy.py tests/test_energy_delta.py tests/test_energy_capture.py tests/test_lpm01a_probe.py tests/test_autonomy.py -q`
- `make -C firmware/stm32f4_blink test` (attendu : 145 tests, 0 échec) et vérifier que le
  `.bss` du build **par défaut** vaut toujours **105 036 B**.
- Invoquer le skill `graphify_sprint_update` en fin de tâche.

## 4. Règles d'honnêteté propres à ce sprint

- Une cellule non mesurée reste `"à mesurer"` — **jamais 0, jamais une estimation**.
- Ce que le protocole delta ne peut pas produire garde son `na_reason` : profil par phase,
  `energy_uj_per_update`, `by_component.mcu/periph`.
- `by_component.sensor` reste `"na"` : les capteurs sont simulés par UART, il n'y a pas de
  capteur physique alimenté (S5001).
- Le courant mesuré inclut **tout le rail VDD_MCU**, pas seulement le cœur : le dire dans les
  conclusions plutôt que de présenter les µJ comme « l'énergie du modèle ».
- L'énergie par inférence obtenue est une **énergie marginale moyenne**, pas un profil
  temporel : ne pas la présenter comme un breakdown par phase.

## 5. Fichiers concernés

| Fichier | Rôle |
|---|---|
| `scripts/lpm01a_probe.py` | pilote sonde (selftest, capture, `--hold-run`, `--power-on/off`) |
| `scripts/run_s50_energy_delta.py` | campagne delta → cellule JSON |
| `scripts/energy_capture.py` | primitives d'énergie (dont `energy_uj_per_inference_delta`) |
| `scripts/run_s50_energy.py` | agrégation + autonomie (schéma JSON de référence) |
| `firmware/stm32f4_blink/src/eth_phy.c` | veille du PHY, sous `-DETH_PHY_POWERDOWN` |
| `configs/hw_profile_f439zi.yaml` | `energy_calibration:` — source unique des paramètres |
| `tests/test_energy_delta.py`, `tests/test_lpm01a_probe.py` | garde-fous du calcul et de l'honnêteté |

---

## 6. Compte rendu d'exécution (2026-08-04 / 2026-08-05)

### 6.1 Ce qui s'est passé

| Étape prévue | Résultat |
|---|---|
| 0 — vérifier le banc | ✅ FW 1.0.1, `power supply to target: ok`, rails 3300/3002/1801 mV |
| 1 — flasher la veille PHY | ✅ `-DETH_PHY_POWERDOWN` flashé, `.bss` défaut **invariant 105 036 B** |
| 2 — mesurer l'effet du PHY | ✅ mesuré : **63,24 mA** → la piste du mode dynamique est **fermée par la mesure** (§ 4bis.1 du runbook) |
| 3 — campagne delta 8 cellules | ❌ **abandonnée** : deux constats de banc l'invalident (§ 6.2) — remplacée par la campagne « courant moyen » |
| 4 — autonomie + agrégation | ✅ 8/8 cellules, autonomie chiffrée par une autre voie (S5003) |
| 5 — figures, docs, tests | ✅ 6 PNG, docs à jour, 96 tests Python + Unity 145/0 |

### 6.2 Les deux constats qui ont changé la méthode

1. **La première acquisition d'une session est biaisée d'environ +8 mA** (62,74 puis 54,94 /
   54,86 / 54,82 mA dans des conditions strictement identiques). Le protocole delta plaçait
   sa fenêtre de référence *en premier* : le biais dépassait l'effet cherché et **inversait
   le signe du résultat**. Corrigé par `lpm01a_probe.warmup()`, appelé par les deux pilotes.
   Sans ce contrôle, la campagne aurait rendu des cellules « delta négatif » qu'on aurait pu
   prendre pour un problème de N ou de fenêtre — c'est-à-dire chercher longtemps au mauvais
   endroit.
2. **La référence « au repos » est plus haute que tous les régimes de flux** (54,78 ± 0,13 mA
   contre 46,40–51,22 mA), y compris pour Mahalanobis dont l'inférence occupe ~0,05 % du
   temps. Écart non explicable par le taux d'occupation, **cause non établie**. Le firmware
   attend la trame UART par scrutation active (`pipeline.c`), donc le « repos » n'est pas
   inactif — mais d'autres causes de banc ne sont pas exclues, et ce n'est pas affirmé.
   ⇒ Les µJ/inférence n'ont **pas de sens** face à cette référence : ils sortiraient négatifs.

Un contrôle a tranché entre « effet réel » et « artefact » : quatre captures identiques au
repos, puis un flux vérifié vivant pendant la fenêtre active (2499/2500 trames, 0 CRC). Ce
sont ces deux vérifications qui ont évité de publier un résultat inversé.

### 6.3 Méthode livrée à la place

`scripts/run_s50_board_current.py` — **courant moyen à cadence imposée** (100 Hz, fenêtre
10 s, 3 répétitions, préchauffage écarté). À cadence identique, le trafic UART et le travail
hôte sont communs à toutes les cellules ; l'écart entre deux cellules est imputable au
modèle. `N = cadence × fenêtre` est exact par construction — ce qui **lève le point
d'attention du § 3** sur le nombre réel d'inférences, sans avoir à l'estimer.

Résultats et réponse Gap 3 : [`S5002_capture_energie.md`](S5002_capture_energie.md) §
Résultat. Autonomie : [`S5003_autonomie.md`](S5003_autonomie.md).

### 6.4 Corrections d'outillage faites au passage

| Correctif | Pourquoi |
|---|---|
| `run_s50_energy.py` : `load_measured_cell` | rejouer le manifeste **écrasait** les cellules mesurées par des placeholders (perte de données silencieuse) |
| `lpm01a_probe.warmup()` | rebut de préchauffage, appelé aussi par `run_s50_energy_delta.py` |
| `run_s50_board_current.BUILD_SPECIFIC` | empêche d'écrire `maha_int8` depuis le build FP32 — même classe de bug que le drapeau TinyOL du Sprint 52 |
| `autonomy.average_current_ma_from_delta` | 3ᵉ voie d'`I_moy`, pour le cas delta |
| `write_autonomy` : `duty_cycle → null` + `regime_mesure` | ne pas présenter une autonomie mesurée à 100 Hz comme celle d'un scénario duty-cyclé à 1 Hz |
| docstring `run_s50_energy_delta.py` | l'exemple citait `--n` (inexistant) et omettait `--model` — un flux sans `--model` exécute Mahalanobis en silence |

### 6.5 Ce qui reste ouvert

- **µJ/inférence** : exige une mise en sommeil (`WFI`) de l'attente UART du firmware, pour
  disposer d'une vraie référence au repos. Levier identifié, non implémenté (hors périmètre).
- **Cause de l'écart repos↔flux de −8 mA** : non établie. À instrumenter si le chiffre doit
  être interprété, et non seulement utilisé comme référence commune.
- **`by_component` MCU/périphériques** : exige un câblage séparé des rails, pas une campagne.
- **`energy_uj_per_update`** : même blocage que les µJ/inférence.
