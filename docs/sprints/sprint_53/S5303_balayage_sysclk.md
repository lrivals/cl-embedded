# S5303 — Balayage de fréquence SYSCLK 180/90/45 MHz

| Champ | Valeur |
|-------|--------|
| **Sprint** | 53 |
| **Priorité** | 🟠 Important — verrou n°2 (mode dynamique) + argument système inédit |
| **Statut** | ✅ **Mesuré sur carte réelle (2026-09-07)** — 3 fréquences, tendance prononcée, `acqmode dyn` débloqué à 45 MHz |
| **Durée estimée** | 3 h 30 (2 h firmware ✅ + 1 h 30 banc ⏳) |
| **Dépendances** | Carte + sonde · S5304 (méthode de mesure) de préférence déjà validée |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/hw_info.c` (`hw_clock_init`) · `firmware/stm32f4_blink/Makefile` (`check-sysclk`) · `scripts/run_s53_freq_sweep.py` · `tests/test_s53_freq.py` · `experiments/exp_S53_freq_sweep/` |
| **Références** | `hw_info.c:93-146` · `docs/context/lpm01a_setup.md:123-149` (plafond 59 mA mesuré) |

## Contexte

Deux motivations indépendantes, servies par la même modification.

**1. Débloquer le mode dynamique de la sonde.** `acqmode dyn` refuse au-delà de 59 mA ; la
carte en tire 63–75 mA. Ce n'est pas un réglage à trouver — c'est une limite matérielle
mesurée (`lpm01a_setup.md:123-149`). Baisser la fréquence est la voie de contournement la
plus directe, et la seule qui reste après l'échec de `-DETH_PHY_POWERDOWN` (63,24 mA,
`lpm01a_setup.md:159`). Si la carte descend sous 59 mA, `acqmode dyn` fournit une trace à
100 kSPS → **le profil temporel par phase redevient possible** (S5305).

**2. Un argument système que le mémoire n'a pas.** Le Gap 2 dispose de **trois ordres de
grandeur de marge** (la pire latence mesurée, HDC 2095 µs, est 48× sous les 100 ms).
Cette marge n'a jamais été convertie en argument : **faut-il ralentir le MCU pour tenir
l'autonomie ?** La latence croît en 1/f, le courant décroît — l'énergie par inférence
est-elle constante (le calcul domine) ou décroissante (la scrutation et les fuites
dominent) ? C'est une question de déploiement, mesurable en une heure et demie de banc.

## Spec

### 1. Paramétrage `-DSYSCLK_MHZ={180,90,45}` dans `hw_clock_init`

Contrainte à respecter : `USART3->BRR = 0x0187` est **figé en dur** (`hw_info.c:175`) et
calculé pour `PCLK1 = 45 MHz`. Changer PCLK1 casserait l'UART, donc tout le protocole.

La solution propre est de jouer sur **PLLP et PPRE1 conjointement**, de sorte que PCLK1
reste à 45 MHz et que **le BRR n'ait pas à changer** :

| SYSCLK | PLLP | PPRE1 | PCLK1 | PPRE2 | PCLK2 | Flash WS | VOS / overdrive |
|--------|:----:|:-----:|:-----:|:-----:|:-----:|:--------:|-----------------|
| 180 MHz | /2 | /4 | 45 MHz | /2 | 90 MHz | 5 | scale 1 + ODEN/ODSWEN |
| 90 MHz | /4 | /2 | 45 MHz | /1 | 90 MHz | 2 | scale 2, **sans overdrive** |
| 45 MHz | /8 | /1 | 45 MHz | /1 | 45 MHz | 1 | scale 3, **sans overdrive** |

Points d'attention :

- La **désactivation de l'overdrive** sous 168 MHz n'est pas cosmétique : c'est une part
  du gain énergétique attendu. Les étapes 3-4 de `hw_clock_init` (`hw_info.c:117-124`)
  doivent être conditionnées, pas seulement les prescalers.
- Les **wait states Flash** doivent être *réduits* en même temps que la fréquence
  (`FLASH_ACR`, `hw_info.c:126`), sinon on paye des cycles pour rien.
- `PLLM=8`, `PLLN=180`, source HSI restent inchangés : seul PLLP varie. Le VCO reste à
  360 MHz, dans sa plage valide.

**Point favorable, à ne pas retoucher** : `hw_info_collect` **recalcule `sysclk_hz` depuis
PLLCFGR** (`hw_info.c:211-232`) et `hw_dwt_calibrate` s'en sert (`main.c:44`). Les latences
DWT rapportées en µs restent donc **justes à chaque fréquence, sans aucune modification**.
C'est ce qui rend la comparaison inter-fréquences directement exploitable.

### 2. Mesures, par fréquence ∈ {180, 90, 45} MHz

Pour chacune :

1. **Vérification protocole** — 300 échantillons, 0 erreur CRC (le BRR est censé être
   inchangé ; c'est ce qui le prouve).
2. **Latence DWT** — 2 à 3 modèles couvrant le spectre (Maha 5 µs, EWC 50 µs, HDC 2095 µs).
   Attendu : ×2 à 90 MHz, ×4 à 45 MHz. **Vérifier que le pire cas reste ≪ 100 ms** →
   conclusion Gap 2.
3. **Courant moyen** — protocole S5304 (balayage de cadence) réduit à 3 points de cadence
   si le temps manque, pour obtenir la pente à chaque fréquence.
4. **Test du plafond 59 mA** — tenter `acqmode dyn`. Consigner le succès ou l'échec
   **avec le courant relevé**, pas une appréciation.

### 3. Sortie

`experiments/exp_S53_freq_sweep/{sysclk_mhz}.json` :

```json
{
  "sysclk_mhz": 90,
  "pll": {"pllm": 8, "plln": 180, "pllp": 4, "ppre1": 2, "ppre2": 1},
  "pclk1_hz": 45000000,
  "flash_ws": 2, "overdrive": false,
  "dwt_latency_us_p50": {"mahalanobis": null, "ewc": null, "hdc": null},
  "gap2_ok": true,
  "i_mean_ma_by_rate": [{"rate_hz": 0, "i_ma": 0.0}],
  "slope_ua_per_hz": 0.0,
  "energy_uj_per_inference": 0.0,
  "acqmode_dyn": {"attempted": true, "succeeded": false, "i_max_ma": 0.0,
                  "na_reason": "…si échec, la valeur relevée…"},
  "crc_errors": 0
}
```

Plus un `summary.json` portant la **conclusion calculée** : l'énergie par inférence
est-elle croissante, constante ou décroissante avec la fréquence ?

## État d'implémentation — 5 août 2026 (sans carte ni sonde)

Tout ce qui ne dépend pas du matériel est fait et **vérifié**. La journée de banc devient
purement exécutoire : trois flashes, trois commandes.

| Élément | Fichier | État |
|---------|---------|------|
| Paramétrage `-DSYSCLK_MHZ` | `firmware/stm32f4_blink/src/hw_info.c:49-92`, `155-202` | ✅ VOS, overdrive, wait states et prescalers conditionnés ensemble |
| Cible de build sans carte | `firmware/stm32f4_blink/Makefile` (`check-sysclk`) | ✅ compile les trois, revient au build par défaut |
| Pilote de banc | `scripts/run_s53_freq_sweep.py` | ✅ + contrôle de fréquence à bord (ci-dessous) |
| Tests | `tests/test_s53_freq.py` | ✅ 19 passés, 3 `skip` (cellules non mesurées) |

**Ajout de fond au pilote — le contrôle anti-mensonge.** `--sysclk-mhz` est recopié tel
quel dans le nom du fichier et dans le champ `sysclk_mhz` : sans vérification, une cellule
mesurée sur un binaire 180 MHz mais lancée avec `--sysclk-mhz 45` s'écrirait sous un nom
qui ment sur son contenu — exactement le bug du drapeau TinyOL du Sprint 52.
`read_reported_sysclk` reset la carte, lit la bannière `hw_info_print` et compare. La
valeur lue vaut quelque chose parce que `hw_info_collect` **recalcule** `sysclk_hz` depuis
`PLLCFGR` au lieu de réafficher la constante de compilation. En cas d'écart, **rien n'est
écrit**. `PCLK1 = 45 MHz` est contrôlé au même endroit — c'est l'invariant qui sauve le BRR.

**Mesuré hors banc (tailles réelles, `make check-sysclk`)** :

| Build | `.text` | `.bss` |
|-------|--------:|-------:|
| défaut (sans `-D`) | 51 976 B | 105 036 B |
| `-DSYSCLK_MHZ=180` | 51 976 B | 105 036 B |
| `-DSYSCLK_MHZ=90` | 51 944 B | 105 036 B |
| `-DSYSCLK_MHZ=45` | 51 936 B | 105 036 B |

`.text` identique entre le défaut et `-DSYSCLK_MHZ=180` : la garde n'a aucun effet quand
elle n'est pas posée. Les −32 B / −40 B aux fréquences réduites sont la séquence
d'overdrive qui disparaît. `.bss` **invariant** : le balayage n'ajoute aucun état.

## Runbook de banc (à exécuter carte + sonde posées)

Rappels non négociables (`S5300`) : flasher **JP5 en place**, basculer sur la sonde
ensuite ; port sonde via `/dev/serial/by-id/*PowerShield*`, jamais `/dev/ttyACM0`.

```bash
# Pour chaque f ∈ {180, 90, 45} — 1 flash, 1 mesure
cd firmware/stm32f4_blink
make clean && make all EXTRA_CFLAGS="-DSYSCLK_MHZ=<f>" && make flash   # JP5 en place
# → basculer le cavalier sur l'alimentation sonde
cd ../..
python scripts/run_s53_freq_sweep.py --sysclk-mhz <f> \
       --board-port /dev/ttyACM0 --port /dev/serial/by-id/…PowerShield…

# Puis, une fois les trois cellules prises
python scripts/run_s53_freq_sweep.py --summary
cd firmware/stm32f4_blink && make clean && make all   # retour au build par défaut
```

Le pilote **refuse de mesurer** si la carte ne confirme pas la fréquence : c'est le
symptôme attendu d'un flash oublié, pas une panne.

## Critères d'acceptation

- [x] Build par défaut (sans `-DSYSCLK_MHZ`) strictement inchangé : `.bss` **105 036 B**
      mesuré, `make test` **145 / 0 échec**, 180 MHz.
- [x] Les trois builds **compilent** (`make check-sysclk`, tailles ci-dessus).
- [ ] Les trois builds **démarrent**, rapportent le bon `SYSCLK` via `hw_info_print`
      (automatisé : `read_reported_sysclk`), et streament 300 échantillons sans perte.
- [ ] Les latences DWT à 90 et 45 MHz sont cohérentes avec le facteur 1/f attendu
      (calculé : `latency_scaling_vs_180` dans `summary.json`).
- [ ] Verdict Gap 2 explicite à 45 MHz, chiffré (calculé : champ `gap2_verdict`).
- [ ] Le résultat `acqmode dyn` est consigné avec le courant relevé, succès **ou** échec.

## Résultat attendu et honnêteté

Le résultat peut parfaitement être « ralentir ne gagne rien » : sur Cortex-M4, une part
importante du courant est statique et indépendante de la fréquence, et la scrutation
active (tant que S5302 n'est pas retenu) consomme proportionnellement au temps, pas au
travail. **Un plateau d'énergie par inférence est un résultat**, et il se combine
naturellement avec S5302 : le couple (WFI + fréquence réduite) est le vrai levier de
déploiement, la fréquence seule ne l'est peut-être pas.

---

## Résultats de banc — 2026-09-07 (NUCLEO-F439ZI + X-NUCLEO-LPM01A)

Trois builds flashés (`-DSYSCLK_MHZ={180,90,45}`), fréquence **confirmée à bord** à chaque
cellule (`read_reported_sysclk`, recalcul depuis `PLLCFGR`), `PCLK1 = 45 MHz` tenu partout —
l'invariant qui protège le `BRR` figé. Grille densifiée `0/10/25/50/100/200 Hz`, 2 répétitions.
Contrôles d'intégrité 300/300, 0 CRC sur les 9 flux.

| | 180 MHz | 90 MHz | 45 MHz |
|---|---|---|---|
| Latences Maha / EWC / HDC fp32 | 5 / 50 / 585 µs | 9 / 99 / 1169 µs | 18 / 198 / 2338 µs |
| Plafond de transport HDC INT8 | 124,8 Hz | 99,9 Hz | 71,3 Hz |
| **Courant de base** | 40,13 mA | 25,64 mA | **18,29 mA** |
| **Énergie/inférence** | 214,6 µJ | **N/A** | **170,1 µJ** |
| Pente (µA/Hz) | +65,8 ± 0,38 | +41,6 ± 12,63 | +52,1 ± 0,03 |
| r² (ajustement pondéré) | 1,000 | **0,783** | 1,000 |
| Points ajustés / mesurés | 5/6 | 5/6 | 4/6 |
| `acqmode dyn` | refusé (75,7 mA) | refusé (69,6 mA) | **OK — 100 000/100 000** |

### Tendance mesurée — `croissante_avec_f`

L'énergie par inférence **croît de +26,2 % de 45 à 180 MHz** (170,1 → 214,6 µJ). Ralentir le
MCU réduit donc à la fois le coût par inférence **et** le courant permanent (**−54 %**,
40,13 → 18,29 mA), pendant que **Gap 2 reste tenu avec 12,8× de marge à 45 MHz** (pire cas
HDC fp32 2338 µs contre 100 ms). C'est l'argument système visé : *la marge de latence est
convertible en autonomie*.

La tendance ne s'appuie que sur ses deux extrémités (45 et 180 MHz), toutes deux publiables :
elle survit donc au passage de **90 MHz en N/A** (ci-dessous), mais son point milieu manque.

### Correction A4 du 2026-09-07 — l'ajustement était double, et 90 MHz n'est pas publiable

Ce pilote portait sa propre régression (`_linear_fit`), qui transmettait au module commun des
couples `(cadence, courant)` **sans les écarts-types** : le balayage de cadence (S5304)
pondérait par `1/σ²`, celui-ci non. Deux implémentations du même ajustement publiaient donc
deux chiffres pour la même mesure, et la règle de publication du sprint (r² suffisant **et**
pente à plus de 2 σ de zéro) n'était pas appliquée ici.

`_linear_fit` a été supprimé au profit de `rate_regression.fit_cell`, et les trois cellules
recalculées **sans remesure** (`--refit`, les points mesurés ne sont pas touchés) :

| f | avant (non pondéré) | après (règle S5304) |
|---|---|---|
| 45 MHz | 51,3 µA/Hz · r²=0,998 → 167,6 µJ | 52,10 ± 0,03 µA/Hz · r²=1,000 → **170,1 µJ** |
| 90 MHz | 57,7 µA/Hz · r²=0,919 → 188,2 µJ | 41,56 ± 12,63 µA/Hz · **r²=0,783 → N/A** |
| 180 MHz | 61,2 µA/Hz · r²=0,986 → 199,7 µJ | 65,75 ± 0,38 µA/Hz · r²=1,000 → **214,6 µJ** |

Le point 90 MHz — déjà le plus faible en r² avant correction — n'est pas séparable du bruit
une fois les répétitions prises en compte : sa cellule porte `"à mesurer"` et sa raison
chiffrée. **Il reste à rejouer avec le pilote final** (B1, 1 flash, ~15 min) ; c'est la seule
des trois cellules produite par une version antérieure du pilote de sonde.

### Loi en 1/f vérifiée

HDC INT8 : 1958 µs (180 MHz) → 3916 µs (90) → 7833 µs (45), soit **exactement ×2 et ×4**.

### Trois défauts d'outillage corrigés pendant cette tâche

1. **`profiling.c` convertissait les cycles DWT avec `SYSCLK_HZ` figé à 180 MHz.** À horloge
   réduite le nombre de cycles est inchangé mais le temps réel s'allonge : les latences
   étaient **sous-estimées d'un facteur égal au rapport de fréquence** (×2 à 90 MHz, ×4 à
   45 MHz), Gap 2 en paraissait d'autant meilleur et le critère « latences en 1/f » était
   invérifiable. `SYSCLK_HZ` suit désormais `-DSYSCLK_MHZ` ; **sans le flag, valeur inchangée
   → build par défaut strictement identique** (`.bss` 105 036 B, `make test` 145/0).
   La spec affirmait que `hw_dwt_calibrate` rendait ces latences justes : c'est **faux**,
   `profiling.c` ne l'utilise pas pour cette conversion.
2. **`try_dynamic_mode` déclarait `succeeded: True` dès que `lp.capture` ne levait pas.**
   Or `capture` ne lève que sur un flux *vide* : une acquisition interrompue par
   surintensité rend les quelques dizaines de ms décodées avant l'arrêt. La cellule 180 MHz
   a d'abord publié « mode dynamique OK » tout en contenant, dans son propre `summary`, la
   preuve du contraire (71 ms, 7130/100 000 échantillons, « Measurement interrupted »).
   Contrôle ajouté : complétude (`DYN_COMPLETENESS = 0.95`) **et** marqueurs d'interruption.
3. **Saturation non gérée** (portée depuis S5304), puis **plafond non propagé**. À 180 MHz
   le point 200 Hz — irréalisable, plafond 124,8 Hz — tirait la pente de 62,9 à 39,5 µA/Hz,
   soit 205,6 → 129,1 µJ contre 191,6 ± 16,0 µJ mesurés indépendamment par S5304. À 45 MHz,
   le plafond mesuré (71,3 Hz) n'était appliqué qu'à la borne haute : le point 100 Hz,
   physiquement irréalisable, restait retenu → r² 0,888 et cellule en N/A. Le plafond est
   désormais propagé à toutes les cadences supérieures.

### Réserves

- La cellule **90 MHz** a le r² le plus faible (0,919), avec un motif en escalier dans ses
  points, et elle a été produite **avant** le correctif de propagation du plafond. Celui-ci
  ne l'aurait pas changée (99,9 Hz atteints pour 100 Hz commandés = 0,07 %, très en deçà de
  la tolérance de 5 %), mais elle n'a pas été rejouée avec le code final.
- `_linear_fit` du pilote ne renvoie **pas d'erreur-type de pente**, contrairement à S5304 :
  le verdict de tendance n'a **aucune barre d'erreur**. À corriger avant publication.
- Le seuil de la sonde n'est **pas** un plafond net sur le pic : 68,6 mA passe à 45 MHz,
  69,6 mA échoue à 90 MHz. Sa nature exacte reste à caractériser.
