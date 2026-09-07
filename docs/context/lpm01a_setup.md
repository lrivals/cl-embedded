# Banc de mesure énergie — X-NUCLEO-LPM01A (PowerShield)

> État **2026-08-04** : sonde branchée, dialogue série validé, acquisitions réelles
> effectuées (1 kSPS ASCII et 100 kSPS binaire). **Aucun µJ n'est encore mesuré** :
> la NUCLEO-F439ZI n'est pas alimentée à travers la sonde. Ce document est le
> runbook du banc ; les paramètres relevés vivent dans
> `configs/hw_profile_f439zi.yaml` → `energy_calibration:` (source unique).

## 1. Identification de la sonde

La carte énumère en USB CDC sous le nom `PowerShield (Virtual ComPort in FS Mode)` :

```bash
lsusb | grep 0483:5740
ls -l /dev/serial/by-id/ | grep -i powershield
```

| Élément | Valeur relevée |
|---------|----------------|
| Firmware | `1.0.1` (commande `version`) |
| Port | `/dev/ttyACM0` |
| Débit série | 921 600 (validé jusqu'au flux 100 kSPS) |
| Groupe d'accès | `dialout` (l'utilisateur en fait déjà partie) |

> STM32CubeMonitor-Power n'est **pas requis** : `scripts/lpm01a_probe.py` pilote la
> sonde en direct via son interface ASCII (UM2243) et écrit le CSV consommé par
> `scripts/energy_capture.py`.

## 2. Vérification du banc

```bash
python scripts/lpm01a_probe.py --selftest      # version + autotest + température
python scripts/lpm01a_probe.py --calibrate     # auto-calibration (après ±5 °C)
```

L'autotest doit rapporter `Board power supply to target status: ok` et les trois
rails 3,3 V / 3,0 V / 1,8 V proches de leur consigne.

## 3. Matériel nécessaire (demande labo)

| Matériel | Quantité | Pourquoi |
|---|---|---|
| **Multimètre numérique** (tension DC + continuité) | 1 | Deux vérifications **obligatoires avant tout câblage** : (a) repérer quelle broche de `CN14` est VOUT(+) et laquelle est GND — la numérotation n'est pas lisible sur la carte ; (b) identifier laquelle des 2 broches de `JP5` est côté VDD_MCU (recommandation ST). Se tromper revient à brancher la sortie de la sonde contre le régulateur 3,3 V de la Nucleo. |
| **Fils Dupont femelle-femelle**, 10–20 cm | 4 (2 utiles + 2 rechange) | Relier `CN14` (sonde) à `JP5`/GND (Nucleo). Femelle-femelle car les deux extrémités sont des **broches mâles** (header de la sonde, embase du cavalier IDD). |
| **Cavaliers 2,54 mm** (shunts) | 2–3 | `JP5` doit être retiré : garder des shunts de rechange pour le remettre ensuite (et ne pas immobiliser la carte si l'original est perdu). |
| **2 ports USB libres** (ou hub alimenté) | — | La sonde **et** l'ST-LINK doivent être connectés **en même temps** : la sonde alimente le MCU, l'ST-LINK sert au flash et au streaming UART. |

Optionnel, selon la décision sur le PHY Ethernet (cf. §5) :

| Matériel | Pourquoi |
|---|---|
| Station de soudage + tresse à dessouder, pointe fine | Uniquement si l'on choisit de **retirer `SB13`** plutôt que de mettre le PHY en power-down par logiciel. À ne prendre qu'après avoir mesuré si le PHY pèse réellement sur le résultat. |
| Bracelet antistatique | Manipulation de cartes nues. |

> **À vérifier avant de commander les fils** : le pas de `CN14`. Des broches espacées
> d'environ 2,5 mm dans une embase plastique = standard 2,54 mm → les fils Dupont
> conviennent. Si les broches sont nettement plus fines et resserrées (~1,25 mm, type JST),
> il faut le câble confectionné correspondant, les Dupont ne s'y enfichent pas.

Pour repérer VOUT au multimètre, maintenir la sortie sous tension :

```bash
python scripts/lpm01a_probe.py --power-on 30   # 30 s à 3,3 V sur CN14
```

## 4. Repérage au multimètre (avant tout câblage)

> ⚠️ **Règle absolue** : n'utiliser que les modes **tension continue (V⎓)** et
> **continuité (bip)**. Ne JAMAIS mettre le multimètre en mode **courant (A)** entre ces
> points : en mode ampèremètre les pointes sont quasi un court-circuit, et les poser entre
> 3,3 V et GND met le régulateur en défaut. Ne pas laisser une pointe ponter deux broches
> voisines.

### 4.1 Trouver VOUT(+) et GND sur `CN14` (sonde)

1. Seule la sonde est branchée (USB), rien de câblé sur `CN14`.
2. Multimètre en **V⎓**, calibre 20 V (ou automatique).
3. Pointe **noire** sur une masse connue de la sonde : une broche marquée `GND` sur son
   connecteur Arduino, ou la coque métallique du connecteur USB.
4. Mettre la sortie sous tension et sonder les 4 broches de `CN14` une par une avec la
   pointe **rouge** :

   ```bash
   python scripts/lpm01a_probe.py --power-on 30
   ```

   - la broche qui lit **≈ 3,3 V** est **VOUT(+)** ;
   - celle qui lit **0 V** est **GND** (le confirmer en continuité, sonde hors tension :
     bip entre cette broche et la masse de référence) ;
   - noter ce que lisent les deux autres (probablement inutilisées ou lignes de sense).
5. Noter le repère physique (bord de carte, détrompeur de l'embase) — la numérotation
   sérigraphiée n'est pas exploitable à l'œil.

### 4.2 Trouver le côté VDD_MCU de `JP5` (Nucleo)

Cavalier **en place**, les deux broches sont court-circuitées : impossible de les
distinguer. Il faut donc le retirer d'abord.

1. **Débrancher l'USB** de la Nucleo, puis retirer le cavalier `JP5`.
2. **Méthode sûre, carte non alimentée — continuité (bip)** : chercher un bip entre chaque
   broche de `JP5` et la broche `3V3` du connecteur Zio.
   - la broche qui **bipe** avec `3V3` est le **côté régulateur** ;
   - l'autre est le **côté VDD_MCU** → c'est là que va VOUT(+) de la sonde.
3. **Confirmation sous tension (optionnel)** : rebrancher l'USB ST-LINK, `JP5` toujours
   retiré, pointe noire sur une masse de la Nucleo :
   - côté régulateur ≈ **3,3 V** (toujours alimenté) ;
   - côté VDD_MCU ≈ **0 V** (le MCU n'est plus alimenté, c'est normal et attendu :
     UM1974 §6.7 — `JP5` OFF sans ampèremètre = STM32 non alimenté).
4. Rebrancher le cavalier `JP5` tant que le câblage définitif n'est pas fait.

### 4.3 Après câblage, avant la campagne

Sonde reliée (`CN14` GND → GND Nucleo, `CN14` VOUT+ → côté VDD_MCU de `JP5` retiré),
USB ST-LINK branché :

```bash
python scripts/lpm01a_probe.py --capture --freq 1k --duration 2 --out /tmp/check.csv
```

Le courant doit être de l'ordre du **mA** (MCU actif), pas quelques centaines de nA
(= rien d'alimenté, câblage à revoir) ni une valeur qui sature.

## 4bis. Contrainte de plage mesurée — la F439ZI dépasse le mode dynamique

**Relevé 2026-08-04, carte câblée à travers la sonde** (`JP5` retiré, VOUT sur VDD_MCU) :

| Mode | Résultat mesuré |
|---|---|
| `acqmode dyn` (100 kSPS, profil temporel) | ❌ **`error: Overcurrent >59mA, dynamic acquisition stopped`** — acquisition interrompue après 70 ms, I_max relevé 75,6 mA |
| `acqmode stat` (moyenne, jusqu'à 200 mA) | ✅ acquisition complète — **I_moy = 59,07 mA** (1 valeur moyennée) |

La NUCLEO-F439ZI à 180 MHz consomme **au-dessus du plafond du mode dynamique** du LPM01A
(~59 mA). Conséquences directes sur la méthode de campagne :

- Le **profilage temporel par inférence** (100 kSPS, séparation des phases dans le temps)
  est **hors de portée** tant que le courant reste à ce niveau — ce n'est pas un réglage à
  trouver, c'est une limite de la sonde.
- Le **mode statique reste exploitable** et suffit au **protocole delta** : I_moy sur une
  fenêtre au repos vs I_moy sur une fenêtre contenant N inférences ⇒
  `µJ/inférence = (I_actif − I_repos) × V × T / N`. C'est déjà la voie retenue faute de
  voie de synchronisation (§7).
- **Piste pour récupérer le mode dynamique** : neutraliser le PHY Ethernet (§5). UM1974 le
  demande déjà pour la justesse de la mesure ; un PHY 100BASE-TX actif consomme plusieurs
  dizaines de mA, donc le retirer du rail peut faire repasser la carte **sous les 59 mA**.
  À mesurer — c'est l'étape suivante, et elle est doublement motivée.

> Le 59,07 mA est une **mesure de mise en service**, pas une cellule de campagne : il inclut
> tout ce que porte le rail VDD_MCU (PHY Ethernet compris) et le firmware alors flashé.
> Aucun µJ n'en est dérivé.

### 4bis.1 Veille du PHY Ethernet — mesurée, sans effet utile

Firmware flashé avec `-DETH_PHY_POWERDOWN` (`src/eth_phy.c`, mise en veille par SMI/MDIO),
puis capture statique 10 s à travers la sonde :

| Configuration | I mesuré |
|---|---|
| Mise en service (PHY actif, firmware antérieur) | 59,07 mA |
| **PHY mis en veille (`-DETH_PHY_POWERDOWN`)** | **63,24 mA** |

La veille du PHY **ne fait pas repasser la carte sous le plafond de 59 mA** : le mode
dynamique reste inaccessible et le profilage temporel par inférence reste hors de portée.
La piste du §4bis est donc **fermée par la mesure**.

> Réserve d'honnêteté : les deux lignes ne forment pas un A/B strict — elles proviennent de
> deux firmwares et de deux sessions différentes, et l'on sait depuis (§4bis.2) que la
> première acquisition d'une session est biaisée. L'écart de 4 mA n'est donc pas
> interprétable comme un coût du PHY ; la seule conclusion tirée ici est **négative** et
> elle suffit : on reste largement au-dessus de 59 mA.

### 4bis.2 La première acquisition d'une session est biaisée (~+8 mA)

**Constat mesuré le 2026-08-04**, quatre acquisitions statiques enchaînées dans des
conditions **strictement identiques** (carte au repos, rien de changé entre les captures) :

| Acquisition | I mesuré |
|---|---|
| 1 | **62,74 mA** ← biaisée |
| 2 | 54,94 mA |
| 3 | 54,86 mA |
| 4 | 54,82 mA |

La sonde lit environ **8 mA de trop sur la première acquisition**, puis se stabilise avec
une dispersion de **±0,06 mA** — une répétabilité largement suffisante pour séparer des
modèles.

**Ce biais a une conséquence qui a failli passer inaperçue** : le protocole delta plaçait sa
fenêtre de référence *en première position*. Son biais dépassait donc l'effet cherché, au
point d'**inverser le signe du résultat** (µJ/inférence négatifs, cellules rejetées en
`na_reason`). Toute session de mesure doit commencer par une acquisition **jetée** —
c'est ce que fait `lpm01a_probe.warmup()`, appelé par les deux pilotes de campagne.

### 4bis.3 Deux niveaux de repos — la carte bascule au **premier flux reçu** (S5301b)

**Constat mesuré le 2026-08-05**, sonde LPM01A alimentant `VDD_MCU` (JP5 retiré), build par
défaut, fenêtres de 10 s. La carte présente **deux niveaux de repos stables**, séparés
d'environ **10 mA**, et la variable qui les sépare n'est ni le temps, ni l'état du port
hôte :

| Condition de repos (aucune trame en cours) | I mesuré |
|---|---|
| après coupure/remise de l'alimentation, jamais streamée (4 acquisitions) | 49,74 – 49,87 mA (plat) |
| idem, port série **ouvert** et maintenu, aucune trame | 49,72 / 49,89 mA |
| idem, après une **impulsion DTR** seule (comme à l'ouverture de `sensor_stream`) | 49,90 mA |
| **après le premier flux reçu** depuis la mise sous tension | 39,74 – 40,08 mA (plat) |

Le pas mesuré est de **−9,8 mA** (`step_first_stream_a`), reproductible sur trois sessions.
La série post-coupure est **plate** sur plusieurs minutes : ce n'est donc pas un
établissement thermique ou une constante de temps, mais un **changement d'état**, déclenché
par le traitement de vraies trames — l'ouverture du port et l'impulsion DTR ne suffisent
pas. **Le mécanisme côté carte n'est pas identifié** (il ne l'a pas été cherché ici : le
diagnostic visait la validité de la référence, pas la cause matérielle).

Deux hypothèses ont été **réfutées** au passage, mesures à l'appui
(`experiments/exp_S53_counterbalance/idle_states_diagnostic.json`) :

- l'interruption du flux (`proc.terminate()`, fermeture du port, DTR/RTS relâchés) ne
  laisse **pas** la carte à l'arrêt : le repos après flux interrompu est indiscernable du
  repos port fermé, et la carte répond **20/20 trames** juste après, à chaque cycle ;
- l'**ouverture** du port, à elle seule, ne déplace pas le courant de façon significative.

Le surcoût de charge mesuré dans la même session (flux à 100 Hz vs repos) vaut **+2,0 mA** :
le pas de 10 mA le dépasse d'un facteur 5, ce qui suffit à **inverser le signe** d'un delta
repos↔charge.

**Règle de protocole qui en découle** — elle explique et corrige le défaut du Sprint 50 :

1. **La référence au repos se prend APRÈS au moins un flux**, jamais sur une carte qui n'a
   rien reçu depuis sa mise sous tension. C'est exactement ce que faisait le protocole
   delta du Sprint 50 (référence en tête de session, avant toute cellule) : sa référence
   était sur le niveau haut, ses cellules sur le niveau bas, d'où des µJ/inférence négatifs.
2. **Aucune comparaison de courants absolus entre sessions.** Les niveaux relevés varient
   d'une session à l'autre (49,8 mA ici contre 54,8 mA les 4–5 août sur le même build) ;
   seules les différences **intra-session** sont interprétables. Cela concerne directement
   toute comparaison veille/scrutation qui exigerait deux firmwares, donc deux sessions
   (§ S5302) : elle doit être présentée comme inter-session, ou ancrée sur un écart mesuré
   dans chaque session.

Diagnostic reproductible :

```bash
python scripts/diag_s53_idle_states.py --board-port /dev/serial/by-id/…STLink…-if02 \
    --repeats 2 --with-power-cycle --settling-points 3
```

**Réplication du 2026-08-06** — seconde session contre-balancée (`--repeats 4`, 9 repos
ordonnés), firmware par défaut reconstruit et reflashé entre les deux sessions
(`.bss` = 105 036 B). Le pas est retrouvé, et surtout : **les niveaux absolus se déplacent,
les écarts intra-session tiennent**. C'est la règle 2 ci-dessus vérifiée par la mesure, et
la raison pour laquelle elle doit être respectée.

| Grandeur | 2026-08-05 | 2026-08-06 |
|---|---|---|
| repos établi (**absolu — non comparable**) | 43,970 mA | 40,073 mA |
| pas repos avant tout flux → régime établi | +10,440 mA | +10,057 mA |
| surcoût `ewc_fp32` vs repos établi | +2,167 mA | +2,027 mA |
| surcoût `hdc_fp32` vs repos établi | +3,457 mA | +3,355 mA |

Conséquence pratique : **une référence de repos ne se réutilise jamais d'une session à
l'autre** — elle se reprend dans la session où l'on mesure, et après au moins un flux.

> ⚠️ **`JP5` retiré = la cible n'est alimentée que pendant une acquisition.** Une fois la
> campagne terminée, `probe.release()` coupe l'alimentation : tout flux UART ou `make flash`
> lancé ensuite se bloque, la carte étant hors tension. Passer par
> `lpm01a_probe.py --hold-run "<commande>"`, qui maintient l'acquisition pendant la commande
> hôte.

## 5. Câblage de la cible (étape restante)

**Empiler le shield sur la NUCLEO ne suffit pas** : il faut **2 fils**. C'est la
procédure officielle ST — la sonde alimente le VDD du MCU via son connecteur `CN14`,
en lieu et place du cavalier IDD de la carte cible.

> ⚠️ **Piège de nommage** : `CN14` désigne le connecteur 4 points de mesure **sur le
> PowerShield**, mais le connecteur **RJ45 Ethernet** sur la NUCLEO-144. Ne pas confondre.

| Sonde (X-NUCLEO-LPM01A) | → | NUCLEO-F439ZI |
|---|---|---|
| `CN14` pin 1 — GND | → | n'importe quelle broche GND (ex. connecteur Zio `CN8`) |
| `CN14` pin 3 — VOUT (+) | → | côté **VDD_MCU** du cavalier `JP5` (IDD), cavalier **retiré** |

Procédure :

1. **Retirer le cavalier `JP5`** (marqué IDD) de la NUCLEO. UM1974 §6.7 : `JP5` ON = STM32
   alimenté (défaut) ; `JP5` OFF = un ampèremètre **doit** être connecté, sinon le STM32
   n'est pas alimenté du tout. C'est précisément le rôle que prend la sonde.
2. **Identifier au multimètre** laquelle des deux broches de `JP5` est côté VDD_MCU avant
   de câbler (recommandation ST) — l'autre vient du régulateur 3,3 V de la carte.
3. **Garder l'USB ST-LINK branché** : il alimente le reste de la carte et reste
   indispensable au flash et au streaming UART (`sensor_stream.py`). La sonde n'alimente
   alors **que le MCU** — c'est exactement l'isolation « MCU seul » visée par S5001.
4. Régler la tension : `volt 3300m` (fait automatiquement par le pilote).
5. Vérifier qu'un courant non nul et plausible (mA) apparaît une fois la carte alimentée
   (à vide, la sonde ne relève que quelques centaines de nA de bruit — pas une mesure).

> ⚠️ **Spécifique NUCLEO-144 / F439ZI — PHY Ethernet.** UM1974 §6.7 : « pour obtenir une
> consommation correcte, le PHY Ethernet doit être mis en mode power-down, ou `SB13` doit
> être retiré » (§6.11 note 3 : registre Basic Control du PHY, adresse 0x00, bit 11 à 1).
> Le firmware du projet n'utilise pas Ethernet, mais le PHY et son horloge de référence
> polluent la mesure de courant tant qu'ils tournent. **À traiter avant la campagne**,
> sinon les µJ mesurés incluent un consommateur étranger au modèle CL.

### Variante « empilage » (non retenue)

L'UM2243 mentionne l'alimentation d'une carte Nucleo par le connecteur Arduino du shield.
Cette voie alimente la carte **entière** par la broche 3V3 (pas le seul MCU) et entre en
conflit avec l'alimentation par l'USB ST-LINK dont nous avons besoin pour l'UART : elle
n'est donc pas retenue ici. `TODO(dorra)` si l'on veut la documenter (configuration des
cavaliers `JP1`/`JP4`/`JP9`/`JP10` du shield, UM2243).

> La carte n'était **pas branchée** au moment de la configuration de la sonde
> (aucun ST-LINK `0483:374b` énuméré) : cette étape reste à faire.

Sources : [UM1974 (Nucleo-144, MB1137)](https://www.st.com/resource/en/user_manual/um1974-stm32-nucleo144-boards-mb1137-stmicroelectronics.pdf) ·
[UM2243 (X-NUCLEO-LPM01A)](https://www.st.com/resource/en/user_manual/um2243-stm32-nucleo-expansion-board-for-power-consumption-measurement-stmicroelectronics.pdf) ·
[Note ST « How to connect my STM32 board to the X-NUCLEO-LPM01A »](https://community.st.com/t5/stm32-mcus/how-to-connect-my-stm32-board-to-the-x-nucleo-lpm01a/ta-p/49597)

## 6. Capture

```bash
# 10 s à 100 kSPS (format binaire choisi automatiquement au-delà de 10 kHz)
python scripts/lpm01a_probe.py --capture --freq 100k --duration 10 \
    --out captures/ewc_int8.csv
```

Le CSV produit porte les colonnes `time,current,voltage` attendues par
`energy_capture._load_csv`, précédées d'un en-tête de commentaires (firmware,
fréquence, nombre d'échantillons décodés **vs** attendus).

**Contrôle d'intégrité systématique** : le pilote affiche le nombre d'échantillons
décodés face au nombre annoncé par la sonde dans son propre résumé de fin
d'acquisition. Tout écart signale une perte de flux — la trace est alors à rejeter,
jamais à compléter.

### Détail du flux binaire

Le flux n'est pas un bloc continu d'échantillons : il est découpé en blocs de
1000 échantillons précédés d'un en-tête de 9 octets
`F0F3 <horodatage 32 bits, ms> <état> FFFF`, et clos par `F0F4 FFFF`. Ignorer ces
en-têtes intermédiaires désaligne le décodage et fabrique des valeurs aberrantes
(défaut constaté puis corrigé lors de la mise en service). Chaque échantillon est un
mot 16 bits big-endian : quartet haut = exposant, 12 bits bas = mantisse
(`0x52A0` ⇔ 0x2A0 × 16⁻⁵ = 640,9 µA).

## 7. Limite connue — pas de voie de synchronisation

La sonde n'échantillonne **que le courant** : elle ne capture aucune voie numérique.
La colonne `sync` (niveau PA8, `ENERGY_MARKERS`) attendue par `derive_phase_windows`
ne peut donc **pas** venir du LPM01A, et `segment_by_phase` refuse — à raison — de
segmenter sans elle. Deux voies restent ouvertes, aucune ne fabrique de donnée :

| Voie | Principe | Coût |
|------|----------|------|
| `--trigsrc d7` | PA8 câblé sur Arduino D7 (pont à souder, UM2243) **déclenche** l'acquisition → t=0 aligné sur le début de phase ; les fenêtres se déduisent du protocole et de `n_inference` | une soudure |
| Protocole delta | Deux captures réelles (idle seul, puis idle + N inférences) → µJ/inférence = différence / N | aucun câblage |

Le protocole delta est celui déjà anticipé par S5002 pour contourner la limite du
marquage PA8 1 bit (`inference` et `MAJ CL` partagent le niveau haut).

## 8. Enchaînement vers les JSON d'expérience

```bash
# 1. renseigner configs/energy_campaign_s50.yaml (csv:, n_inference:, n_update:)
# 2. rejouer la campagne — les placeholders « à mesurer » deviennent des mesures
python scripts/run_s50_energy.py --manifest configs/energy_campaign_s50.yaml \
    --out experiments/exp_S50_energy/
```

Toute cellule sans CSV réel reste `"à mesurer"` (règle CLAUDE.md : aucun chiffre
inventé).

## 9. Séance du 2026-09-07 — corrections de banc et révision du plafond

### 9.1 Le mode dynamique EST accessible — à 45 MHz

Le §4bis concluait que `acqmode dyn` était hors de portée. **Révision mesurée** : à
`-DSYSCLK_MHZ=45`, une acquisition dynamique complète passe — **1 000 000 échantillons sur
1 000 000 attendus** à 100 kSPS (`experiments/exp_S53_freq_sweep/45.json`).

| SYSCLK | `acqmode dyn` | I_max relevé |
|---|---|---|
| 180 MHz | refusé (71 ms, 7117/100 000) | 75,7 mA |
| 90 MHz | refusé | 69,6 mA |
| **45 MHz** | **complet** | 68,6 mA |

> Le seuil n'est **pas** un plafond net sur le pic : 68,6 mA passe, 69,6 mA échoue. Sa nature
> exacte reste à caractériser — ne pas le présenter comme « 59 mA sur le maximum ».

### 9.2 Un échec de `dyn` ne lève pas d'exception

`lp.capture` ne lève que sur un flux **vide**. Une acquisition interrompue par surintensité
rend les dizaines de ms décodées avant l'arrêt : le code appelant croit avoir réussi. Le
signalement est ailleurs — dans le **nombre d'échantillons** face aux attendus, et dans le
`summary` (`Measurement interrupted`). Tout appelant doit vérifier ces deux points
(cf. `run_s53_freq_sweep.try_dynamic_mode`, `DYN_COMPLETENESS = 0.95`).

### 9.3 `--acqmode` n'a plus de défaut figé

`hold_run` et `power_on` recevaient `args.acqmode`, dont le défaut CLI était `dyn` — mode
plafonné que la carte dépasse. L'acquisition s'arrêtait sur `Overcurrent`, la sortie
retombait, et la cible **redémarrait en boucle pendant toute la commande hôte** : l'hôte
lisait alors les **codes ASCII de la bannière de boot** à la place des trames de réponse
(`pred` entre 10 et 122). Mesuré : **46 bannières en 3 s sous `dyn`, 6 puis stable sous
`stat`**. `--acqmode` est désormais résolu par opération — `dyn` pour une capture, `stat`
pour tout maintien — l'option explicite restant prioritaire.

### 9.4 Contrôle de plausibilité avant toute campagne

Une carte alimentée par la sonde tire des **mA**. Si la mesure donne des µA, la cible n'est
pas alimentée par la sonde — typiquement **`JP5` refermé**, auquel cas le régulateur de la
carte alimente `VDD_MCU` et court-circuite la sonde. Cas vécu : trois exécutions de S5305 ont
profilé une carte qu'elles n'alimentaient pas, sans qu'aucun garde-fou ne le signale.

```bash
python scripts/lpm01a_probe.py --capture --freq 1k --duration 3 --acqmode stat --out /tmp/check.csv
# attendu : ordre du mA. Des µA = câblage ou cavalier à revoir.
```

**Depuis le 2026-09-07, ce contrôle est automatique** : `lpm01a_probe.capture()` refuse toute
acquisition dont le courant moyen est sous `MIN_PLAUSIBLE_CURRENT_A` (1 mA) et lève une
`LPM01AError` nommant la moyenne relevée, le seuil et les causes usuelles. Le garde-fou est
placé là parce que **toutes** les acquisitions du dépôt passent par cette fonction
(`measure_current` des pilotes de banc, `capture_under_load` du profil par phase, `warmup`) :
aucun pilote ne peut donc écrire une cellule depuis une carte non alimentée.

Le seuil est calé sur la mesure, pas sur une intuition : le courant le plus bas jamais relevé
sur cette carte est **18,2 mA** (repos à 45 MHz), soit ×18 au-dessus du seuil — il ne
discrimine aucun régime réel. Une mesure délibérément sous le seuil (veille profonde future)
reste possible via `min_current_a=None`, une échappatoire explicite et jamais implicite.

Symptôme complémentaire : `JP5` mal enfiché donne `Target voltage: 3.26 V` **et**
`init mode failed` sous OpenOCD — la broche de mesure de l'ST-LINK lit le rail `3V3` côté
régulateur, tandis que `VDD_MCU`, séparé par `JP5`, reste flottant.

### 9.5 Fiabilité du dialogue série

- `take_control()` draine le port jusqu'au silence avant `htc` et **retente une fois** : le
  premier `htc` d'une session échouait régulièrement (`Command unknown`) sur un prompt
  résiduel, ce qui a fait tomber une campagne entière.
- `voltage_v()` ne lit plus **le dernier jeton** de `volt get` mais le dernier jeton **au
  format** mantisse/exposant : une ligne de service intercalée (jeton lu : `handled`) faisait
  échouer la cellule.

Ces trois comportements — et avec eux `command`, `power_on`, `hold_run` et `capture` — sont
depuis le 2026-09-07 **couverts par des tests** (`tests/test_lpm01a_probe.py`, fausse liaison
série rejouant des réponses relevées au banc). Ils ne l'étaient pas : les quatre défauts de la
séance étaient tous dans ces fonctions, et l'un des correctifs est passé au vert *sans son
`import re`* — l'erreur n'aurait explosé qu'au banc, au milieu d'une campagne.

### 9.6 L'anomalie « repos > charge » dépend du binaire

Mesure d'isolation (même carte, même sonde, même modèle EWC, à quelques minutes d'écart) :
pente de `I(cadence)` **+17,96 µA/Hz** (r² 0,979) sur le build aux dimensions par défaut,
contre **−0,94 ± 0,62 µA/Hz** (r² 0,366) sur le build S38 `EWC_IN=4`. L'anomalie du Sprint 50
n'est donc **pas une fatalité du banc** ; sa mécanique reste **non établie**.
