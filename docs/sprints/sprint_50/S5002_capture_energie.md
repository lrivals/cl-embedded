# S5002 — Capture énergie réelle (µJ par inférence) depuis CSV LPM01A

| Champ | Valeur |
|-------|--------|
| **Sprint** | 50 |
| **Priorité** | 🔴 Critique — livrable central du CR (énergie mesurée). |
| **Statut** | ✅ **Campagne exécutée board réelle 8/8 cellules (2026-08-04/05)** — courant moyen mesuré ; µJ/inférence N/A **mesuré** (voir § Résultat) |
| **Durée estimée** | 5h |
| **Dépendances** | S5001 (setup + calibration) · `scripts/energy_capture.py` ✅ · `scripts/run_s50_energy.py` ✅ · `configs/energy_campaign_s50.yaml` ✅ |
| **Fichiers cibles** | `experiments/exp_S50_energy/` · MAJ `experiments/exp_S33_energy/` |
| **Références** | CR §4 · S33 (chaîne `segment_by_phase`/`integrate_energy_uj`) |

## Contexte

Exécuter la campagne réelle et remplir les champs `"à mesurer"` du Sprint 33 avec des valeurs mesurées. La
chaîne logicielle existe (S33) et est testée bout-en-bout ; il ne reste qu'à lui fournir les CSV réels.

## Spec

### 1. Exécution

```
python scripts/energy_capture.py --campaign --csv <lpm01a_export.csv> \
    --model {ewc,hdc,tinyol,mahalanobis} --encoding {fp32,int8} \
    --out experiments/exp_S50_energy/
```

- `derive_phase_windows` déduit idle/inférence/MAJ depuis la colonne sync PA8.
- `integrate_energy_uj` → **µJ par phase** ; **µJ/inférence** = énergie phase inférence / N.
- Décomposition par composant (S5001) si le banc l'a permis (sinon `"na"` honnête).

### 2. Remplissage des champs

- `experiments/exp_S50_energy/{model}_{encoding}.json` : `energy_uj_per_inference`, `energy_uj_per_update`,
  `by_component{mcu,periph,sensor}`, `delta_int8_vs_fp32`.
- MAJ `experiments/exp_S33_energy/` : remplacer les `"à mesurer"` par les valeurs (ou laisser si non couvert).

### 3. Règle d'honnêteté

- Une cellule non capturée reste `"à mesurer"` (jamais 0, jamais estimée).
- `by_component` non isolable → `"na"` + `na_reason`.

## Format de sortie

`experiments/exp_S50_energy/*.json` (µJ mesurés) + `summary.json` (deltas int8 vs fp32 calculés).

## État actuel & runbook (implémenté, en attente matériel)

> **2026-07-24** : STM32CubeMonitor-Power / X-NUCLEO-LPM01A pas encore disponibles.
> La chaîne est prête et **`experiments/exp_S50_energy/` est déjà peuplé de 8 cellules
> placeholder honnêtes** (`"à mesurer"`, `source: "placeholder"`) + `summary.json`. Aucun
> µJ inventé. Il ne reste qu'à fournir les CSV réels.

Un driver S50 mince — `scripts/run_s50_energy.py` — **réutilise strictement** la
segmentation/intégration de `energy_capture.py` (aucune réécriture) et **ajoute** ce qui
manquait pour l'autonomie (S5003) : `phase_durations_s` (durées par phase déduites des
mêmes fronts PA8 réels — la clé que `autonomy.py`/`profile_memory.py` lisait sans qu'aucun
producteur ne l'écrive), `energy_uj_per_inference` (= µJ phase active / N), et `by_component`
(delta MCU/périph si CSV séparés, `sensor: "na"` car capteurs simulés UART).

> **2026-08-04 — sonde branchée et pilotable sans CubeMonitor-Power.** Le producteur de CSV
> manquant existe désormais : `scripts/lpm01a_probe.py --capture --freq 100k --duration 10 --out <csv>`
> écrit directement les colonnes `time,current,voltage` attendues par `_load_csv` (validé :
> 100 000/100 000 échantillons décodés à 100 kSPS, chaîne S33 traversée sans adaptation).
> Runbook : [`docs/context/lpm01a_setup.md`](../../context/lpm01a_setup.md).
>
> **Conséquence sur la segmentation** : la sonde ne capture aucune voie numérique, donc pas de
> colonne `sync` PA8 → `derive_phase_windows` refuse (à raison). Les fenêtres viendront soit du
> déclenchement externe (`--trigsrc d7`, PA8 câblé sur D7), soit du **protocole delta** déjà
> prévu ci-dessus. Il reste à alimenter la NUCLEO à travers la sonde : sans cela, **aucun µJ**.
>
> **2026-08-04 (suite) — méthode révisée par la mesure.** Le banc monté a révélé deux
> contraintes qui invalident la segmentation par phases prévue ici : la sonde n'a **aucune
> voie de synchronisation**, et le mode dynamique s'interrompt sur `Overcurrent >59mA`
> (la F439ZI à 180 MHz dépasse son plafond). La voie retenue est le **protocole delta** en
> mode statique — `scripts/run_s50_energy_delta.py` (+ 13 tests). Instructions d'exécution
> complètes : [`S5008_handoff_mesures.md`](S5008_handoff_mesures.md).

**Dès réception du banc** :

1. Campagne LPM01A (protocole S5001, marqueur PA8), un CSV par cellule (colonnes
   `time,current[,voltage],sync`).
2. Renseigner `configs/energy_campaign_s50.yaml` : `csv:` (chemin), `n_inference:`, `n_update:`.
3. Rejouer la campagne — les placeholders deviennent des mesures :

```bash
python scripts/run_s50_energy.py --manifest configs/energy_campaign_s50.yaml \
    --out experiments/exp_S50_energy/
# ou une cellule unique :
python scripts/run_s50_energy.py --model ewc --encoding int8 \
    --csv captures/ewc_int8.csv --n-inference 1000 --out experiments/exp_S50_energy/
```

> **Note honnête (marquage PA8 1-bit, S33)** : `inference` et `MAJ CL` partagent le niveau
> haut → non séparables depuis une seule trace. `energy_uj_per_update` reste `"à mesurer"`
> jusqu'à un protocole delta dédié (inférence-seule vs inférence+MAJ, cf. latences séparées S26)
> ou un encodage PA8 multi-bit (évolution firmware).

## Résultat de la campagne (2026-08-04/05, board réelle NUCLEO-F439ZI)

La campagne a été exécutée. **Deux constats de banc ont imposé de changer de méthode en
cours de route** — ils sont mesurés, reproduits, et documentés dans
[`lpm01a_setup.md`](../../context/lpm01a_setup.md) §4bis.1–4bis.2.

### Ce qui a fait échouer le protocole delta

1. **La première acquisition d'une session est biaisée d'environ +8 mA.** Quatre
   acquisitions identiques au repos : 62,74 / 54,94 / 54,86 / 54,82 mA. Le protocole delta
   plaçait sa fenêtre de référence *en première position* : son biais dépassait l'effet
   cherché et **inversait le signe du résultat**. Correctif : `lpm01a_probe.warmup()`, un
   rebut de préchauffage appelé par les deux pilotes.
2. **La référence « au repos » est plus haute que tous les régimes de flux** (54,81 ± 0,11 mA
   contre 46,40 à 51,22 mA), y compris Mahalanobis dont l'inférence n'occupe que ~0,05 % du
   temps. Un écart que le taux d'occupation n'explique pas et **dont la cause n'est pas
   établie** ; le firmware attend la trame UART par scrutation active (`pipeline.c`), donc le
   « repos » n'est pas inactif, mais d'autres causes de banc ne sont pas exclues.
   ⇒ L'énergie marginale par inférence ressort **négative** : elle n'a pas de sens face à
   cette référence.

### Méthode retenue et livrée

`scripts/run_s50_board_current.py` — **courant moyen à cadence imposée** (100 Hz, fenêtre
10 s, 3 répétitions par cellule, préchauffage écarté). À cadence identique, le trafic UART et
le travail hôte sont communs à toutes les cellules : l'écart entre deux cellules est
imputable au modèle. `N = cadence × fenêtre` est **exact par construction**, ce qui lève le
point d'attention du handoff sur le nombre réel d'inférences.

### Grille 8/8 mesurée — courant moyen (mA)

Référence au repos : **54,78 ± 0,13 mA**. Dispersion inter-répétitions ≤ 0,05 mA.

| modèle | FP32 | INT8 | Δ INT8−FP32 |
|---|---|---|---|
| EWC | 46,550 | 46,547 | −0,003 |
| HDC | 47,843 | **51,220** | **+3,377** |
| TinyOL | 46,600 | 46,513 | −0,087 |
| Maha | 46,403 | 46,373 | −0,030 |

L'ordre des courants suit exactement celui des latences connues (Maha 5 µs → 46,4 mA …
HDC INT8 1958 µs → 51,2 mA), ce qui confirme que la part attribuable au modèle est bien
mesurée.

**Réponse Gap 3** : l'INT8 **ne réduit pas la consommation**. EWC (−0,003 mA) et Maha
(−0,030 mA) sont dans le bruit ; TinyOL gagne marginalement (−0,087 mA) ; **HDC INT8 coûte
7,1 % de courant en PLUS** que son FP32 — cohérent avec sa latence INT8 dégradée (1958 µs vs
585 µs). L'INT8 reste donc justifié par la RAM (÷4, S28/S49), pas par l'énergie.

### Ce qui reste `"à mesurer"`, et pourquoi

| Champ | Statut | Raison |
|---|---|---|
| `energy_uj_per_inference` | N/A **mesuré** | référence de repos non exploitable (ci-dessus) ; levier = mise en sommeil `WFI` du firmware |
| `energy_uj_per_update` | N/A | même référence ; exigerait une campagne dédiée `--update` |
| `phases_uj`, `phase_durations_s` | N/A | pas de voie de synchronisation + mode dynamique inaccessible |
| `by_component.mcu/periph` | N/A | rail VDD_MCU global, décomposition non câblée |
| `by_component.sensor` | `"na"` | capteurs simulés par UART (S5001) |

> **Portée de la mesure** : le courant relevé couvre **tout le rail VDD_MCU**, pas seulement
> le cœur. Ce n'est pas « l'énergie du modèle » : c'est la consommation de la carte sous une
> charge d'inférence donnée, dont seules les **différences entre cellules** sont imputables
> au modèle.

### Reproduire

```bash
python scripts/run_s50_board_current.py --board-port /dev/serial/by-id/…STLink…-if02 \
    --rate-hz 100 --window 10 --repeats 3 --dataset monitoring \
    --out experiments/exp_S50_energy/
# maha_int8 exige un firmware dédié (-DMAHA_INT8) :
make -C firmware/stm32f4_blink all EXTRA_CFLAGS="-DETH_PHY_POWERDOWN -DMAHA_INT8" && \
  make -C firmware/stm32f4_blink flash
python scripts/run_s50_board_current.py --board-port … --only maha_int8 …
```

## Contraintes

- Réutilisation stricte de `energy_capture.py` (0 réécriture de la segmentation/intégration).
- Aucun µJ écrit sans CSV réel correspondant.
- Board uniquement (l'énergie PC n'a pas de sens ici, CR §1).

## Vérification

```bash
# après campagne réelle :
python -c "import json;d=json.load(open('experiments/exp_S50_energy/ewc_int8.json'));\
assert isinstance(d['energy_uj_per_inference'],(int,float)) and d['energy_uj_per_inference']>0"
grep -rl '"à mesurer"' experiments/exp_S50_energy/     # cellules non encore capturées, honnêtes
python -m pytest tests/test_energy_capture.py -q       # chaîne S33 non régressée
```
