# S5001 — Cadrage mesure énergie (carte ENAC + LPM01A) + calibration

| Champ | Valeur |
|-------|--------|
| **Sprint** | 50 |
| **Priorité** | 🔴 Critique — sans setup ni calibration, aucune capture valide ; S5002 en dépend. |
| **Statut** | ✅ Cadrage implémenté (config + protocole) · ✅ **sonde LPM01A branchée et configurée (2026-08-04)** · ⏳ câblage de la cible à travers la sonde restant |
| **Durée estimée** | 4h |
| **Dépendances** | Carte NUCLEO-F439ZI (ENAC) · X-NUCLEO-LPM01A · `scripts/energy_capture.py` ✅ · `configs/hw_profile_f439zi.yaml` ✅ · marqueurs `ENERGY_MARKERS` PA8 ✅ |
| **Fichiers cibles** | `docs/sprints/sprint_50/S5001_cadrage_energie.md` · MAJ `configs/hw_profile_f439zi.yaml` |
| **Références** | CR §4 · `TODO(dorra)` (fréq. échantillonnage/calibration, S33) |

## Contexte

Poser physiquement la sonde et fixer les paramètres de campagne. Résout le `TODO(dorra)` ouvert depuis S33
(fréquence d'échantillonnage, calibration LPM01A).

> **État au 2026-07-24** : la NUCLEO-F439ZI est branchée (ST-LINK/UART), mais **STM32CubeMonitor-Power et le
> X-NUCLEO-LPM01A ne sont pas encore disponibles**. Ce cadrage prépare donc **tout le mesurable dès maintenant**
> (paramètres de sonde, plan d'isolation, protocole de campagne, config de calibration) sans écrire aucun chiffre
> d'énergie. La pose de la sonde et la capture réelle (S5002) restent différées jusqu'à réception du matériel :
> le sprint est **complétable dès que le banc LPM01A est monté** — le code et le protocole sont prêts.
>
> **Mise à jour 2026-08-04 — sonde reçue, branchée et configurée.** Le X-NUCLEO-LPM01A
> est opérationnel sur le poste (`/dev/ttyACM0`, FW 1.0.1) et **STM32CubeMonitor-Power
> n'est pas nécessaire** : `scripts/lpm01a_probe.py` pilote la sonde en direct (interface
> ASCII UM2243) et écrit un CSV `time,current,voltage` directement consommé par
> `energy_capture.py`. Validé par acquisitions réelles : 100 000/100 000 échantillons
> décodés à 100 kSPS (binaire) et 2 000/2 000 à 1 kSPS (ASCII), décodage contrôlé contre
> le résumé de fin d'acquisition émis par la sonde (I_max 9,093 µA ↔ 9,0 µA annoncés).
> Le `TODO(dorra)` « fréquence d'échantillonnage » est **levé** : `sampling_rate_measured_hz: 100000`.
> Runbook complet : [`docs/context/lpm01a_setup.md`](../../context/lpm01a_setup.md).
>
> **Reste à faire** (matériel) : monter le shield sur la NUCLEO et router le VDD cible à
> travers la sonde. Tant que ce n'est pas fait, **aucun µJ n'est mesuré** — S5002/S5003
> restent « à mesurer ».
>
> **Correction de cadrage (§1 ci-dessous)** : la sonde n'échantillonne **aucune voie
> numérique** — la colonne de sync PA8 exigée par `derive_phase_windows` ne peut pas venir
> du LPM01A. Alignement des phases par `trigsrc d7` (PA8 → D7, pont à souder) ou par
> protocole delta (deux captures réelles). Consigné dans `energy_calibration.sync_channel_available: false`.

## Spec réalisée

### 1. Setup matériel (à exécuter dès réception du banc)

- Récupérer la carte à l'ENAC ; connecter le X-NUCLEO-LPM01A (PowerShield) en série sur l'alim MCU (les cavaliers
  d'alimentation NUCLEO doivent router le VDD MCU à travers la sonde — retirer `JP` d'alim directe selon la doc du banc).
- Setup logiciel = **lien partagé** (STM32CubeMonitor-Power ou équivalent) → export CSV (courant, temps, sync).
- Marqueur GPIO **PA8** (`ENERGY_MARKERS`) balise les phases → colonne de sync du CSV (`derive_phase_windows`).
- Le CSV exporté **doit** contenir la colonne de synchronisation (capture du niveau PA8 en parallèle du courant) ;
  sinon `energy_capture.py`/`run_s50_energy.py` **refusent de segmenter** (aucune fenêtre fabriquée — cf. S33).

### 2. Plan d'isolation par composant (CR §4)

| Cible | Méthode | Isolable sur banc ENAC ? |
|-------|---------|--------------------------|
| MCU seul | alim MCU isolée via LPM01A, périphériques désactivés (UART TX coupé après flash, LEDs off) | ✅ attendu (à confirmer) |
| Périphériques | delta avec/sans UART/GPIO actifs (2 captures, même firmware, périph. activés/coupés) | 🟡 partiel (delta seulement) |
| Capteurs | delta avec/sans acquisition capteur | ❌ **non isolable** sur ce banc — les capteurs sont **simulés** (flux UART hôte→board, cf. `sensor_stream.py`), il n'y a pas de capteur physique alimenté par la board |

> **Honnêteté (CR §4)** : la décomposition « capteur » n'a pas de sens sur ce banc (données injectées par UART,
> pas de capteur alimenté). Le champ `by_component.sensor` reste donc `"na"` (+ `na_reason`) dans les JSON S5002,
> jamais un delta fabriqué. MCU et périphériques sont, eux, isolables par delta de captures.

### 3. Calibration LPM01A (`TODO(dorra)` résolu — bloc config)

Ajouté à `configs/hw_profile_f439zi.yaml` → section **`energy_calibration:`** (consommée par les drivers) :

- **Fréquence d'échantillonnage** : consigne **100 kHz** (mode *high sampling* du LPM01A), soit ≳ 10× le taux
  d'évènement à résoudre — la latence d'inférence board mesurée (DWT) va de ~5 µs (Mahalanobis) à ~2 ms (HDC) ;
  pour résoudre une inférence de ~50–100 µs il faut une période ≤ ~5–10 µs. `sampling_rate_measured_hz: null`
  reste à relever sur le banc.
- **Plage de courant** : `current_range` µA (veille) → mA (actif), `autorange: true` pour couvrir la dynamique
  sans écraser la veille.
- **Offset / gain** de la sonde : `offset_ua: null`, `gain: null` (`<à_calibrer>`) — correction `I_corr = gain·I + offset`
  appliquée seulement une fois renseignés ; par défaut aucune correction (offset 0, gain 1).
- **Tension** : `supply_voltage_v: 3.3` V.

### 4. Protocole de campagne

Séquence balisée PA8 : `idle → inférence ×N → MAJ CL ×N → idle`, répétée par (modèle, encodage), pour que
`segment_by_phase` découpe proprement et que `integrate_energy_uj` donne µJ/inférence.

- `N` (nombre d'inférences / de MAJ dans la fenêtre) est **consigné dans le manifeste de campagne**
  (`configs/energy_campaign_s50.yaml`, champ `n_inference` / `n_update`) pour que `µJ/inférence = E_inférence / N`
  soit calculable — sinon `energy_uj_per_inference` reste `"à mesurer"`.
- Limite connue du marquage **1-bit** PA8 (S33) : `startup`/`acquisition`/`inference` partagent le niveau haut →
  reportés `inference`, le niveau bas → `idle`. Granularité 4-phases = encodage multi-bit firmware (évolution future).

## Format de sortie

- `docs/sprints/sprint_50/S5001_cadrage_energie.md` (setup + isolation + protocole) ✅.
- Bloc `energy_calibration:` ajouté à `configs/hw_profile_f439zi.yaml` ✅.
- Manifeste de campagne `configs/energy_campaign_s50.yaml` (gabarit à remplir, S5002) ✅.

## Contraintes

- Aucun chiffre d'énergie ici (cadrage) — seulement paramètres de sonde/protocole. ✅
- Documenter honnêtement les limites d'isolation du banc (capteur non isolable → `"na"`). ✅
- Réutiliser `energy_capture.py` (ne pas réécrire la segmentation). ✅

## Vérification

```bash
grep -i "energy_calibration\|batterie" configs/hw_profile_f439zi.yaml
grep -i "PA8\|isolation\|MCU\|calibration" docs/sprints/sprint_50/S5001_cadrage_energie.md
python -c "import yaml;c=yaml.safe_load(open('configs/hw_profile_f439zi.yaml'));print(c['energy_calibration']['sampling_rate_hz'])"
```
