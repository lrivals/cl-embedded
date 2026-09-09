# Sprint 50 — Mesure d'énergie réelle (carte ENAC + LPM01A) + coût latence INT8

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 50 |
| **Semaine** | Indicative — à confirmer (dès récupération carte ENAC) |
| **Statut** | ✅ **Sprint complété board réelle** — volet latence INT8 (S5004–S5007) **et volet énergie exécuté (S5008, 4–5 août 2026, banc LPM01A monté)**. S5001 cadrage ✅ · **S5002 campagne 8/8 cellules mesurées** (courant moyen à cadence imposée ; µJ/inférence **N/A mesuré** — référence de repos non exploitable) · **S5003 autonomie chiffrée** (43,1 h → 39,0 h sur 2000 mAh) · S5004 breakdown latence INT8 · S5005 coût/bénéfice · S5006 figures (**6 PNG**) + notebook · S5007 tests. **Réponse Gap 3 : l'INT8 ne réduit PAS la consommation** (HDC INT8 +7,1 % de courant) — il reste justifié par la RAM ÷4. Restent ouverts : µJ/inférence (levier `WFI`), cause de l'écart repos↔flux, `by_component` MCU/périph. |
| **Priorité globale** | 🔴 Critique — répond à l'action 🔴 du CR (« récupérer la carte à l'ENAC pour mesurer l'énergie électrique ») et à la tâche 🟡 (« détailler le coût de latence INT8 »). Depuis le Sprint 33, **toute l'infra énergie existe mais porte `"à mesurer"`** (LPM01A jamais posé) ; ce sprint **pose la sonde** et remplit depuis des CSV réels. |
| **Durée estimée totale** | ~24h (cadrage/setup ~4h · capture énergie ~5h · autonomie ~2h · breakdown latence INT8 ~5h · coût/bénéfice ~2h · figures+notebook ~4h · tests+docs ~2h) |
| **Dépendances** | **Carte NUCLEO-F439ZI récupérée à l'ENAC** · X-NUCLEO-LPM01A (PowerShield) · `scripts/energy_capture.py` ✅ (chaîne CSV→`segment_by_phase`→`integrate_energy_uj` déjà **débloquée et testée** S33) · `src/evaluation/autonomy.py` ✅ · `configs/hw_profile_f439zi.yaml` ✅ · marqueurs GPIO `ENERGY_MARKERS` (PA8) ✅ · `src/evaluation/hw_cost_model.py`/`compute_cost.py` ✅ |

## Contexte et motivation

Le CR du 16 juillet acte : **récupérer la carte à l'ENAC** et mesurer l'**énergie électrique consommée par
inférence**, en **isolant les mesures par composant** (MCU seul / périphériques / capteurs), via le **setup
logiciel posté sur le lien partagé**.

Le Sprint 33 a bâti toute la chaîne : `energy_capture.py` (`segment_by_phase`, `integrate_energy_uj`,
`derive_phase_windows` depuis la colonne de sync PA8 du CSV LPM01A), `autonomy.py` (I_moy, Autonomie_h),
`configs/hw_profile_f439zi.yaml` (capacités batterie), marqueurs GPIO PA8. **Décision S33** : « aucun chiffre
inventé » → tous les champs énergie/autonomie portent la valeur littérale `"à mesurer"`, le code restant prêt
à re-remplir depuis un CSV réel. **Ce sprint lève ce blocage** : la sonde est posée, les CSV réels sont
capturés, les champs sont remplis.

Second volet (tâche 🟡 CR) : **détailler le coût de latence INT8**. Le CR note que le processeur est FP32 (FPU)
→ convertir vers INT8 implique des **étapes supplémentaires de déquantification/requantification**. Il faut
**détailler ces étapes et quantifier leur coût en cycles** (segments DWT), pour étayer l'analyse coût/bénéfice
(le gain RAM ÷4 vaut-il la légère chute d'accuracy + le surcoût latence ?) et le **paradoxe latence FPU**
observé au Sprint 29 (INT8 ne réduit pas la latence sur Cortex-M4 FPU).

## Décisions de cadrage (utilisateur, CR du 16 juillet 2026)

- **Carte récupérée à l'ENAC** ; setup logiciel = lien partagé (LPM01A / STM32 PowerShield).
- **Isoler par composant** autant que possible : MCU seul / périphériques / capteurs.
- **Cible = énergie par inférence (µJ)** pour chaque modèle et configuration (fp32 / int8).
- **Résoudre `TODO(dorra)`** : fréquence d'échantillonnage + calibration LPM01A (ouvert depuis S33).
- **Coût latence INT8 = mesure DWT réelle** (pas proxy) : breakdown des étapes dequant/requant.
- **Analyse coût/bénéfice explicite** : RAM ÷4 vs Δaccuracy vs Δlatence/énergie.
- **Justification biblio** (CR §5–6) : distinguer ce qui est **reproduit** de la littérature (Ravaglia ÷4 <0.26 %,
  Capogrosso PTQ/QAT, Zhu/Lin QAS, Benatti, Giménez) de la **contribution propre** (EWC sur PRONOSTIA, compromis
  mesuré sur carte réelle) — à porter dans le rapport (Sprint 41), tracé ici.
- **Aucun chiffre inventé** : `"à mesurer"` reste tant que la sonde n'a pas tourné.
- **Langue** : français.

## Nœud honnête : « à mesurer » → mesuré, sans rien inventer entre-temps

Tant que le CSV LPM01A réel n'existe pas, **aucun champ énergie n'est rempli** (règle S33 maintenue). Ce sprint
distingue trois états explicites dans les JSON : `"à mesurer"` (sonde non passée), valeur mesurée (CSV réel
segmenté), `"na"` (non applicable). Le **coût latence INT8** est mesuré au DWT sur carte — il **confirme
probablement le paradoxe FPU S29** (RAM ÷4 sans gain de latence, voire surcoût de dequant), ce qui est un
**résultat honnête**, pas un échec : c'est précisément ce que l'analyse coût/bénéfice doit exposer.

## Tâches

### Bloc A — Setup & cadrage

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S5001 | **Cadrage mesure énergie** : récupération carte ENAC, setup logiciel (lien partagé), **plan d'isolation par composant** (MCU / périphériques / capteurs), fréquence d'échantillonnage + calibration LPM01A (résout `TODO(dorra)`) ; protocole de campagne (phases balisées PA8) | 🔴 | `docs/sprints/sprint_50/S5001_cadrage_energie.md`, MAJ `configs/hw_profile_f439zi.yaml` (bloc calibration) | ✅ Cadrage (pose sonde différée) |

### Bloc B — Capture & autonomie

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S5002 | **Capture énergie réelle** : `run_s50_energy.py --manifest` sur CSV LPM01A réels → **énergie par inférence (µJ)** pour 4 modèles × {fp32, int8}, décomposée par phase (idle/inférence/MAJ) et par composant ; remplace `"à mesurer"` dans les JSON | 🔴 | `experiments/exp_S50_energy/` (+ MAJ `experiments/exp_S33_energy/`) | ✅ Chaîne + placeholders (capture différée) |
| S5003 | **Autonomie recalculée** depuis les mesures réelles (`I_moy = Σ(I·t)/T_cycle`, `Autonomie_h = Capacité/I_moy` ← `hw_profile_f439zi.yaml`) | 🟠 | `src/evaluation/autonomy.py` (run), `experiments/exp_S50_energy/autonomy.json` | ✅ Câblé + placeholder (chiffres via S5002) |
| S5008 | **Exécution de la campagne de mesure** : passation puis exécution board réelle ; protocole delta invalidé par deux constats de banc mesurés (1re acquisition biaisée ~+8 mA ; référence de repos plus haute que tous les régimes de flux) → remplacé par le **courant moyen à cadence imposée** ; grille 8/8 + autonomie + figure E6 | 🔴 | `docs/sprints/sprint_50/S5008_handoff_mesures.md`, `scripts/run_s50_board_current.py`, `experiments/exp_S50_energy/` | ✅ Mesuré board réelle |

### Bloc C — Latence INT8 & coût/bénéfice

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S5004 | **Coût latence INT8 détaillé** : breakdown cycle-level (segments DWT) des étapes déquant/requant ajoutées par la quantification dans le pipeline d'inférence ; surcoût vs FP32 ; étaye le paradoxe latence FPU (S29) | 🟡 | `docs/context/int8_latency_breakdown.md`, `experiments/exp_S50_int8_latency/` | 📝 Doc |
| S5005 | **Analyse coût/bénéfice** : tableau décision RAM ÷4 vs Δaccuracy vs Δlatence/énergie — « le gain RAM vaut-il le coup ? » ; distinction reproduit-littérature / contribution propre (CR §5) | 🟡 | `docs/context/int8_cost_benefit.md` | 📝 Doc |

### Bloc D — Assemblage & clôture

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S5006 | **Figures + notebook énergie** : énergie/inférence par modèle × encodage, décomposition par composant, autonomie, breakdown latence INT8 ; badges **mesuré / à-mesurer**, N/A gris, garde AST 0-chiffre-en-dur | 🟠 | `src/figures/catalogs/energy_real.py`, `docs/figures/energy_real/`, MAJ `notebooks/cl_eval/energy_cost/comparison.ipynb` | 📝 Doc |
| S5007 | **Tests + docs + clôture** : `test_s50_energy.py` (segmentation CSV, µJ > 0 seulement si CSV réel, badge « à mesurer » honnête, 0-chiffre-en-dur), MAJ roadmap/triple_gap (§ Gap 2/3 énergie), `graphify_sprint_update` | 🟡 | `tests/test_s50_energy.py`, `docs/roadmap_phase2.md`, `docs/triple_gap.md` | 📝 Doc |

## Ordre d'exécution recommandé

```
S5001 (setup ENAC + calibration LPM01A + plan isolation)   ← requiert carte
   │
   ▼
S5002 (capture CSV réels → énergie/inférence µJ) ──► S5003 (autonomie)
   │
   ▼
S5004 (breakdown latence INT8 DWT) ──► S5005 (coût/bénéfice)
   │
   ▼
S5006 (figures + notebook)
   │
   ▼
S5007 (tests + roadmap + triple_gap + graphify)
```

Tout le bloc dépend de la **carte à l'ENAC + sonde LPM01A**. Le code (S33) est prêt ; ce sprint fournit les
**données réelles** et l'analyse.

## Sources de données (Sprint 50)

| Source | Rôle |
| ------ | ---- |
| CSV LPM01A (campagne réelle) | Courant/temps → segmentation PA8 → énergie µJ |
| `configs/hw_profile_f439zi.yaml` | Capacités batterie, calibration (MAJ S5001) |
| `exp_S49_ram/` (Sprint 49) | RAM totale, pour l'analyse coût/bénéfice croisée RAM↔énergie |
| Modèles board EWC/HDC/TinyOL/Maha × {fp32,int8} | Configurations mesurées |

## Livrables

1. `docs/sprints/sprint_50/S5001_cadrage_energie.md` + MAJ `configs/hw_profile_f439zi.yaml` (calibration).
2. `experiments/exp_S50_energy/` — énergie/inférence µJ par modèle × encodage × composant (+ `exp_S33_energy/` rempli).
3. `experiments/exp_S50_energy/autonomy.json` — autonomie recalculée.
4. `docs/context/int8_latency_breakdown.md` + `experiments/exp_S50_int8_latency/` — coût latence INT8 (DWT).
5. `docs/context/int8_cost_benefit.md` — analyse coût/bénéfice.
6. `src/figures/catalogs/energy_real.py` → `docs/figures/energy_real/` + notebook énergie.
7. `tests/test_s50_energy.py` + MAJ roadmap/triple_gap.
