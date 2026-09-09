# S5003 — Autonomie recalculée depuis les mesures réelles

| Champ | Valeur |
|-------|--------|
| **Sprint** | 50 |
| **Priorité** | 🟠 Important — traduit les µJ mesurés en autonomie exploitable (rapport). |
| **Statut** | ✅ **Autonomie chiffrée 8/8 cellules** depuis le courant mesuré (S5002, board réelle 2026-08-04/05) |
| **Durée estimée** | 2h |
| **Dépendances** | S5002 (µJ réels) · `src/evaluation/autonomy.py` ✅ · `scripts/run_s50_energy.py` ✅ · `configs/hw_profile_f439zi.yaml` ✅ (capacités batterie) |
| **Fichiers cibles** | `experiments/exp_S50_energy/autonomy.json` (via `autonomy.py`) |
| **Références** | S33 (`autonomy.py`, `I_moy`/`Autonomie_h`) |

## Contexte

`autonomy.py` existe (S33) mais tournait sur `"à mesurer"`. Avec les µJ réels de S5002, recalculer
l'autonomie réelle par configuration.

## Spec

- `I_moy = Σ(I·t)/T_cycle` sur un cycle représentatif (idle + inférence + MAJ selon duty cycle documenté).
- `Autonomie_h = Capacité / I_moy`, capacités ← `hw_profile_f439zi.yaml:batterie`.
- Par (modèle, encodage) : `I_moy`, `Autonomie_h`, sensibilité au duty cycle (inférence rare vs continue).
- RAM profiling de la routine (`profile_memory.py --model autonomy`) déjà à ~208 B (S33) — invariant.

## Format de sortie

`experiments/exp_S50_energy/autonomy.json` : `[model][encoding]` → `{I_moy_mA, autonomie_h, duty_cycle}`.

## État actuel (implémenté, en attente matériel)

> **2026-07-24** : `scripts/run_s50_energy.py` produit déjà `experiments/exp_S50_energy/autonomy.json`
> à chaque campagne, en réutilisant `autonomy.average_current_ma` + `sweep_capacities`
> (capacités batterie ← `hw_profile_f439zi.yaml`, jamais en dur). Tant que S5002 n'a pas de
> µJ réels, `i_moy_ma` et `autonomy_h_by_mah` restent `"à mesurer"` (aucun courant fabriqué).

**Déblocage clé** : l'autonomie exige `phase_durations_s`, que la chaîne S33 n'écrivait jamais
(clé lue par `profile_memory.py` mais sans producteur). Le driver S50 la dérive des **fenêtres
PA8 réelles** (`phase_durations_from_windows`) — donc dès qu'un CSV réel est fourni (S5002),
`autonomy.json` se remplit automatiquement, sans étape manuelle. Le `duty_cycle`
(`inference_period_s`) vient du manifeste (sensibilité inférence rare vs continue).

## Résultat (2026-08-05) — autonomie chiffrée, par une autre voie que prévue

La voie prévue ici (`I_moy = Σ(I·t)/T_cycle` depuis les µJ par phase) est **restée
inaccessible** : la campagne S5002 n'a pas pu produire de profil par phase (ni voie de
synchronisation, ni mode dynamique). Mais elle a produit **mieux pour cet usage** — le
courant moyen directement mesuré. `I_moy` n'a donc pas à être reconstruit : il **est** la
mesure.

`write_autonomy` (`scripts/run_s50_energy.py`) essaie désormais les voies dans cet ordre :

1. profil par phase → `autonomy.average_current_ma` (voie d'origine, inutilisée ici) ;
2. **courant moyen mesuré** → `_i_moy_from_current` (voie retenue) ;
3. protocole delta → `autonomy.average_current_ma_from_delta` (ajoutée au passage,
   `I_moy = I_repos + charge_marginale / période`) ;
4. sinon `"à mesurer"`.

`sweep_capacities` et `load_battery_capacities` sont inchangés (capacités ← profil HW).

### Autonomie mesurée — extrêmes de la grille (batterie 2000 mAh)

| cellule | I_moy mesuré | autonomie |
|---|---|---|
| Maha INT8 (la plus sobre) | 46,373 mA | 43,1 h |
| HDC INT8 (la plus gourmande) | 51,220 mA | 39,0 h |

Soit **~10 % d'autonomie d'écart** entre le modèle le plus sobre et le plus gourmand de la
grille — l'ordre de grandeur utile pour arbitrer, là où les µJ/inférence manquants n'auraient
rien changé à la décision.

> **Réserve de lecture, portée par le JSON lui-même** : chaque entrée a `duty_cycle: null` et
> un champ `regime_mesure` qui rappelle que cette autonomie est celle du **régime réellement
> mesuré** — flux continu à 100 Hz, carte jamais endormie. Ce n'est **pas** l'autonomie d'un
> déploiement duty-cyclé, qui supposerait une mise en sommeil que le firmware ne fait pas.
> Afficher le `inference_period_s: 1.0` du manifeste à côté d'un courant mesuré à 100 Hz
> aurait laissé croire l'inverse : c'est pourquoi il est explicitement neutralisé.

## Contraintes

- Ne recalcule que si S5002 a fourni des grandeurs réelles ; sinon `"à mesurer"`.
- Capacités batterie depuis le profil HW (pas en dur).

## Vérification

```bash
python -c "import json;d=json.load(open('experiments/exp_S50_energy/autonomy.json'));print(list(d))"
python -m pytest tests/test_autonomy.py -q
```
