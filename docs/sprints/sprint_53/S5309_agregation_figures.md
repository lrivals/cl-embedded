# S5309 — Agrégation, figures et notebook

| Champ | Valeur |
|-------|--------|
| **Sprint** | 53 |
| **Priorité** | 🟠 Important |
| **Statut** | 📝 Spec |
| **Durée estimée** | 3 h |
| **Dépendances** | S5301–S5308 (données produites) — **sans carte** |
| **Fichiers cibles** | `scripts/aggregate_s53_energy.py` · `src/figures/catalogs/energy_real.py` · `notebooks/cl_eval/energy_cost/comparison.ipynb` |
| **Références** | `scripts/aggregate_ram.py` (patron) · `src/figures/registry.py` (registre S4201) |

## Spec

### 1. Agrégateur — lecture seule

`scripts/aggregate_s53_energy.py`, sur le patron de `aggregate_ram.py` et
`aggregate_sprint48.py` : il **lit** les JSON produits et n'en réécrit aucun.

Sortie `experiments/exp_S53_summary.json`, indexé `[axe][cellule][méthode]` où
`axe ∈ {rate_sweep, freq_sweep, policy, depth, components}`.

**Règle structurante** : les trois estimateurs de `energy_uj_per_inference` — **delta**
(S5302), **régression** (S5304), **intégration du profil** (S5305) — sont conservés
**côte à côte**, jamais fusionnés ni moyennés. Un bloc `estimator_comparison` porte leur
écart relatif : c'est le contrôle de validité de toute la campagne, et un résultat de
méthodologie en soi.

Grandeurs dérivées à recalculer (jamais recopier) :
`uj_per_us_compute`, `uj_per_uart_frame`, `energy_uj_per_update`, `uj_per_byte_saved`,
`energy_saved_vs_always_pct`, autonomies.

### 2. Figures — catalogue `energy_real` étendu

Le catalogue existe (`@register_catalog("energy_real")`, 5 figures E1–E5). L'étendre plutôt
que d'en créer un second, pour que les badges et la palette restent cohérents.

**Figures existantes à débloquer** :

- **E1** (`e1_energie_par_inference`) — grise aujourd'hui car
  `summary["per_model"][m][enc]["energy_uj_per_inference"]` vaut `"à mesurer"`. Doit passer
  en couleur si S5304 aboutit, avec le badge de méthode.
- **E5** (`e5_cout_benefice`) — 1 barre sur 3 grise (la barre Énergie lit `ratio_int8_fp32`,
  toujours `"à mesurer"`). Doit se compléter.
- **E2** / **E3** — encarts internes à corriger : ils disent encore « banc LPM01A non posé »
  (l. 184) et « dépend des µJ réels — S5002 » (l. 225) alors que E3 est chiffrée depuis le
  Sprint 50. Utiliser `NA_ENERGY_TEXT` (défini l. 60-62 et **inutilisé**) ou le supprimer.

**Figures à ajouter** :

| Fig | Contenu | Source |
|-----|---------|--------|
| E6 | Droite `pente vs latence DWT` sur les 8 cellules, avec µJ/µs et µJ/trame en annotation | `exp_S53_rate_sweep/slope_vs_latency.json` |
| E7 | Énergie par inférence **vs fréquence** (180/90/45 MHz) + marge Gap 2 en second axe | `exp_S53_freq_sweep/` |
| E8 | Économie du gate : 4 politiques × {MAJ évitées, µs, mA, autonomie h}, F1 en annotation | `exp_S53_policy_energy/economy_energy.json` + `exp_S38_summary.json` |
| E9 | Arbitrage packing : `.bss` économisé (axe X) vs µJ ajoutés (axe Y), 6–12 points | `exp_S53_depth_energy/packing_tradeoff.json` + `exp_S48_summary.json` |
| E10 | Comparaison des 3 estimateurs de µJ/inférence, par cellule | `exp_S53_summary.json:estimator_comparison` |

Contraintes du catalogue à respecter : badges **mesuré-board** / **à-mesurer** /
**modélisé**, N/A en gris (`NA_GRAY`), valeurs chargées via `load_experiment` /
`metric_or_na` — **0 chiffre en dur**, la garde AST de `test_figures_library.py` doit rester
verte.

### 3. Notebook

Étendre `notebooks/cl_eval/energy_cost/comparison.ipynb` d'une section Sprint 53 : galerie
FR commentée, tableaux chargés depuis `exp_S53_summary.json`, aucune valeur saisie.
Exécution `nbconvert` sans erreur.

## Critères d'acceptation

- [ ] `exp_S53_summary.json` produit, agrégateur **lecture seule** (aucun JSON source
      modifié).
- [ ] Les 3 estimateurs coexistent, avec leur écart calculé ; aucun n'est moyenné avec un
      autre.
- [ ] E1 et E5 cessent d'être grises si les données le permettent ; sinon elles restent
      grises **avec la raison mesurée**, jamais avec une valeur inventée.
- [ ] Encarts E2/E3 resynchronisés avec l'état réel des données.
- [ ] 5 nouvelles figures produites sous `docs/figures/energy_real/`.
- [ ] `test_figures_library.py` PASS, garde AST « 0 chiffre en dur » verte.
- [ ] `nbconvert` OK.
