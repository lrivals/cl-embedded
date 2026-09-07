# S5310 — Tests, documentation et resynchronisation

| Champ | Valeur |
|-------|--------|
| **Sprint** | 53 |
| **Priorité** | 🟠 Important — clôture du sprint |
| **Statut** | 📝 Spec |
| **Durée estimée** | 2 h |
| **Dépendances** | S5301–S5309 — **sans carte** |
| **Fichiers cibles** | `tests/test_s53_energy.py` · `docs/context/int8_cost_benefit.md` · `docs/triple_gap.md` · `docs/roadmap_phase2.md` · `CLAUDE.md` |

## Spec

### 1. Tests — `tests/test_s53_energy.py`

| Test | Ce qu'il garantit |
|------|-------------------|
| `test_regression_schema` | Chaque JSON `exp_S53_rate_sweep/` porte `points[]`, `slope_ua_per_hz`, `r2`, `method` |
| `test_slope_positive_or_na` | Une pente non significative (< 2σ) ou `r² < 0.9` **doit** produire `"à mesurer"` + `na_reason`, jamais une énergie négative |
| `test_estimators_not_merged` | Les trois `method` (delta / régression / intégration) restent distincts dans `exp_S53_summary.json` ; aucun champ ne les moyenne |
| `test_counterbalance_verdict_computed` | Le `verdict` de S5301 est dérivé d'une règle codée, reproductible sur des données de test |
| `test_phase_windows_from_current` | La segmentation par courant refuse (N/A) si l'écart au nombre de bursts attendu dépasse 5 % |
| `test_derive_phase_windows_unchanged` | La fonction PA8 historique et ses tests S33 sont **intacts** |
| `test_na_honesty` | Aucun champ énergie ne vaut `0` là où il devrait valoir `"à mesurer"` ; toute `na_reason` est non vide |
| `test_no_hardcoded_numbers` | Garde AST sur `aggregate_s53_energy.py` et les nouveaux catalogues de figures |
| `test_s38_s48_values_loaded` | S5306/S5307 chargent `exp_S38_summary.json` / `exp_S48_summary.json`, ne les recopient pas |
| `test_gap2_preserved` | Toutes les latences des builds S53 (WFI, SYSCLK réduit) restent < 100 ms |
| `test_veille_ua_only_if_wfi` | `hw_profile_f439zi.yaml:veille_uA` reste `null` si le WFI n'a pas abouti |

Régression attendue : `pytest tests/test_s50_energy.py tests/test_energy_capture.py
tests/test_energy_delta.py tests/test_autonomy.py tests/test_figures_library.py`
(**81 PASS** vérifiés au 5 août 2026) **sans régression**.

Firmware : `make test` → **145 / 0 échec**, `.bss` défaut invariant **105 036 B**, sur chacun des
builds ajoutés (`-DUART_WFI_IDLE`, `-DSYSCLK_MHZ=90|45`).

### 2. Resynchronisation des docs en retard sur les données

Le Sprint 50 a laissé plusieurs documents plus pessimistes que les données qu'ils décrivent.
À reprendre **en fin de campagne**, une fois les résultats S53 connus :

| Fichier | Problème | Action |
|---------|----------|--------|
| `docs/context/int8_cost_benefit.md:21,55-58` | La ligne « Énergie / inférence » est « à statuer (banc LPM01A non posé) » alors que le banc a tourné et que `ratio_i_int8_fp32` existe. La question l. 57 (« l'INT8 réduit-il malgré tout les µJ ? ») est présentée comme ouverte. | Statuer avec les données S5304, ou expliciter ce qui reste ouvert et pourquoi |
| `src/figures/catalogs/energy_real.py:5-21` | Docstring : « tant que le banc X-NUCLEO-LPM01A n'a pas tourné » | Resynchroniser |
| idem `:184`, `:225` | Encarts E2/E3 : « banc LPM01A non posé », « dépend des µJ réels — S5002 » | Corriger (E3 est chiffrée depuis S50) |
| idem `:60-62` | `NA_ENERGY_TEXT` défini et **jamais utilisé** | Utiliser ou supprimer |
| `docs/sprints/sprint_50/S5008_handoff_mesures.md:§1` | « Énergie mesurée ❌ aucune » — dépassé pour le courant | Ajouter un renvoi vers S53 |
| `experiments/exp_S50_energy/*.json` | `energy_na_reason` affirme que la cause de l'anomalie « n'est pas établie » | Mettre à jour selon le verdict S5301 |

### 3. `TODO` à clore ou à requalifier

| TODO | Emplacement | Condition de clôture |
|------|-------------|----------------------|
| `TODO(dorra)` × 5 — fréquence d'échantillonnage et calibration LPM01A | `scripts/energy_capture.py:13,78,102,350,478` | Clore si la campagne établit la fréquence et la plage retenues, avec leur justification mesurée |
| `TODO(fred)` — `inference_period_s` | `configs/energy_campaign_s50.yaml:20` | Requalifier : S5308 fournit le balayage, **le choix du profil d'usage industriel reste à valider** — le TODO ne se clôt pas tout seul |
| `TODO(fred)` — capacités batterie réalistes | `configs/hw_profile_f439zi.yaml:115` | Idem, reste ouvert |
| `TODO(arnaud)` — `efficacite.fp32/int8 = 0.3` | `configs/hw_profile_f439zi.yaml:31-32` | Affinable par cross-check avec `uj_per_us_compute` (S5304) — le tenter |

### 4. Documents de synthèse

- `docs/context/energy_measurement.md` — **nouveau** : la méthodologie de banc du sprint
  (contre-balancement, régression de cadence, trois estimateurs et leurs domaines de
  validité, contraintes matérielles). C'est le pendant énergie de
  `docs/context/ram_measurement.md`.
- `docs/triple_gap.md` — enrichir le § Gap 2 (latence à fréquence réduite, marge conservée)
  et le § Gap 3 (verdict énergie de l'INT8, arbitrage RAM↔énergie du bit-packing).
- `docs/roadmap_phase2.md` — Sprint 53 ✅ + bilan.
- `CLAUDE.md` — bloc de statut Sprint 53.

### 5. Clôture

1. Mettre à jour chaque `S53xx_*.md` avec son statut réel et ses chiffres mesurés.
2. Invoquer le skill **`graphify_sprint_update`**.
3. Fournir le message de commit complet du sprint.

## Critères d'acceptation

- [ ] `test_s53_energy.py` PASS, suite énergie+figures sans régression (≥ **81 PASS**).
- [ ] `make test` 145 / 0 échec, `.bss` défaut invariant sur tous les builds.
- [ ] Aucune doc ne décrit un état antérieur aux données présentes dans `experiments/`.
- [ ] Chaque `TODO` listé est clos **ou** requalifié explicitement — aucun laissé
      silencieusement en l'état.
- [ ] `docs/context/energy_measurement.md` créé, sans chiffre en dur (généré ou chargé).
