# S4607 — Tests + documentation + clôture

| Champ | Valeur |
|-------|--------|
| **Sprint** | 46 |
| **Priorité** | 🟡 Normal — verrouille l'honnêteté (N/A, 0-chiffre) et met à jour roadmap/triple_gap. |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 3h |
| **Dépendances** | S4602–S4606 ✅ · `tests/` conventions ✅ · `docs/roadmap_phase2.md`, `docs/triple_gap.md` |
| **Fichiers cibles** | `tests/test_s46_quant_moment.py`, `docs/roadmap_phase2.md`, `docs/triple_gap.md` |
| **Références** | S3912 (tests quant sweep) · S4207 (tests figures) · skill `graphify_sprint_update` |

---

## Contexte

Dernière tâche : tests de structure et d'honnêteté du harnais, puis mise à jour de la documentation
transverse (roadmap, triple_gap) et du graphe de connaissance. Le point sensible est que le sprint ne
doit **rien affirmer de chiffré** tant que les runs n'ont pas tourné — les tests le garantissent.

## Spec

### 1. `tests/test_s46_quant_moment.py`

| Test | Vérifie |
|------|---------|
| `test_json_schema` | les JSON `exp_S46_*` ont bien les 4 moments (EWC/TinyOL) ou le contexte N/A (HDC/Maha) |
| `test_three_moments_present` | `before`, `after`, `both` distincts pour EWC + TinyOL |
| `test_na_honest_hdc_maha` | HDC/Maha : `moments_3way == "N/A"` + `na_reason` non vide, **aucun chiffre 3-way** |
| `test_both_path_wiring` | le chemin `both` réutilise les poids QAT (mock `EWCMlpInt8Classifier` → `from_state_dict`) et ne réentraîne pas |
| `test_no_invented_numbers` | tant qu'un run n'a pas produit le JSON, les champs métriques valent `null` (pas 0) |
| `test_deterministic_seed` | deux runs seed 42 → mêmes métriques |

### 2. Mise à jour documentation

- **`docs/roadmap_phase2.md`** : section « ### Sprint 46 — … » (table `| Bloc | Tâches | Statut |
  Résultat attendu |`, message scientifique, liens triple gap, `→ Détail`). Optionnel : ligne macro.
- **`docs/triple_gap.md`** : paragraphe sous **Gap 3** — `**Renforcement Sprint 46 (…) — moments de
  quantification** :` citant `experiments/exp_S46_*`, message « le *moment* (before/after/both) et la
  *calibration* dominent la préservation de métrique ; `both` = seule variante fidèle au déploiement ».
- **`CLAUDE.md`** : ligne de statut Sprint 46.

### 3. Graphe

Invoquer `graphify_sprint_update` (évalue la pertinence d'un update du graphe).

## Format de sortie

```
tests/test_s46_quant_moment.py           # nouveaux tests
docs/roadmap_phase2.md                    # section Sprint 46
docs/triple_gap.md                        # § Gap 3 renforcement
CLAUDE.md                                 # ligne statut
```

## Contraintes

- Les tests doivent passer **avant** exécution réelle (structure/honnêteté), pas exiger de chiffres.
- Aucune régression : `pytest -k "quant or figures"` reste vert ; suite de collecte 0 erreur.
- Roadmap/triple_gap : suivre exactement le format des sprints 44/45 (voir S4500).

## Vérification

```bash
pytest tests/test_s46_quant_moment.py -q
pytest -k "quant or figures" -q          # 0 régression
grep -c "Sprint 46" docs/roadmap_phase2.md docs/triple_gap.md   # > 0 chacun
```

---

## Résolution (implémentée)

✅ **Implémenté.**

**Tests** `tests/test_s46_quant_moment.py` — **9 PASS** (conventions `test_s39_quant.py` :
skips honnêtes si artefact absent) :
- `test_json_schema_ewc` / `test_json_schema_tinyol` : les JSON `exp_S46_{ewc,tinyol}/*` portent
  les 4 moments {fp32,before,after,both} + les 3 `delta_*_vs_fp32`.
- `test_three_moments_present` : before/after/both présents et distincts ; `both` porte la note
  « fidèle au déploiement », `before` la note « borne haute ».
- `test_na_honest_hdc_maha` : `exp_S46_context/*` → `moments_3way=="N/A"`, `na_reason` non vide,
  **aucune clé 3-way** ni bloc `moments` (pas de cellule artificielle) ; couvre HDC + Mahalanobis.
- `test_both_path_wiring` : le chemin `both` **lit** fc1/fc2/fc3 d'un `EWCMlpInt8Classifier` mocké
  via `_weights_from_model` → `EWCHeadWeights.from_state_dict`, **sans réentraîner** (poids intacts) ;
  `test_both_path_multiclass_exportable` prouve que le head QAT multiclasse board (`EWCMlpMulticlassInt8`,
  S4608) s'exporte comme un head FP32 (fc3 = n_classes = 2).
- `test_no_invented_numbers` (+ `_template`) : une métrique vaut un float réel OU `null`, **jamais 0**
  sentinelle ; le squelette `_assemble` met `null` (pas 0) pour un moment non calculé.
- `test_deterministic_seed_qat_multiclass` : deux entraînements QAT seed 42 → poids identiques.

**Régression** : `pytest -k "quant or figures"` → **38 PASS, 0 régression**.

**Documentation** :
- `docs/roadmap_phase2.md` § Sprint 46 : blocs D (S4607 ✅) + E (S4608 ✅ board), ligne Statut →
  ✅ Sprint 46 implémenté (S4601–S4608).
- `docs/triple_gap.md` § Gap 3 : paragraphe Sprint 46 passé de « 📝 doc — spec » à **mesuré board**
  (F1 `both`, parité, latence, `.bss`, A/B `both` vs `after`).
- `CLAUDE.md` : ligne de statut Sprint 46.

**Graphe** : `graphify_sprint_update` invoqué en clôture.
