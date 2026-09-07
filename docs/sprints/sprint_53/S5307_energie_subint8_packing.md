# S5307 — Énergie sub-INT8 : le bit-packing paie-t-il son dépacking ?

| Champ | Valeur |
|-------|--------|
| **Sprint** | 53 |
| **Priorité** | 🟠 Important |
| **Statut** | 📝 Spec |
| **Durée estimée** | 3 h (1 h pilote + 1 h 30 banc + 30 min analyse) |
| **Dépendances** | S5304 (méthode) · `scripts/run_s48_board_depth.py` ✅ · `experiments/exp_S48_summary.json` ✅ |
| **Fichiers cibles** | `scripts/run_s53_depth_energy.py` · `experiments/exp_S53_depth_energy/` |
| **Références** | Sprint 48 (S4804–S4807) · Sprint 47 (taxonomie profondeur) |

## Contexte

Le Sprint 48 a mesuré sur carte les 12 cellules packé/non-packé × {INT4, ternaire,
binaire} × {Monitoring, Pronostia}, avec **parité board↔émulateur = 1.000** partout. Deux
résultats nets en sont sortis :

- Le gain RAM sub-INT8 est **réel mais conditionnel au bit-packing** : sans packing,
  `.bss` est **invariant** (conteneur `int8_t`), avec packing il descend jusqu'à ÷8
  (Monitoring 336 B INT4 → 572 B binaire, selon le mode).
- Le dépacking coûte **≈ +55 µs** (67–70 µs non-packé → 123–130 µs packé), sans menacer le
  Gap 2.

**La dimension manquante est l'énergie.** La question est nette et n'a pas de réponse
évidente : *le dépacking, qui double presque la latence, coûte-t-il plus d'énergie que la
RAM ÷8 n'en économise ?* Sur un MCU sans DVFS mémoire, la RAM statique consomme qu'on la
lise ou non — l'intuition dit que le packing **coûte** de l'énergie et n'en économise pas,
ce qui ferait du bit-packing un arbitrage **RAM contre énergie**, pas un gain net. Il faut
le mesurer.

## Spec

### 1. Cellules

**12 cellules** (packé/non-packé × 3 profondeurs × 2 datasets), réductibles à **6** si le
temps de banc manque (un seul dataset — retenir **Pronostia**, k=5, le cas le plus proche
d'un déploiement réel).

Builds, repris strictement de `run_s48_board_depth.py:112+` :

```
EXTRA_CFLAGS="-D<EWC_INT4|EWC_INT2|EWC_INT1> [-DEWC_INTx_PACKED] -DEWC_SUBINT8_WEIGHTS_PROVIDED"
```

⚠️ **6 flashes minimum** — c'est l'axe le plus coûteux en manipulations de cavalier JP5.
Le prévoir en fin de journée de banc, quand le protocole est rodé.

La garde `#error` de cohérence header↔build (`pipeline.c:39-46`) protège déjà contre le
débordement `memcpy` int8→packé : ne pas la contourner.

### 2. Mesure

Méthode S5304 (balayage de cadence) → pente, donc `energy_uj_per_inference` par cellule.
Le forward sub-INT8 est **frozen** (poids figés PC, pas d'update online) : la mesure est
donc propre, sans mise à jour parasite.

Grandeurs dérivées, calculées :

- `delta_uj_packing` = `E(packé) − E(non-packé)`, par profondeur et dataset.
- `bss_saved_by_packing` — **rechargé** depuis `exp_S48_summary.json`, jamais recopié.
- `uj_per_byte_saved` = `delta_uj_packing / bss_saved_by_packing` — la **métrique
  d'arbitrage** : combien de µJ par inférence coûte chaque octet de RAM économisé.
- Cohérence croisée : `delta_uj_packing` doit être compatible avec
  `55 µs × uj_per_us_compute` (issu de `slope_vs_latency.json`, S5304). Un écart important
  signalerait que le dépacking n'a pas le même profil de consommation qu'un calcul FPU
  ordinaire — ce qui serait en soi un résultat intéressant.

### 3. Sortie

`experiments/exp_S53_depth_energy/{dataset}_{weight_bits}_{packed|unpacked}.json` — schéma
de cellule commun, plus `packing_tradeoff.json` portant les grandeurs dérivées et le
tableau d'arbitrage RAM ↔ énergie.

## Critères d'acceptation

- [ ] Au moins 6 cellules mesurées, 0 erreur CRC, parité board↔émulateur **conservée à
      1.000** (vérifier avant la mesure d'énergie : un build faux invaliderait tout).
- [ ] `bss_saved_by_packing` chargé depuis S48, jamais recopié ni recalculé.
- [ ] `uj_per_byte_saved` calculé par profondeur, avec le signe explicité dans le texte
      (positif = le packing coûte de l'énergie).
- [ ] Contrôle de cohérence avec `uj_per_us_compute` de S5304 rapporté.
- [ ] Si `delta_uj_packing` n'est pas significatif face à la dispersion du banc → N/A
      honnête avec la valeur relevée, **pas** un « gain nul » affirmé.

## Résultat attendu et lecture honnête

L'issue la plus probable est que **le packing coûte de l'énergie** (+55 µs de travail réel)
et n'en économise aucune (la SRAM statique consomme indépendamment du remplissage). Si
c'est ce qui sort, la conclusion à écrire n'est pas « le bit-packing est mauvais » mais :

> Le bit-packing est un arbitrage **RAM contre énergie et latence**, pas un gain net. Il se
> justifie exactement quand la RAM est le facteur limitant — ce qui est précisément le cas
> du Gap 2 sur cette classe de cible — et se déjustifie sur une plateforme où la contrainte
> est l'autonomie.

C'est une conclusion plus forte, et plus utile, qu'un gain qu'on n'aurait pas mesuré.
