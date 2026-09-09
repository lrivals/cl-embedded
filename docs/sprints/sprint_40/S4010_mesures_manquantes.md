# S4010 — Registre des mesures manquantes de l'article EWC

| Champ | Valeur |
|-------|--------|
| **Sprint** | 40 (refonte) |
| **Statut** | ✅ Registre tenu — **aucune valeur écrite tant que la mesure n'a pas eu lieu** |
| **Source de vérité** | `experiments/exp_S40_article_metrics/summary.json`, clé `missing` (28 cellules) |
| **Régénération** | `python scripts/aggregate_article_ewc.py` puis `print(sorted(d["missing"]))` |

Ce document ne contient **aucun chiffre estimé**. Chaque entrée renvoie à la cellule de l'agrégat qui
porte `null` ou le sentinel `"à mesurer"`, et à la raison — mesurée, pas supposée — pour laquelle elle
est vide. Les remplir consiste à relancer la campagne citée ; rien d'autre n'est à modifier, l'agrégat
et les figures se mettant à jour d'eux-mêmes.

## Distinguer deux natures de « manquant »

Les 28 cellules ne demandent pas le même travail, et les confondre serait trompeur :

* **16 cellules relèvent du banc** — il faut la carte, et pour l'énergie la sonde X-NUCLEO-LPM01A ;
* **12 cellules sont N/A par construction** (RAM PC) : le profil PC est un proxy `tracemalloc` FP32,
  il n'a ni `.data` ni `.bss`, et le gain INT8 y est analytique. Ce sont des N/A définitifs, portant
  déjà leur `na_reason` depuis le Sprint 49 ; elles n'appellent aucune campagne.

## Cellules à mesurer au banc

| Mesure | Nature | Cellules d'agrégat | Trace / raison mesurée |
|--------|--------|--------------------|------------------------|
| Grille board v2 `q15` | banc | `{ds}.recovery_board.q15_{frozen,online}_f1_faulty` | `S4002` : 4 cellules `per_channel` mesurées sur 12 ; le `q15` n'a jamais été flashé |
| A/B `int8_legacy` sous kernel v2 | banc | `{ds}.recovery_board.int8_legacy_{frozen,online}_f1_faulty` | `S4002` ; l'effondrement legacy n'est mesuré qu'avec le kernel v1 (Sprint 36) |
| Fichiers de parité S40 | banc | — (attendus par `TestS40Parity`, 2 skips) | `exp_S40_board_v2/parity_{scheme}_frozen_{ds}.json` jamais produits |
| Énergie par mise à jour CL | banc | `energy.estimators.delta_wfi.cells.ewc_{fp32,int8}_energy_uj_per_update` | campagne d'inférence seule ; isoler la MAJ exige la même mesure avec `--update` |
| Énergie par MAJ, par politique | banc | `energy.estimators.policy_update.cells.{ds}_energy_uj_per_update` | `exp_S53_policy_energy/` : régressions **non publiables** (r² 0,696 et 0,366 < 0,9) — le courant ne suit pas une droite en cadence, la pente ne mesure donc pas un coût marginal |
| Profil énergie par phase | banc | non agrégée (estimateur S5305 incomplet) | `exp_S53_phase_profile/` : marqueur **1 bit** — `startup`/`acquisition`/`inference` partagent le niveau actif ; une granularité 4 phases exige un encodage multi-bit (PA8 + PA9) absent du firmware |
| Énergie sub-INT8 packé vs non packé | banc | non agrégée | S5307 non exécuté — relierait directement A7 (gain RAM packé) et A9 (énergie) |
| `by_component` MCU / périphériques | banc | non agrégée | S5308 non exécuté ; demande un câblage séparé des domaines |

> **Note.** La cellule **90 MHz du balayage de fréquence n'est plus manquante** : elle a été recalculée
> par `--refit` et vaut 192,64 µJ/inférence avec r² = 0,999. Elle figurait comme « à mesurer » dans les
> notes antérieures du dépôt ; l'agrégat la lit désormais comme mesurée.

## Limites documentées (pas des mesures manquantes)

Ces points ne se comblent pas par une campagne : ce sont des bornes de portée, à énoncer plutôt qu'à
combler.

| Limite | Constat |
|--------|---------|
| Oubli non éprouvé sur les deux jeux de l'article | `exp_S36` donne AF ≤ 0,01 sur 3 tâches : le scénario CL est trop court pour stresser la régularisation. L'oubli catastrophique est mesuré sur **CWRU** (`exp_S54`, 0,8475 EWC / 0,8578 naïf), jeu qui ne recouvre pas les nôtres. Les deux constats sont reportés séparément (bloc `context` de l'agrégat), jamais fondus. |
| RAM PC | Proxy `tracemalloc` FP32 : ni `.data`, ni `.bss`, pas de pic de MAJ. Le gain INT8 y est analytique — seule la carte le mesure. |
| Score système composite (Sprint 51) | Spec `S5103` jamais implémentée. L'agrégat S4008 en fournit désormais toutes les entrées (RAM, latence, énergie, paramètres/MACs, métrique) ; il ne manque que la fonction de score et sa pondération, qui est un choix à arbitrer, pas une mesure. |
