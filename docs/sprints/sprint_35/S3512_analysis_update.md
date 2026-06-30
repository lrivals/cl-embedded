# S3512 — Mise à jour de l'analyse des résultats

| Champ | Valeur |
|-------|--------|
| **Sprint** | 35 |
| **Priorité** | 🟡 Important — « mets à jour l'analyse des résultats » |
| **Statut** | ⬜ À démarrer |
| **Durée estimée** | 3h |
| **Dépendances** | S3503, S3508, S3510 (heatmaps), S3505 (RAM) |
| **Fichiers cibles** | `docs/datasets_analysis.md` (ou `docs/sprints/sprint_35/S3512_analysis_update.md`), `docs/triple_gap.md` (Gap 1) |

---

## Contexte

Une fois les 12 heatmaps produites, écrire l'interprétation : que coûte/rapporte le choix
des features, et la métrique accuracy vs F1.

## Spec

Rédiger une analyse couvrant :

1. **Impact features × modèle** : pour chaque modèle, gain F1 `best`/`all` vs `5feat` ; quels
   datasets souffrent le plus du 5-feat (ex. pronostia natif 13, monitoring natif 4).
2. **F1 vs accuracy** : où l'accuracy masque l'effondrement de la classe `faulty`
   (lien Sprint 26, `F1_MC=0.243`).
3. **Coût board** : RAM (`.bss`, S3505) et latence par condition ; la condition `all`
   reste-t-elle Gap 2 (< 100 ms) ? (`FIXME(gap2)`).
4. **Correction HDC×monitoring** : ancienne valeur 0.113 (artefact zéro-padding) → vraie valeur.
5. **Recommandation** : le 5-feat est-il un bon compromis perf/RAM, ou faut-il un sous-ensemble
   par modèle ? (réponse chiffrée, pas d'opinion).

**Règle** : conclusions adossées aux chiffres mesurés (heatmaps + ram.json), aucun chiffre inventé.

## Vérification

Lecture croisée : chaque affirmation chiffrée renvoie à un `exp_S35_*` ou une heatmap.

---

## Analyse (rédigée — S3512)

> Tous les chiffres ci-dessous sont lus depuis `experiments/comparison_sprint23.json`
> (`results_by_condition[condition][dataset][model][platform]`), les `exp_S35_board_*/results.json`
> (`.bss`, latence DWT, F1) et les `exp_S35_PC_*/ram.json`. **Aucun chiffre inventé** ; les cellules
> non mesurées sont « pending » (masquées dans les heatmaps). Conditions : `5feat` (référence board
> Sprint 32), `all` (dims natives), `best` (sous-ensemble optimal par modèle, k optimisé sur F1 val).

> **Mise à jour (complétion 120/120 cellules)** : les 12 heatmaps sont désormais **complètes**
> (PC 0 pending, board 0 pending), après trois correctifs amont qui débloquaient des cellules :
> (1) **Paderborn** — le normalizer en cache (clés top-5) écrasait l'accès aux 7 features natives →
> refit si les clés ne couvrent pas `feature_names` (`paderborn_loader.py`) ; (2) **TinyOL PC** —
> `_train_tinyol` fournissait des `encoder_dims`/`decoder_dims` à 2 éléments alors que l'autoencodeur
> en attend 3 → TinyOL PC ne tournait jamais (`feature_conditions.py`) ; (3) **sélection k\* CMAPSS**
> — option documentée `--max-samples` (sous-échantillonnage stratifié **uniquement** pendant la
> sélection ; le sweep S3503 garde les données complètes) pour rendre tractable les 22 ré-entraînements
> sur 16k–49k échantillons/tâche. Board : 7 cellules `(condition,dataset)` reflashées
> (paderborn ×3, best ×{cwru,monitoring,pronostia,cmapss}), **parité EWC+Maha 30/30**.

### 1. Impact features × modèle

**EWC** est le modèle qui profite le plus du choix des features, sur les datasets à forte dimension :

| Dataset (dims natives) | EWC F1 `5feat` | EWC F1 `all` | EWC F1 `best` | Plateforme |
|---|---|---|---|---|
| cmapss (21) | 0,456 | **0,658** | pending | PC |
| cmapss (21) | 0,381 | **0,615** | pending | board |
| pronostia (13) | 0,930 | 0,849 | **0,997** | PC |
| cwru (9) | 1,000 | 0,998 | 1,000 | PC |
| monitoring (4) | 0,893 | 0,893 | 0,886 | PC |

→ Le gain est réel là où le `5feat` ampute des features informatives : **cmapss** (21 sensors
ramenés à 5) gagne **+0,20 de F1 EWC** en condition `all`, board comme PC. Sur **pronostia** (13),
le sous-ensemble `best` (k=1, exp `exp_S35_board_best_ewc_pronostia`) bat le `5feat` (F1 PC
0,930→0,997) : plus de features ≠ meilleur, c'est la *pertinence* qui compte. À l'inverse,
**monitoring** est natif **4 features** : `5feat` ≡ `all` ≡ `best` (aucun gain possible, F1 ≈ 0,89).

**HDC** suit la même tendance sur PC : cmapss F1 0,000 (`5feat`) → 0,515 (`all`), pronostia
0,425 → 0,709 (`best`). **Mahalanobis** reste la baseline faible sur CWRU (F1 `best` 0,246 vs
`5feat` 0,127) : non supervisée, elle ne voit pas les labels — le choix des features l'aide peu.

**Datasets les plus pénalisés par le 5-feat** : cmapss (21→5, −0,20 F1 EWC) puis pronostia
(13→5). Datasets indifférents : monitoring (natif 4) et cwru pour les modèles supervisés
déjà saturés (EWC F1≈1,0 dans toutes les conditions).

### 2. F1 vs accuracy — l'accuracy trompeuse

Le cas emblématique est **Mahalanobis × CWRU** : accuracy board `5feat` = 0,160 mais F1 = 0,125 ;
l'accuracy *et* le F1 sont bas, donc cohérents ici. Le piège apparaît quand l'**accuracy paraît
correcte mais le F1 s'effondre** : sur **cmapss**, Mahalanobis PC affiche accuracy `5feat` = 0,745
pour un F1 de seulement **0,269** — l'accuracy est portée par la classe majoritaire `normal` tandis
que la classe `faulty` est ratée. Même logique au Sprint 26 (F1_MC=0,243 derrière une accuracy
flatteuse). **Conclusion méthodologique** : sur la détection de panne déséquilibrée, la heatmap
**F1 (classe faulty)** est le juge, pas l'accuracy.

### 3. Coût board — RAM (`.bss`) et latence par condition

La RAM board croît avec le nombre de features (buffers d'E/S + poids dimensionnés natifs) :

| Condition / dataset | n_features | `.bss` (B) | Latence EWC P50 | Latence Maha P50 | Latence HDC P50 |
|---|---|---|---|---|---|
| 5feat (réf.) | 5 | 104 956 | 50 µs | 5 µs | 585 µs |
| all cwru | 9 | 124 484 | 57 µs | 9 µs | 824 µs |
| all pronostia | 13 | 144 140 | 65 µs | 16 µs | 1 068 µs |
| all cmapss | 21 | **183 936** | 79 µs | 34 µs | **1 557 µs** |

→ **Gap 2 préservé partout où c'est mesuré** : même le pire cas (HDC × cmapss `all` = 1 557 µs ≈
**1,6 ms**) reste **≪ 100 ms** (`gap2_latency_compliant=true`). La condition `all` **reste Gap 2**.
Côté RAM, `all cmapss` à 183 936 B (≈ **70,2 %** des 256 Ko) est le poste le plus lourd mais tient
encore dans le budget NUCLEO-F439ZI. La condition **`best` HDC/TinyOL est désormais mesurée** (plus
de « à mesurer ») : pire cas `best cmapss` HDC k=18 = **1 374 µs** (`.bss` 156 864 B) — toujours
**≪ 100 ms**, `FIXME(gap2)` **levé**, Gap 2 ✅ sur les 30 cellules de parité (EWC+Maha).

### 4. Correction HDC × monitoring

L'artefact **accuracy 0,1133** (`exp_S33_board_gap1`, monitoring zéro-paddé 4→5, 5ᵉ feature nulle
faisant s'effondrer la projection HDC embarquée) est **remplacé par la valeur board réelle** obtenue
en features natives (4-feat, sans padding) : **accuracy 0,8667** (`exp_S35_board_all_hdc_monitoring`,
override `_apply_s3509_override`), cohérente avec la valeur PC légitime (0,850). La heatmap board et
`comparison_sprint23.json` portent désormais 0,867.

### 5. Recommandation chiffrée

Le **5-feat est un bon compromis perf/RAM par défaut** : il atteint l'essentiel de la performance
EWC (cwru/monitoring/pronostia F1 ≥ 0,89 PC) pour une `.bss` minimale (104 956 B). **Mais un
sous-ensemble par modèle est justifié sur les datasets à forte dimension** :

- **cmapss** : préférer `all` (21) — +0,20 F1 EWC board, coût +79 Ko `.bss` / +29 µs latence, Gap 2 OK.
- **pronostia** : préférer `best` (k=1–2) — F1 PC EWC 0,930→0,997, `.bss` 137 676 B, latence ≤ 39 µs.
- **monitoring / cwru** : conserver `5feat` (ou `all`=4 pour monitoring) — aucun gain, RAM minimale.

Autrement dit : **5-feat par défaut, `best`/`all` ciblé sur cmapss et pronostia**. Le choix n'est pas
binaire — il dépend du couple (dataset, modèle), et le coût board d'aller au-delà de 5 features reste
sous le budget Gap 2.

### 6. Compléments après complétion des 120 cellules

**Paderborn (débloqué).** Avec le loader réparé, Paderborn est mesuré PC **et** board, toutes
conditions. Constat clé : c'est un scénario **class-incremental mono-classe par tâche** (task0=normal,
task1/2=fault), donc seul un modèle supervisé robuste à l'oubli s'en sort — **EWC F1=0,80** (PC et
board, parité exacte), stable sur `5feat/all/best`. Les autres décrochent : Mahalanobis 0,07–0,11
(non supervisé), TinyOL very variable (0,01 `all` → 0,80 `best`), HDC 0,56–0,63 PC mais **0,00 board**
(cf. infra). Paderborn n'est donc **pas N/A** mais discrimine fortement les modèles.

**TinyOL (débloqué sur PC).** TinyOL PC apparaît enfin : fort sur cwru (F1 0,85→0,93 `5feat`→`best`),
faible sur pronostia/cmapss (≤0,36) — cohérent avec une détection d'anomalie one-class sensible au
choix de features. Le `best` aide nettement (cwru 0,85→0,93, paderborn 0,70→0,80).

**HDC board F1 = 0,00 partout — résultat réel, pas un artefact.** Sur la carte, HDC prédit la classe
majoritaire (`normal`) : l'**accuracy** reste correcte (ex. monitoring board 0,867) mais le
**F1_faulty s'effondre à 0**. C'est l'illustration la plus nette du message « accuracy trompeuse →
F1 » : un modèle peut sembler bon en accuracy tout en ne détectant **aucune** panne. À distinguer du
PC où la projection HDC (offline, bornes calibrées) donne des F1 non nuls (0,43–0,99). Honnête, non
masqué dans la heatmap board F1.

**Coût board des nouvelles dims.** `.bss` max global = **183 936 B (70,2 %)** sur `all cmapss` ;
`best cmapss` (HDC k=18, `PROTO_MAX_N=18`) = 156 864 B / 1 374 µs ; `all/best paderborn` (k=7/k=1) =
114 704 / 85 848 B, EWC ≤ 54 µs. **Toutes ≪ 100 ms — Gap 2 préservé sur l'intégralité de la grille.**
