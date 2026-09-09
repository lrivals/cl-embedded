# Script oral — présentation encadrants, sprints 44 → 53

Support : [`01_slides.md`](01_slides.md). Durée visée 35–40 min + questions.
Repères de minutage indicatifs, à ajuster selon les interruptions.

---

## Bloc 0 — Cadrage (slides 1–3) · ~4 min

**Slide 1 — la frise.** Ouvrir sur le volume : dix sprints, quatre axes, **huit sur dix
avec des mesures carte réelle**. Signaler tout de suite S51 en gris : *spécifié, non
exécuté*. Le dire soi-même vaut mieux que se le faire demander.

**Slide 2 — la carte des gaps.** Ne pas lire les cases. Un seul point : depuis juillet, le
Gap 2 a été **corrigé dans sa définition même** (S49), et le Gap 3 a reçu une réponse
**négative et mesurée** (S53). Ce sont les deux nouveautés de fond.

**Slide 3 — le plan.** Annoncer les cinq messages, puis enchaîner.

---

## Bloc 1 — Drift (slides 4–8) · ~7 min

**Slide 4.** Poser la question : jusqu'ici le modèle apprenait quand *nous* le décidions.
Insister sur le corpus synthétique — c'est lui qui permet de **valider la chaîne de
mesure** (pics à ±1 fenêtre des points imposés) avant de faire confiance aux corpus réels.
Electricity n'a pas de vérité-terrain : reporté `null`.

**Slide 5.** 36 cellules sur 36. Le message n'est pas le classement mais le **coût d'état**
qui sépare les deux familles : O(1) contre croissant avec la dimension. ADWIN sur 128
features = 245 Ko d'état, soit toute notre SRAM. C'est ce qui fixe la liste de portage.

**Slide 6 — le slide méthodologique du bloc, prendre son temps.**
« Notre proxy PC disait 6 µs. La carte dit 270. » Laisser le chiffre agir. Puis expliquer :
ce n'est pas une erreur du proxy, c'est le paradoxe FPU — le proxy sert à **ordonner** des
candidats, jamais à annoncer une latence. C'est la règle qu'on applique dans tout le reste
du projet.

**Slide 7.** Parité verdict = 1.000 : la décision d'apprendre prise à bord est identique à
celle du PC. Sur PSI, ne pas s'excuser — expliquer : l'état de PSI tient en 132 octets,
c'est **sa source de signal** (Mahalanobis en O(k²)) qui déborde. Portable en basse
dimension. La limite est documentée, pas effacée.

**Slide 8.** Assumer les F1 modestes : seuils de la littérature, non réglés sur ce flux.
Le réglage est un travail PC, hors du périmètre du portage.

---

## Bloc 2 — Quantification (slides 9–15) · ~10 min

**Slide 9.** Le point de cadrage : « quantifier » recouvre trois décisions que la
littérature confond. Les séparer est ce qui rend chaque conclusion attribuable.

**Slide 10.** Annoncer le résultat contre-intuitif : `both ≥ after`, mais de **4 à 8
millièmes**. Donc, sur cette tête, le QAT préserve sans apporter de gain décisif. Si on
demande « alors le QAT ne sert à rien ? » → slide 11.

**Slide 11 — le vrai contraste.** PTQ naïve : AUROC 0.498, c'est-à-dire le hasard. PTQ
calibrée : tout récupéré. *Quantifier ≠ quantifier.* Le choix n'est pas « INT8 ou FP32 »
mais **quelle calibration**.

**Slide 12.** Frontière ternaire ; le per-channel repousse le décrochage ; l'affine ne
rachète rien. Trois conclusions, trois balayages séparés.

**Slide 13 — le nœud d'honnêteté du sprint 48.** Sans packing, la `.bss` est **strictement
invariante** entre int4, ternaire et binaire — parce que les poids restent dans des
conteneurs `int8_t`. C'était le risque annoncé en début de sprint ; la mesure l'a confirmé.
Avec packing, le gain devient réel : 336 → 504 → 572 octets.

**Slide 14.** Le dépacking coûte ≈ 55 µs, constant, sans effet sur le Gap 2. Arbitrage
explicite : quelques centaines d'octets contre 55 µs.

**Slide 15.** Récapituler en quatre lignes, sans relire les chiffres.

---

## Bloc 3 — RAM (slides 16–18) · ~5 min

**Slide 16.** Annoncer d'emblée la correction : « nos chiffres RAM étaient sous-estimés ».
C'est une auto-correction, elle vient du CR du 16 juillet, et elle est traitée comme un
résultat : 32 cellules, invariant vérifié, méthode écrite, rapport **généré**.

**Slide 17.** Trois constats que seule la mesure de pile donne. Le troisième est le plus
important pour la suite : en `.bss`, **INT8 ≡ FP32**. La RAM des *poids* est bien ÷4 ;
la RAM *système* ne bouge pas. Poser la distinction ici, elle resservira slide 24.

**Slide 18.** Deux affirmations, une vraie une fausse, et c'est la mesure qui tranche.

---

## Bloc 4 — Latence et énergie (slides 19–25) · ~11 min

**Slide 19.** Le paradoxe était connu depuis le Sprint 29 ; il est maintenant chiffré poste
par poste. Le chiffre à retenir : la **requantification** (`lroundf`) consomme un tiers du
budget. Le MAC entier n'apporte rien face à la FPU.

**Slide 20.** Raconter honnêtement : première campagne, grille complète, dispersion
excellente… et deux constats de banc qui **invalident le protocole prévu**. Le repos
ressortait au-dessus de tous les régimes de flux : les µJ seraient sortis négatifs.
Insister : « à mesurer » avec une **raison mesurée**, pas « pas encore fait ».

**Slide 21.** Comment on a tranché : plan contre-balancé, verdict **calculé**
(`artefact_ordre`), pas déclaré. Puis les deux outils qui en sortent : préchauffage
systématique, garde-fou de plausibilité en point d'étranglement unique.

**Slide 22.** La cause profonde était dans le firmware : attente UART en scrutation active.
« Le repos n'était pas du repos. » Un `__WFI()` → −45 %. Formule à garder : *sur un banc
énergie, ce qu'on croit mesurer dépend de ce que le firmware fait quand il ne fait rien.*

**Slide 23.** Expliquer la ruse : la trame coûte plus cher que le calcul, donc on fait
varier le nombre d'inférences **par trame** et on lit la pente. Donner les deux contrôles
de validité — parité de prédiction 1.000 entre lots, et exclusion des points où la cadence
atteinte décroche (l'UART sature en silence). Finir sur l'autonomie : ≈ 73 h à 1 Hz.

**Slide 24 — le résultat principal du bloc.** Δ = −0,18 µJ pour ±2,34 µJ d'incertitude.
Verdict `non_significatif`. Conclure fermement : **le gain de l'INT8 est la RAM, ni la
latence, ni l'énergie.** Ajouter que c'est une conclusion négative *et* une contribution :
elle contredit l'argumentaire habituel, qui suppose un cœur sans FPU.

**Slide 25.** Ce qui, lui, change l'énergie : la fréquence. +26,2 % de 45 à 180 MHz, Gap 2
tenu avec ×43 de marge à 45 MHz. Formule : **la marge de latence est convertible en
autonomie**. Mentionner au passage le bug `profiling.c` corrigé (latences sous-estimées ×2
à ×4 aux fréquences réduites) — cela montre que le balayage a été audité.

---

## Bloc 5 — Qualité (slides 26–27) · ~4 min

**Slide 26.** Ne pas minimiser : une ligne du manuscrit était fausse. Raconter la
mécanique — pas de flag, donc branche par défaut, donc Mahalanobis publié sous le nom de
TinyOL, **jusqu'au champ `date` identique**. Puis la correction et la ré-mesure complète.
Effet de bord positif : les deux échecs Unity « préexistants » tombent.

**Slide 27.** Sept défauts, **trois produisaient des résultats crédibles et faux** — le
pire cas, puisque rien ne les signalait. Enchaîner sur les correctifs structurels (tests,
garde-fou unique, ajustement unifié, règle de publication arrêtée **avant** la mesure).
Terminer sur ce qui reste « à mesurer », chacun avec sa raison. Sur l'énergie de mise à
jour : les poids sont **écartés** comme cause, la dimension reste candidate, le mécanisme
n'est **pas établi** — le dire exactement comme ça.

---

## Bloc 6 — Bilan (slide 28) · ~3 min

Trois messages, puis les points ouverts. Ne pas conclure sur une note d'excuse : la
campagne énergie a produit une **réponse ferme au Gap 3** et un **levier d'autonomie**
identifié.

---

## Chiffres à connaître par cœur

| Sujet | Chiffre |
|---|---|
| Proxy PC vs carte (DDM) | 6 µs → **270 µs** |
| Parité verdict drift carte ↔ PC | **1.000** (0 / 13 910) |
| Surcoût `.bss` des détecteurs | +36 o (PH), +40 o (DDM), +132 o (PSI) |
| Moment de quantification (F1 carte) | `after` 0.9173 / 0.8995 → `both` **0.9213 / 0.9072** |
| PTQ naïve non calibrée | AUROC **0.498 / 0.546** (le hasard) |
| Frontière de profondeur | **ternaire** ; binaire OK sur Monitoring (−0.0117), casse sur Pronostia (−0.0275) |
| `.bss` sub-INT8 non packée | **invariante** (105 640 / 106 152 o) |
| Gain du bit-packing | 336 → 504 → **572 o** |
| Coût du dépacking | **≈ +55 µs** (67 → 123 µs) |
| RAM totale carte | **40 – 42 %** de 256 Ko ; `.data` 460 o ; pile 4,2 – 4,7 Ko |
| Ratio RAM totale INT8 / FP32 | **1.0002** (les poids sont ÷4, pas le système) |
| Latence EWC | FP32 **48 – 50 µs** vs INT8 **74 µs** |
| Cycles INT8 | déquant 200 · MAC 6 785 · **requant 3 777** |
| Repos scrutation → WFI | 49,77 → **27,37 mA** (−45,0 %) |
| Énergie par inférence (lot) | EWC **5,076 µJ** · Maha **0,437 µJ** (r² 0,9998) |
| Gap 3 énergie | Δ **−0,18 µJ** pour **± 2,34 µJ** → `non_significatif` |
| Fréquence | 45 → 180 MHz : **+26,2 %** d'énergie ; Gap 2 ×43 à 45 MHz |
| Autonomie duty-cyclée | ≈ **73 h** à 1 Hz / 2000 mAh |
| TinyOL après correctif | parité **44/44** et **45/45** ; F1 0,9268 / 0,8889 ; `make test` **141, 0 échec** |

---

## Questions anticipées

**« Pourquoi votre proxy PC se trompait-il autant ? »**
Ce n'est pas une erreur de mesure mais un changement de machine : jeu d'instructions,
hiérarchie mémoire et compilateur différents. Le proxy classe correctement les candidats
(O(1) contre O(dimension)) ; c'est tout ce qu'on lui demande. La latence, elle, se mesure
au DWT sur la carte.

**« Si l'INT8 ne gagne ni en latence ni en énergie, pourquoi le garder ? »**
Pour la RAM des poids, divisée par 4, et parce que la métrique est préservée quand la
calibration est correcte. Sur un cœur avec FPU matérielle, l'INT8 est une technique de
compression, pas d'accélération. Le résultat serait différent sur un cœur sans FPU ou avec
des instructions SIMD entières (piste CMSIS-NN).

**« Pourquoi certains chiffres restent-ils "à mesurer" ? »**
Parce que la mesure a été tentée et a échoué de façon identifiée. Exemple : le 3ᵉ
estimateur détecte 3 792 créneaux là où on en attend 100 ; diviser par ce dénominateur
**fabriquerait** une énergie. Chaque `na_reason` dit ce qui a été essayé et pourquoi ça n'a
pas abouti.

**« Comment savez-vous que les résultats publiés sont justes, si trois ne l'étaient pas ? »**
C'est la raison des correctifs structurels : le pilote de sonde est sous tests, le
garde-fou de plausibilité est en point d'étranglement unique, l'ajustement statistique a
une source unique, et la règle de publication d'une différence de pentes a été arrêtée
**avant** la mesure. Par ailleurs, chaque figure de ce support est régénérée depuis les
JSON — un chiffre faux dans une figure viendrait forcément d'un JSON, pas d'une saisie.

**« Le F1 de détection de drift est faible — le portage est-il utile ? »**
Le portage démontre la **fidélité** (parité 1.000) et le **coût** (270 µs, +40 octets),
qui sont les questions embarquées. La qualité de détection dépend du réglage des seuils,
qui se fait sur PC et n'a pas été fait ici.

**« Et le Sprint 51 ? »**
Spécifié, non exécuté. Il dépend d'une normalisation par dimension dont les entrées
viennent justement de S49 et S53, qui n'étaient pas stabilisées.
