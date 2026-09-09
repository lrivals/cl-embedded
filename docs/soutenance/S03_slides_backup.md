# S03 — Slides de secours et table de routage

Slides placées **après** la slide 15, non projetées pendant l'exposé, appelées uniquement
pendant les 15 min de questions. **15 slides, numérotées B1 à B16 — il n'y a pas de B15**
(voir ci-dessous).

> **Le passage de l'exposé à 15 min a alourdi ce lot.** Trois slides qui figuraient dans le
> fil de la version 30 min en sont sorties et vivent désormais ici : **B13** (Paderborn),
> **B14** (pile par phase) et **B16** (gelé contre en ligne). Une quatrième aurait dû suivre —
> le détail du gate autonome, ancienne slide 25 — mais elle est **déjà couverte par B9**, et
> la dupliquer aurait créé deux slides pour une même réponse : **le numéro B15 reste donc
> volontairement vide**. Ces slides ne sont pas des compléments exotiques : ce sont des
> résultats du fil principal, et il faut pouvoir les sortir vite.

**Règle d'usage** : ne pas ouvrir une slide de secours pour chaque question. Une réponse orale
suffit dans la plupart des cas ; la slide sert quand la question porte sur un **chiffre
précis**, une **objection méthodologique**, ou un travail **absent du manuscrit** dont
l'existence renforce la réponse. Annoncer l'appel : *« j'ai une slide là-dessus »* — cela
montre que la question était anticipée.

---

## Table de routage — question → slide

| Question probable | Réf. `S01` | Slide |
|---|---|---|
| « Comment la carte peut-elle battre le PC si la parité est exacte ? » | **Q-01** | **B1** |
| « Sur combien d'échantillons ? Quelle barre d'erreur ? » | **Q-02** | **B1** |
| « La grille complète sur les cinq jeux ? » | Q-24, Q-34 | B2 |
| « Comment fonctionne concrètement votre firmware ? » | Q-15, Q-41 | B3 |
| « Comment avez-vous isolé la cause de l'effondrement INT8 ? » | Q-22, Q-28 | B4 |
| « Le moment de la quantification change-t-il quelque chose ? » | Q-16, Q-43 | B5 |
| « Peut-on descendre sous 8 bits ? » | Q-28, Q-46 | **B6** |
| « Latence des autres modèles ? Pourquoi HDC est-il lent ? » | Q-21, Q-60 | B7 |
| « Et l'énergie ? » | **Q-30**, Q-55 | **B8** |
| « Votre gate autonome, en détail ? » | Q-35, Q-45, Q-51 | B9 |
| « Avez-vous étudié la détection de dérive elle-même ? » | Q-35, Q-45 | **B10** |
| « Effet du nombre de variables sur perf et RAM ? » | Q-32 | B11 |
| « Ce travail est-il publiable ? » | Q-44 | B12 |
| « Avez-vous un modèle universel ? Et sur Paderborn ? » | Q-36 | **B13** |
| « D'où vient le pic de pile ? La mise à jour coûte-t-elle de la mémoire ? » | Q-25 | B14 |
| « Que compare exactement votre parité ? Gelé, en ligne ? » | **Q-01**, Q-02 | **B16** |

Les slides en gras sont celles qui portent soit une question dangereuse (B1, B8, B16), soit un
travail **absent du manuscrit** que le jury ne peut pas anticiper (B6, B10) — effet de
profondeur maximal. **B13** est en gras pour une autre raison : elle sort de l'exposé alors
qu'elle porte le meilleur résultat d'EWC, et c'est la réponse à la question « votre modèle
tient-il partout ? » qui viendra sûrement.

---

# B1 — Protocoles d'évaluation : ce que chaque chiffre mesure ★

**Appelée par Q-01 et Q-02 — la slide la plus importante du lot.**

Tableau à trois colonnes qui dissipe la confusion en un regard :

| | Grille du ch. 5 (tab. 5.1) | Comparaison appariée du ch. 6 | Ce que ça mesure |
|---|---|---|---|
| Population | échantillon du flux | 7 534 à 7 672 échantillons | |
| Protocole PC | évaluation CL complète, toutes tâches | rejeu de la séquence exacte | |
| Protocole carte | flux rejoué, régime gelé | même séquence, même ordre, même graine | |
| Comparaison | **indirecte** (deux populations) | **appariée**, prédiction à prédiction | |
| Résultat | classement des modèles | **parité 1,000** en gelé · 0,963–0,989 en ligne · Δacc ≤ 0,007 | |

**À dire** : « La parité mesure la fidélité du portage. Le tableau 5.1 mesure la performance
des modèles. Ce sont deux questions différentes, avec deux protocoles différents, et c'est
l'unique raison pour laquelle les deux colonnes du tableau diffèrent. »

**Figure d'appui** : `docs/figures/soutenance/s21_parite_desaccords.png` (déjà slide 21 — la reprojeter).

---

# B2 — Grille complète : quatre modèles × cinq jeux

**Appelée par** une demande de résultats au-delà des trois jeux focus.

Reprendre le tableau A.2 du manuscrit en entier (Monitoring, CMAPSS, Pronostia, CWRU,
Paderborn, PC | carte), plus les deux heatmaps :
- `docs/figures/soutenance/b2_grille_complete.png` (PC et carte sur la même ligne)

**Trois lectures à donner** : EWC est le seul robuste sur toute la grille et le seul à ne
jamais s'effondrer au passage sur carte · HDC affiche un F1 nul sur les cinq jeux embarqués
alors qu'il atteint 0,955 sur CWRU côté PC · Mahalanobis décroche sur les deux jeux
vibratoires (0,127 et 0,071), limite attendue d'une gaussienne unique.

---

# B3 — Architecture firmware et protocole UART

**Appelée par** toute question sur le fonctionnement concret. Le manuscrit ne contient **aucun
schéma d'architecture** (chapitres 1–4 sans figure) : cette slide comble un vide réel.

Figure : `docs/figures/soutenance/b3_architecture_firmware.png`.

Points : les quatre familles co-résidentes en mémoire, sélectionnées par un champ de la trame ·
trame UART à format figé avec CRC · le flux hôte → carte → prédiction, un échantillon par
trame · zéro erreur CRC sur toutes les campagnes.

---

# B4 — Ablation : isoler la cause de l'effondrement INT8

**Appelée par** une question sur la méthode de diagnostic.

Figure : `docs/figures/soutenance/b4_ablation_echelle.png`.

Trois causes candidates testées **séparément** au moyen de l'émulateur bit-exact :
accumulateur int16 qui sature · échelle fixe 1/128 non calibrée · quantification en une passe
sans recalibration. **Un seul étage produit un saut : la calibration de l'échelle.**
L'accumulateur int32 — le coupable évident — n'apporte rien et dégrade même Monitoring tant que
l'échelle reste fausse ; le passage au par-canal ne rachète plus rien une fois le par-tenseur
calibré.

**À dire** : la crédibilité de l'émulateur ne repose pas sur une affirmation — sa parité contre
la carte est exacte.

---

# B5 — Le moment de la quantification

**Appelée par** une question sur QAT contre PTQ.

Figure : `docs/figures/soutenance/b5_moment_quantification.png`.

Quatre moments à modèle, jeu et graine fixés : référence flottante · QAT · PTQ calibrée · les
deux combinées (le déploiement réel). **Échelle verticale très resserrée : les quatre sont
quasi confondus.**

**Message à porter tel quel, sans l'embellir** : le QAT ne dégrade pas, mais il n'apporte pas
de gain décisif sur cette tête — c'est la **calibration du noyau** qui récupère l'essentiel de
la métrique.

---

# B6 — En dessous de 8 bits : profondeur, granularité, symétrie ★

**Absent du manuscrit.** À sortir seulement si l'on est poussé sur « jusqu'où peut-on aller ? »
ou « le ÷4 est-il votre plafond ? ». Effet de profondeur fort.

Figure : `docs/figures/soutenance/b6_profondeur_bits.png` (la version manuscrit
`ch7_profondeur_bits.png` est présente dans `images/` mais **non incluse dans
le rapport**).

Étude balayant la profondeur des poids (INT8 → INT4 → ternaire → binaire), la granularité
(par tenseur contre par canal) et la symétrie, sur EWC × {Monitoring, Pronostia}, avec un
émulateur bit-exact puis **portage et mesure sur carte** :
- **Monitoring tient jusqu'au binaire** (Δ AUROC = −0,012) ; **Pronostia casse au binaire**
  (−0,028) mais **le ternaire tient** (−0,015).
- Le **par-canal repousse le point de rupture** : Pronostia 2 bits passe de −0,046 (par tenseur)
  à −0,009 (par canal).
- Le **zero-point affine n'apporte rien**.
- Sur carte : le gain RAM sub-INT8 n'est **réel que si les poids sont empaquetés** — sans
  empaquetage, le conteneur reste un octet et la RAM est celle de l'INT8. Le dépaquetage coûte
  ≈ +55 µs, sans menacer le budget de 100 ms.

**À dire** : « la frontière recommandée est le ternaire, et l'agressive le binaire — mais le
gain n'existe qu'avec un vrai bit-packing, ce qui est exactement le genre de nuance qu'on ne
voit qu'en mesurant sur la cible. »

---

# B7 — Latence par famille de modèles

**Appelée par** une question sur la dispersion ou sur HDC.

Figure : `docs/figures/soutenance/b7_latence_par_modele.png` (échelle logarithmique).

Deux ordres de grandeur : Mahalanobis ≈ 3–5 µs · EWC ≈ 48–50 µs · TinyOL ≈ 81–85 µs ·
HDC ≈ 518–585 µs, jusqu'à ≈ 2,1 ms en INT8. La latence de TinyOL croît avec le nombre de
variables — signature de l'auto-encodeur `k → 32 → 16 → k` — là où Mahalanobis reste constant.
L'encodage hyperdimensionnel domine le coût de HDC.

---

# B8 — Énergie : ce qui est mesuré et ce qui ne l'est pas ★

**Appelée par Q-30 et Q-55.** La question tombera presque à coup sûr ; avoir une slide
transforme une lacune en démonstration de rigueur.

Figure : `docs/figures/energy_real/e6_courant_moyen_mesure.png`, et
`docs/figures/energy_real/e5_cout_benefice.png` si l'on développe.

Trois points, dans cet ordre :
1. **La chaîne est prête** : marqueurs matériels dans le firmware délimitant les phases,
   segmentation et intégration d'un profil de puissance, calcul d'autonomie depuis la capacité
   batterie.
2. **Ce qui a été mesuré** : le courant moyen à cadence imposée, par modèle et par précision.
   L'ordre des consommations suit l'ordre des latences. **L'INT8 ne réduit pas la
   consommation** — les écarts sont dans le bruit, et HDC INT8 consomme même davantage.
3. **Ce qui ne l'est pas** : les microjoules par inférence, parce que la référence « repos »
   mesurée s'est révélée supérieure aux régimes de flux — le firmware attend en scrutation
   active sur l'UART, donc le « repos » n'est pas inactif — ce qui produirait des énergies
   négatives. Le champ reste littéralement « à mesurer ».

**À dire** : « je n'ai pas dérivé un chiffre d'énergie depuis la latence et un courant de fiche
technique. Cela aurait été présentable et faux — et notamment cela aurait raté le fait que le
gain mémoire de l'INT8 ne se traduit ni en gain de temps, ni en gain d'énergie. »

---

# B9 — Le gate de mise à jour autonome, en détail

**Appelée par Q-35, Q-45, Q-51.**

Figures : `docs/figures/soutenance/b9_economie.png` et
`docs/figures/soutenance/b9_parite_gate.png` · `docs/figures/soutenance/s25_gate_economie.png`
en variante de synthèse (elle était la slide 25 du plan à 30 min ; l'exposé à 15 min n'en garde
que trois phrases en clôture, donc cette slide porte désormais **tout** le détail du gate).

Quatre politiques mesurées sur carte, deux jeux, deux modes d'initialisation :

| Politique | Taux de MAJ | Latence | F1 Monitoring | F1 Pronostia |
|---|---|---|---|---|
| gelée | 0,000 | 48–50 µs | 0,919 | 0,916 |
| systématique | 1,000 | 238–251 µs | 0,901 | 0,930 |
| déclenchée, étiquette vraie | 0,025 | 79–82 µs | **0,919** | 0,889 |
| déclenchée, pseudo-étiquette | 0,025 | 79–82 µs | 0,919 | **0,504** |

**Parité de verdict carte ↔ PC = 1,000** : la décision prise à bord est exactement celle que
reconstruit la référence hors ligne. Surcoût mémoire de l'état du détecteur ≈ 300 B.

**À dire** : la recommandation de déploiement est l'**apprentissage actif** — le capteur décide
*quand* demander, l'humain décide *quoi* apprendre. L'autonomie complète est le sujet de
recherche, pas le produit.

---

# B10 — Détection de dérive : famille de détecteurs et portage ★

**Absent du manuscrit.** À sortir sur « comment détecte-t-on la dérive ? » ou « votre gate
utilise-t-il un vrai détecteur ? ».

Neuf détecteurs implémentés et comparés sur quatre jeux à dérive labellisée, avec métriques de
délai de détection, taux de fausses alarmes, taux de manqués :
- **supervisés à état constant** — DDM, EDDM, Page-Hinkley : 16 à 20 octets d'état, coût
  indépendant du jeu de données ;
- **non supervisés** — ADWIN, KSWIN, KS, MMD, PSI : coût qui **croît avec la dimensionnalité**,
  jusqu'à 245 Ko d'état pour ADWIN sur un jeu à 128 variables, donc non portable.

**Trois portés sur carte** (Page-Hinkley, DDM, PSI) avec **parité de verdict carte ↔ PC = 1,000
sur 13 910 échantillons**, latence 270 µs, zéro erreur CRC.

**Une limite mesurée à citer** : PSI tire son signal du détecteur de Mahalanobis, dont l'inverse
de covariance est en O(k²) ; à 128 variables cette matrice pèse à elle seule ≈ 64 Ko et fait
déborder la SRAM à l'édition de liens. Le goulot est la **source du signal**, pas l'état du
détecteur. PSI n'est donc portable qu'en basse dimension — c'est d'ailleurs une des limites
déclarées de l'annexe du manuscrit.

---

# B11 — Effet du nombre de variables : le compromis performance / RAM

**Appelée par** une question sur le choix des variables ou la validité de la condition `5feat`.

Trois conditions balayées — `5feat`, `all`, `best` — sur quatre modèles et cinq jeux, mesurées
**sur carte**.

- EWC × CMAPSS : F1 **0,38 → 0,62** en passant de `5feat` à `all`, mais `.bss` **104 956 →
  183 936 B** (40,0 % → 70,2 %).
- Parité EWC et Mahalanobis **exacte sur toutes les dimensions**, de k = 1 à k = 21.
- Toutes les latences restent ≤ 1,6 ms, donc le budget de 100 ms est préservé dans toutes les
  conditions.

**À dire** : le compromis est à arbitrer par couple (modèle, jeu), et l'automatiser est la
deuxième perspective du manuscrit.

---

# B12 — Valorisation

**Appelée par Q-44 ou une question sur la suite.**

- **Article court** rédigé en français et en anglais, centré sur la tête EWC en INT8 sur
  microcontrôleur : parité FP32 mesurée → effondrement de la PTQ naïve mesuré board →
  récupération par noyau calibré émulé bit-exact, avec distinction explicite entre *mesuré sur
  carte* et *émulé sur PC*. Cibles : ateliers TinyML et embarqué.
- **Chaîne de publication** du dépôt : export reproductible et documenté, prêt pour un dépôt
  institutionnel.
- **Ce qui manquerait pour une conférence pleine** : répétitions multi-graines, une seconde
  carte (idéalement sans FPU, pour la symétrie du résultat INT8), et les mesures d'énergie.

---

# B13 — Paderborn : le régime le plus sévère, où seule la régularisation tient ★

**Appelée par Q-36**, ou par toute question de la forme « votre modèle tient-il partout ? ».
**Était la slide 16 du plan à 30 min** — sortie du fil faute de temps, mais c'est le meilleur
argument en faveur d'EWC, donc il faut savoir la sortir vite.

Figure : `docs/figures/soutenance/s16_paderborn_ewc_seul.png`.

Jeu **strictement mono-classe par tâche** : à chaque étape, le modèle ne voit qu'une seule
classe. C'est le régime class-incremental poussé à sa limite, et celui où l'oubli est le plus
mécaniquement destructeur.

- **EWC : 0,800 sur PC comme sur carte** — même valeur des deux côtés, ce qui recoupe la
  parité de la slide 12.
- Les trois autres familles se replient sur la **prédiction de la classe majoritaire** :
  accuracy correcte, F1 de la classe fautif nul. Troisième illustration du piège de la
  slide 7, sur un jeu encore différent.

**À dire**, et c'est la phrase qui vaut le plus cher : *« je n'ai pas de modèle universel — sur
CMAPSS, EWC plafonne à 0,456. Ce que je peux dire, c'est que plus le régime d'oubli est sévère,
plus l'écart se creuse en faveur de la régularisation. »*

---

# B14 — Le pic de pile par phase : ce que coûte la mise à jour

**Appelée par Q-25**, ou par une question sur la composition de la RAM. **Était la slide 19 du
plan à 30 min**, repliée dans la slide 10 qui n'en donne plus que le total.

Figure : `docs/figures/soutenance/s19_pile_par_phase.png`.

Relevé par *stack painting* **après chaque phase**, sur Monitoring :

| Phase | Pic de pile |
|---|---|
| repos | ligne de base |
| inférence seule | **4 416 B** |
| inférence + mise à jour CL | **4 688 B** |

Soit **+272 octets** imputables à l'adaptation en ligne — les tampons temporaires de la
rétropropagation et de la mise à jour de Fisher.

**À dire** : *« le coût mémoire de l'adaptation est ce que la littérature embarquée passe sous
silence. Il existe, je le mesure, et il est borné — c'est le pendant mémoire du facteur 5 de
latence de la slide 11. »* Préciser que le watermark donne ce qui **s'est produit**, pas un
pire cas théorique : c'est une mesure, pas une analyse statique.

---

# B16 — Gelé contre en ligne : ce que la parité compare exactement ★

**Appelée par Q-01 et Q-02**, en appui direct de **B1** — les deux se sortent souvent
ensemble. **Était la slide 12 du plan à 30 min**, repliée en une phrase dans la slide 6.

Figure : `docs/figures/soutenance/s12_gele_vs_en_ligne.png`.

Le dispositif tient en trois invariants : **mêmes poids** · **mêmes échantillons, même ordre,
même graine** · **comparaison prédiction à prédiction**. De là, deux régimes aux attentes
différentes :

| | Régime **gelé** | Régime **en ligne** |
|---|---|---|
| Ce qui tourne | inférence seule, poids figés | inférence + mise à jour CL |
| Parité attendue | **exacte** | **approchée** |
| Mesurée | **1,000**, sur 7 534–7 672 échantillons | 0,963 à 0,989 |
| Origine de l'écart | aucune | float32 sur carte, float64 sur PC |
| Où sont les écarts | — | concentrés aux **frontières de décision** |

**À dire**, mot à mot, c'est la formulation qui règle la question : *« un écart en régime gelé
signale un bug de portage ; un écart en régime en ligne, borné et concentré sur les frontières
de décision, est attendu. C'est cette distinction qui me permet d'affirmer que 0,240 sur PC
contre 0,243 sur carte est de l'oubli fidèlement reproduit, et pas une erreur d'implémentation. »*

---

## Note d'usage pendant les questions

1. **Écouter la question en entier.** Ne pas commencer à répondre sur les cinq premiers mots.
2. **Reformuler si elle est ambiguë** — cela donne du temps et évite de répondre à côté.
3. **Concéder d'abord quand la critique est juste**, puis expliquer ce que le résultat vaut
   malgré tout. C'est plus solide que de défendre l'indéfendable, et le jury le sait.
4. **Ne jamais inventer un chiffre.** « Je n'ai pas ce chiffre en tête, mais l'ordre de
   grandeur est X » est une réponse acceptable ; un chiffre faux ne l'est pas.
5. **Ne pas ouvrir plus de trois ou quatre slides de secours** sur 15 min — au-delà, cela
   ressemble à une seconde présentation.
