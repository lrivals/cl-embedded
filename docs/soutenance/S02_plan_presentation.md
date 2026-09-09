# S02 — Plan de la présentation de soutenance

**Format** : 15 min d'exposé + 15 min de questions · **15 slides** · public : jury n'ayant lu
que le manuscrit.

Ce document est le **plan de contenu**, pas le deck. Pour chaque slide : le message unique, ce
qui est affiché, la figure exacte à insérer, les notes orateur et le minutage.

> **Quinze minutes, pas trente.** Ce plan remplace une version calibrée sur 30 min et 26
> slides. La coupe n'est pas uniforme : la surface a été rétrécie plutôt que chaque slide
> raccourcie. Rien n'est perdu — ce qui sort du fil descend en slide de secours dans
> [`S03`](S03_slides_backup.md), et sert pendant les 15 min de questions.

---

## Quatre principes de conduite

**1. Un message par slide.** Si une slide a besoin de deux phrases pour être résumée, elle
doit être coupée en deux. Les tableaux complets vont en backup (`S03`), pas dans le fil.

**2. Le fil rouge est un modèle, pas une liste de modèles.** ★ *Principe nouveau, et c'est lui
qui rend l'exposé tenable en 15 min.* Un seul modèle — **EWC** — est suivi de bout en bout :
ce qu'il oublie sur données réelles (Gap 1), ce qu'il coûte en mémoire et en temps sur la
carte (Gap 2), ce qu'il devient quantifié (Gap 3). Les trois autres familles n'apparaissent
**qu'une fois**, slide 9, pour situer EWC. Chaque fois qu'une slide cite un chiffre, dire de
quel modèle il vient : le jury doit sentir qu'on suit le même objet.

**3. La slide 4 est le fil rouge.** Le triple gap y est posé, puis rappelé en bandeau discret
aux slides 7, 10 et 13 : le jury doit toujours savoir quel gap est en cours de traitement.
C'est ce qui donne son unité à un exposé qui couvre beaucoup de terrain.

**4. Annoncer les limites avant que le jury ne les trouve.** Une limite énoncée par l'orateur
est un point de rigueur ; trouvée par le jury, c'est un point faible. Trois endroits où il
faut le faire explicitement, en une phrase, sans s'appesantir :
- **slide 9** : les deux colonnes du tableau n'ont pas le même effectif ni le même protocole
  (désamorce Q-01 et Q-02, les deux questions les plus dangereuses) ;
- **slide 10** : trois chiffres de RAM existent, voici lequel est *le* chiffre ;
- **slide 14** : le bilan du Gap 3 est nuancé, deux résultats négatifs sont publiés.

---

## Table de minutage

| Bloc | Slides | Durée | Cumul |
|---|---|---|---|
| 1 — Contexte : le problème et EWC | 1–3 | 2:50 | 2:50 |
| 2 — Positionnement et méthode | 4–6 | 3:20 | 6:10 |
| 3 — Gap 1 : données industrielles | 7–9 | 2:40 | 8:50 |
| 4 — Gap 2 : mesures sous contrainte | 10–12 | 2:25 | 11:15 |
| 5 — Gap 3 : quantification | 13–14 | 2:25 | 13:40 |
| 6 — Clôture | 15 | 1:20 | 15:00 |

**15:00 pile pour 15 annoncées, donc aucune marge.** À ce budget, la seule variable
d'ajustement est le **débit**, pas le contenu. Ne jamais rogner les slides 4, 6, 9 et 12, qui
portent respectivement le plan, la contribution méthodologique, la limite annoncée et
l'argument scientifique.

> **Ce minutage a été calibré sur le texte réellement écrit**, pas estimé : chaque durée
> ci-dessus correspond au décompte de mots de la slide dans [`S05`](S05_script_oral.md), au
> débit d'un script répété. Deux conséquences que la première répartition « en blocs ronds »
> masquait : la **slide 6 tient 1:20 et non 1:00** — elle fusionne trois slides de la version
> 30 min, et une minute n'y suffit pas ; la **slide 9 tient 1:00 et non 0:45** — elle porte
> trois points obligatoires dont l'auto-annonce de la limite. Le temps est repris sur les
> slides 2, 7, 11, 13 et 15. **Toute modification d'une durée ici doit être répercutée dans
> `S05`**, où le budget de mots la contrôle.

---

# Bloc 1 — Contexte : le problème et EWC (2:50)

### Slide 1 — Titre
**Durée** 0:20 · **Figure** aucune

Titre du mémoire, nom, les trois encadrants avec leur institution (Arnaud Dion ISAE-SUPAERO,
Dorra Ben Khalifa ENAC/LII, Frédéric Zbierski Edge Spectrum), date.

**Notes orateur** — Ne pas lire la slide. Une phrase d'accroche, qui pose la thèse en un
souffle : *« Je vais vous parler de modèles qui continuent d'apprendre après leur déploiement,
sur un microcontrôleur à 256 Ko de mémoire — et surtout de ce qu'il en coûte, mesuré. »*

---

### Slide 2 — La machine change, et on ne peut pas simplement réentraîner
**Message** : un modèle déployé se dégrade parce que la machine change — et les trois portes de
sortie habituelles sont fermées sur un microcontrôleur.
**Durée** 1:10 · **Figures** `docs/figures/soutenance/s2_cas_usage.png` + `docs/figures/soutenance/s3_cycle_de_vie.png` + `docs/figures/soutenance/MaintenancePredictive_Intro.png` (visuel d'introduction à la maintenance prédictive — capture externe, à citer comme telle)

Machine industrielle → capteurs → modèle de détection de panne. Puis la flèche du temps : la
machine vieillit, le capteur dérive, les conditions opératoires changent → la performance du
modèle figé décroît.

Et les trois obstacles au réentraînement, une ligne chacun : **mémoire** (pas de place pour
l'historique) · **connectivité** (débit, disponibilité, confidentialité) · **réactivité** (un
cycle centralisé prend des heures).

**Notes orateur** — Cette slide fusionne les deux slides d'ouverture de la version 30 min :
elle pose le phénomène *et* ferme les issues, parce qu'à 15 min on ne peut pas s'offrir une
slide par idée ici. Nommer la dérive de distribution et donner les trois types du chapitre 2
en une phrase (covariables, domaine, abrupte contre graduelle) — sans détailler. Enchaîner
directement : *« donc le modèle doit apprendre sur place, à partir du flux, sans revoir le
passé — et c'est là qu'un obstacle propre à l'apprentissage apparaît. »*
*Marge : le paragraphe « réactivité » est marqué ⏱ dans `S05` et se coupe en premier (≈ 20 mots,
≈ 7 s). Les deux autres obstacles sont attendus du jury, les garder.*

---

### Slide 3 — L'obstacle, et la réponse : oubli catastrophique → EWC
**Message** : apprendre du nouveau détruit l'ancien ; EWC répond en rendant rigides les poids
importants.
**Durée** 1:20 · **Figures** `docs/figures/soutenance/KirkArticle_Plot.png` **panneau A**
(Kirkpatrick 2017, publié) + `docs/figures/soutenance/KirkArticle_SchemaLoss.png` (bassins de
perte, figure originale du même article)

**Deux temps sur une seule slide.** D'abord le problème : la descente de gradient optimise la
tâche courante au détriment des représentations construites pour les précédentes — compromis
stabilité–plasticité. Puis la réponse : pénalité quadratique pondérée par la diagonale de
Fisher, les poids importants deviennent rigides. Tête EWC embarquée : `k → 32 → 16 → 2`.

**Notes orateur** — Point clé, parce qu'il fonde tout le chapitre 5 : le décrochage n'est pas
un bug, c'est une conséquence de l'optimisation. Annoncer qu'on le **mesurera** plus loin sur
données réelles — première promesse faite au jury, tenue slide 8.

Ne pas écrire la formule d'EWC — **le schéma des bassins de perte la remplace** : les deux
ellipses sont les régions de faible erreur des tâches A et B, les trois flèches montrent ce que
devient le point de départ selon la contrainte (aucune → on sort du bassin A ; L2 → on ne bouge
presque plus ; EWC → on rejoint B **en restant dans** A). Une phrase par flèche suffit.

> **Les deux figures viennent de `Kirkpatrick2017EWC` et doivent être citées comme telles**,
> visiblement sur la slide et pas seulement à l'oral.
>
> **Le rapprochement avec le panneau A est volontaire — et il faut l'assumer.** Notre mesure
> d'oubli (slide 8) a la même forme (tâches enchaînées, décrochage à la bascule) mais pas la
> même conclusion : chez Kirkpatrick, la courbe EWC **tient** la tâche A ; dans notre mesure,
> elle décroche presque comme le témoin — l'oubli moyen vaut 0,858 sans EWC et 0,848 avec
> λ = 400. La différence tient au régime : *permuted MNIST*, tâches de même nature et de même
> difficulté, contre **CWRU class-incremental à 10 classes** où chaque tâche introduit des
> classes inédites. Formulation qui tient devant le jury : *« la régularisation suffit dans le
> régime de l'article ; sur nos données industrielles, elle ne suffit pas — c'est un des
> résultats du travail, pas un échec de mise en œuvre. »* Ne surtout pas annoncer ici « et EWC
> règle le problème » : ce serait démenti slide 8.

> **Variante à arbitrer en répétition.** La version 30 min projetait ici notre **mesure**
> d'oubli (`docs/figures/soutenance/s4_oubli_mesure.png`, `exp_S54_forgetting_{ewc,naive}`)
> en regard du panneau A. Elle a été retirée pour tenir 1:20 avec deux figures au lieu de
> trois — le chiffre mesuré arrive de toute façon slide 8. **Si la slide 3 paraît trop
> théorique à la répétition, c'est cette figure qu'il faut réintroduire**, en supprimant alors
> la vignette de la page de titre slide 4 pour compenser. Trois figures sur cette slide est
> le maximum absolu.

---

# Bloc 2 — Positionnement et méthode (3:20)

### Slide 4 — L'état de l'art et le triple gap ★ SLIDE PIVOT
**Message** : quatre travaux tracent le champ, chacun laisse un côté ouvert — trois lacunes
qu'aucun ne comble ensemble, et c'est le plan de l'exposé.
**Durée** 1:15 · **Figures** `docs/figures/soutenance/s5_etat_de_lart.png` (tableau) +
`docs/figures/soutenance/s6_triple_gap.png` · en vignette `docs/figures/soutenance/KirkArticle_Frontpage.png` (page de titre du travail dont EWC est issu)

| Travail | Ce qu'il démontre | Ce qu'il laisse ouvert |
|---|---|---|
| TinyOL (Ren 2021) | apprentissage en ligne sur Cortex-M4, +10 % de latence | pas de décomposition mémoire |
| QLR-CL (Ravaglia 2021) | rejeu latent quantifié en 8 bits, ÷4 d'empreinte | entraînement resté flottant |
| HDC (Benatti 2019) | apprentissage sans gradient sur SoC très basse conso. | pas de cadre PdM industriel |
| LifeLearner (Kwon 2023) | 212 Ko sur Cortex-M7 | tâche supervisée, surcoût d'adaptation non mesuré |

- **Gap 1** — validation sur données industrielles temporelles réelles, protocole reproductible
- **Gap 2** — démonstration sous contrainte mémoire, avec mesures précises composant par composant
- **Gap 3** — quantification INT8 **pendant** l'entraînement incrémental

**Notes orateur** — La slide la plus importante de l'exposé, et la seule qui justifie de
consacrer 1:15 à du positionnement quand on en a 15 en tout. Ne pas lire le tableau : insister
sur la **colonne de droite**, une phrase par ligne maximum, c'est elle qui produit les trois
gaps.

Puis **annoncer le fil et la promesse**, explicitement, dans ces termes :
*« Les trois blocs qui suivent traitent chacun l'un de ces gaps, dans cet ordre. Et pour que
quinze minutes suffisent, je les traite sur un seul modèle — EWC — que je suis jusqu'au bout :
ce qu'il oublie, ce qu'il coûte, ce qu'il devient quantifié. »*

Dire dès maintenant, en une phrase, que le bilan du troisième gap sera **nuancé** — cela
installe la crédibilité et désamorce Q-28 avant qu'elle n'arrive. Prévoir un rappel en bandeau
discret sur les slides 7, 10 et 13.

---

### Slide 5 — La cible et le banc
**Message** : voilà l'enveloppe matérielle et les données dans lesquelles tout doit tenir.
**Durée** 0:45 · **Figures** `docs/figures/soutenance/s7_fiche_carte.png` + `docs/figures/soutenance/s8_jeux_donnees.png` (tab. 4.1 condensé)

NUCLEO-F439ZI · Cortex-M4 @ 180 MHz · FPU simple précision · **256 Ko SRAM** (192 + 64 CCM) ·
2 Mo Flash · **pas de NPU**. Budget de latence fixé : **100 ms** par inférence + mise à jour.

Six jeux industriels, deux mis au focus : **Monitoring (D2)**, domain-incremental par type
d'équipement (pompe → turbine → compresseur) · **Pronostia (D4)**, class-incremental, la faute
n'apparaît qu'après une phase normale.

**Notes orateur** — Slide de cadrage, à passer vite mais pas à sauter : le jury a besoin des
deux scénarios pour lire les slides 8 et 9. Une phrase par scénario, et préciser que le
découpage en tâches est fixé par configuration, donc reproductible.

Dire en une phrase pourquoi cette carte et pas la STM32N6 initialement prévue : elle n'était
pas disponible, j'ai travaillé sur du matériel réellement en main. Et signaler tout de suite
que **la présence de la FPU** produira le résultat le plus contre-intuitif de l'exposé — cela
crée une attente pour la slide 14. *Marge : la phrase sur la FPU est marquée ⏱ dans `S05` (≈ 20
mots) — mais la couper prive la slide 14 de son effet d'annonce, donc en dernier recours.*

---

### Slide 6 — La chaîne de portage et le protocole de mesure ★
**Message** : la parité PC ↔ carte n'est pas espérée, elle est garantie par construction — et
chaque grandeur a une méthode de mesure explicite.
**Durée** 1:20 · **Figures** `docs/figures/soutenance/s10_chaine_portage.png` + `docs/figures/soutenance/s11_protocole_mesure.png`

Quatre étapes : entraînement PyTorch → **export des poids en en-têtes C générés** → firmware
C compilé en FP32 avec FPU → streaming UART hôte ↔ carte.

Et les quatre grandeurs mesurées : **latence** (compteur de cycles DWT, converti à 180 MHz,
inférence seule *contre* inférence + mise à jour, systématiquement séparées) · **RAM statique**
(symboles du linker, `_ebss − _sbss` et `_edata − _sdata`) · **pile** (*stack painting* : on
peint la zone libre, on relit jusqu'où le motif a été écrasé) · **intégrité** (CRC par trame,
taux visé et obtenu à zéro).

**Notes orateur** — C'est la contribution méthodologique, à vendre comme telle : c'est ce qui
se réutilise au-delà du cas d'étude, et la slide 15 y reviendra. Les trois invariants à
énoncer : **les poids sont générés, jamais écrits à la main** · **le format de trame est figé,
toute évolution firmware est répercutée côté hôte** · **la sélection des variables a une source
unique, partagée par les deux plateformes**. Conclure : *« c'est ce qui fait que la parité est
attendue par construction — mais je la vérifie quand même, prédiction par prédiction. »*

Sur le protocole, une seule idée à faire passer : *mesuré* contre *estimé* — le watermark donne
ce qui s'est produit, pas un pire cas théorique. C'est ce qui rendra crédibles les slides 10
à 12.

> **La distinction gelé / en ligne est repliée ici**, en une phrase, faute de slide dédiée. La
> formuler telle quelle, parce qu'elle sera reprise slide 12 : *« je compare les deux
> plateformes sur les mêmes poids, les mêmes échantillons, le même ordre et la même graine —
> en régime gelé la parité doit être exacte, en régime en ligne elle est approchée parce que
> la carte calcule en float32 et le PC en float64. »* La slide de secours **B16** porte le
> schéma complet si la question vient.

---

# Bloc 3 — Gap 1 : validation sur données industrielles (2:40)

> Bandeau : **Gap 1 — données industrielles réelles**

### Slide 7 — L'accuracy est trompeuse
**Message** : sur des jeux déséquilibrés, l'accuracy peut masquer un modèle qui ne détecte rien.
**Durée** 0:40 · **Figure** `docs/figures/soutenance/s13_accuracy_trompeuse.png`

Cas d'ouverture : Mahalanobis × CMAPSS → **accuracy 0,745 pour un F1 de 0,269** *sur PC*
(**0,853 / 0,214 sur carte** — la figure projetée trace les cellules **carte**, citer la paire
correspondant à ce qu'on montre). Puis la figure généralise à tous les couples modèle × jeu
mesurés sur carte.

**Notes orateur** — Slide courte et à fonction unique : justifier, une fois pour toutes, le
choix du **F1 de la classe fautif** comme métrique de référence — tout le reste de l'exposé s'y
appuie, à commencer par la slide suivante. C'est le premier enseignement que seule une
évaluation sur données réelles fait apparaître. Montrer du doigt sur la figure les
configurations à plus de 0,85 d'accuracy pour un F1 nul, et passer.

---

### Slide 8 — L'oubli d'EWC, mesuré
**Message** : sur le modèle du fil rouge, la bonne métrique fait apparaître un effondrement que
la lecture naïve masque.
**Durée** 1:00 · **Figure** `docs/figures/soutenance/s14_oubli_bilan.png`

Quatre lectures du **même modèle EWC multiclasse** : moyenne des F1 post-tâche **0,981** → F1 du
modèle final confronté à toutes les tâches **0,240** → carte en inférence gelée **0,243** →
carte en régime en ligne **0,507**. Oubli moyen **AF = 0,847**.

**Notes orateur** — Le moment fort du bloc, à raconter comme une enquête, et à rattacher
explicitement au fil : *« voilà ce que le modèle que je suis depuis le début oublie
réellement. »* L'écart 0,981 → 0,240 *est* l'oubli, et c'est la promesse de la slide 3 qui se
tient ici.

Puis le point méthodologique : la contre-performance embarquée avait d'abord été suspectée
d'être un bug de portage ; la parité 0,240 contre 0,243 a établi qu'il s'agissait d'oubli
fidèlement reproduit. **C'est le meilleur argument de l'exposé en faveur du dispositif de
parité** : sans lui, on aurait cherché un bug qui n'existait pas — et cela prépare la slide 12.
Signaler en passant que ce cas vient de CWRU (désamorce Q-37).

---

### Slide 9 — EWC face aux trois autres familles ⚠ SLIDE À RISQUE
**Message** : EWC domine, et le portage n'introduit pas de perte structurelle.
**Durée** 1:00 · **Figure** `docs/figures/soutenance/s15_grille_classement.png`

F1 de la classe fautif, condition 5feat, PC et carte côte à côte. **Seule slide comparative de
l'exposé** : TinyOL (architecture, tête entraînable), HDC (non neuronal) et Mahalanobis
(baseline non supervisée) n'apparaissent que là.

**Notes orateur — la slide la plus sensible de l'exposé, et la plus dense au regard de son
budget.** L'annoncer pour ce qu'elle est : *« je situe EWC une fois, et une seule. »* Trois
choses à dire, dans cet ordre, sans en sacrifier aucune :

1. **La lecture** : EWC au-dessus sur les trois jeux, l'écart se creusant sur Pronostia, le
   scénario class-incremental le plus dur.
2. **La limite, annoncée spontanément** : *« un mot sur la lecture de ces paires. Les deux
   colonnes ne portent pas le même effectif ni le même protocole — la colonne PC est
   l'évaluation d'apprentissage continu complète, la colonne carte est un flux rejoué en
   régime gelé sur un échantillon. Ce que ce tableau établit, c'est le classement des modèles,
   pas une comparaison au centième entre plateformes. La comparaison appariée rigoureuse, je
   vous la montre dans trois slides. »* → désamorce **Q-01 et Q-02**.
3. **La réserve TinyOL** : les deux colonnes de cette ligne ne portent pas la même
   architecture ; l'écart n'est pas imputable au seul portage.

Puis pointer **HDC : F1 nul sur carte pour une accuracy de 0,867** — deuxième illustration,
indépendante, du piège de la slide 7. *« Sans le F1, ce tableau se lisait comme un succès. »*
La grille complète sur les cinq jeux est en secours (**B2**), Paderborn en **B13**.

---

# Bloc 4 — Gap 2 : mesures sous contrainte (2:25)

> Bandeau : **Gap 2 — mesures sous contrainte**

### Slide 10 — La RAM : trois niveaux, et le total réel ⚠
**Message** : je publie les trois chiffres, je dis lequel est le mien — et `.bss` seul
sous-estime.
**Durée** 1:00 · **Figures** `docs/figures/soutenance/s17_ram_trois_niveaux.png` + `docs/figures/soutenance/s18_ram_totale_cascade.png`

Trois niveaux :
1. noyau minimal, tête EWC seule : ≈ **1 000 B** — établit la faisabilité
2. **système multi-modèle par défaut : `.bss` = 105 036 B, soit 40,1 %** — le chiffre du travail
3. pire cas, CMAPSS toutes variables (k = 21) : 183 936 B, soit 70,2 %

Et la correction : `.bss` ne compte que les globales et **exclut la pile**, où vivent les gros
tampons locaux. **RAM totale = `.data` + `.bss` + pic de pile** — sur Monitoring,
**460 + 100 152 + 4 688 = 105 300 B** (40,2 %) ; sur Pronostia, 110 192 B (42,0 %).

**Notes orateur** — Deux idées sur une slide, liées par la même exigence d'honnêteté ; les
enchaîner sans respirer entre les deux.

Sur les trois niveaux : *« publier uniquement le premier serait flatteur et malhonnête — c'est
ce que je reproche à la littérature, donc je m'y astreins. »* Désamorce Q-25.

Sur le total : assumer que les mesures antérieures du projet étaient optimistes. L'écart est
d'environ 4,5 Ko, donc les conclusions tiennent — *« mais il fallait le mesurer plutôt que le
supposer »*. C'est une slide qui démontre de la rigueur, pas une faiblesse. Le détail du pic
par phase (4 688 B en mise à jour contre 4 416 en inférence seule) part en secours **B14** :
ne le donner que si on le demande.

---

### Slide 11 — Latence : le surcoût de l'apprentissage en ligne
**Message** : apprendre en ligne coûte un facteur 5 — et c'est ce facteur que personne ne publie.
**Durée** 0:35 · **Figure** `docs/figures/soutenance/s20_latence_surcout.png`

Toujours la **tête EWC** : inférence seule **48–65 µs** (croissant avec k) · inférence + mise à
jour CL **239–340 µs**. Budget : 100 ms. Dispersion inter-modèles sur deux ordres de grandeur,
de ≈ 5 µs (Mahalanobis) à ≈ 2,1 ms (HDC INT8).

**Notes orateur** — Devancer l'objection « votre budget est trivialement satisfait » (Q-27) :
*« ce qui est intéressant n'est pas la marge, c'est le facteur 5 — c'est lui qui dimensionnera
un système à cadence plus élevée. »* Le coût de l'adaptation est ce que la littérature
embarquée passe sous silence. La latence des autres modèles est en secours (**B7**).
*Marge : la dispersion inter-modèles est marquée ⏱ dans `S05` (≈ 25 mots, ≈ 10 s) — c'est la
coupe la moins coûteuse de tout l'exposé, le chiffre vit en slide de secours **B7**.*

---

### Slide 12 — Parité PC ↔ carte ★ ARGUMENT SCIENTIFIQUE
**Message** : les chiffres mesurés sur carte décrivent bien le calcul de la référence.
**Durée** 0:50 · **Figure** `docs/figures/soutenance/s21_parite_desaccords.png`

- **Régime gelé : parité exactement 1,000**, aucun désaccord, sur **7 534 à 7 672 échantillons**
- **Régime en ligne : 0,963 à 0,989** — float32 sur carte contre float64 sur PC, divergences
  concentrées aux frontières de décision · **Δ acc_final ≤ 0,007**

**Notes orateur** — La slide qui justifie tout le chapitre 6 : *« mesurer une latence sur un
portage infidèle ne démontrerait rien. »* Reprendre telle quelle la formule posée slide 6 :
*« un écart en régime gelé signale un bug de portage ; un écart en régime en ligne, borné et
concentré sur les frontières de décision, est attendu. »* Rappeler que c'est ce dispositif qui
a tranché le cas de la slide 8 — le fil se referme ici.

Attention à l'axe tronqué de la figure — le signaler, sinon quelqu'un le signalera.
**Ne jamais rogner cette slide.**

---

# Bloc 5 — Gap 3 : quantification (2:25)

> Bandeau : **Gap 3 — quantification pendant l'entraînement**

### Slide 13 — Effondrement, diagnostic, récupération
**Message** : la perte venait d'un seul choix d'implémentation, pas de la quantification.
**Durée** 1:10 · **Figure** `docs/figures/soutenance/s23_effondrement_recuperation.png`
*(garder `docs/figures/soutenance/b4_ablation_echelle.png` en backup — cf. `S03`, B4)*

Toujours la **tête EWC**, en trois temps : **QAT côté PC préserve** (Δ ≤ 0,006) → **PTQ naïve
embarquée s'effondre** (F1 0,07–0,15 contre ≈ 0,92 en FP32) → **ablation** : une seule cause,
l'**échelle fixe 1/128 non calibrée** sur la dynamique réelle des poids → **noyau calibré**,
métrique intégralement récupérée **sur la cible matérielle**, parité exacte contre l'émulateur
bit-exact.

**Notes orateur** — Le récit est plus efficace que le résultat : trois causes suspectées,
testées séparément, une seule coupable. Souligner que l'accumulateur int32, coupable évident,
n'apporte rien. Et que la récupération est mesurée sur la carte, pas simulée. Le rattacher au
fil en une incise : *« c'est le même modèle depuis le début, cette fois en 8 bits. »*
L'effet du *moment* de la quantification est en secours (**B5**), la descente sous 8 bits en
**B6**.

---

### Slide 14 — Le gain est mémoire, pas temps ⚠
**Message** : ÷4 sur les poids, RAM système quasi inchangée, et sur un cœur à FPU l'INT8 est
plus **lent**.
**Durée** 1:15 · **Figures** `docs/figures/soutenance/s22_ram_deux_echelles.png` + `docs/figures/soutenance/s24_paradoxe_latence.png`

Gain RAM mesuré **×2,33 à ×4,0** selon modèle et jeu. Mais sur le firmware complet, la RAM
totale est pratiquement inchangée : les poids de la tête pèsent quelques centaines d'octets
dans une empreinte dominée par les tampons du pipeline multi-modèle.

Et la latence : INT8 **74 µs** contre **48–50 µs** en FP32, soit +24 à +26 µs. Décomposition à
180 MHz : MAC entier ≈ **6 785 cycles** (n'avance pas sur le flottant) · **requantification
≈ 3 777** (le surcoût dominant, un arrondi flottant par neurone) · déquantification ≈ 200.

**Notes orateur** — Deux résultats nuancés sur une slide : les tenir tous les deux, c'est le
prix du format court, mais c'est aussi le bilan honnête promis slide 4.

Sur la RAM, montrer les deux échelles côte à côte et concéder franchement : *« le gain est réel
et décisif sur le modèle, marginal sur le système dans cette configuration. »* Une concession
volontaire vaut mieux qu'une objection subie (Q-28).

Sur la latence, assumer pleinement le résultat le plus contre-intuitif de l'exposé — l'attente
a été créée slide 5 : *« la FPU du Cortex-M4 exécute le flottant aussi vite que le cœur exécute
l'entier, si bien que les deux étages de conversion s'ajoutent en pure perte. »* Puis le
retourner en positif : sur un cœur **sans FPU**, la conclusion s'inverserait — ce que je publie
n'est pas « l'INT8 est inutile » mais « son bénéfice dépend de la cible et doit être mesuré
dessus ». Mentionner CMSIS-NN comme piste non explorée, avec le rendement prévisible depuis
cette décomposition.

---

# Bloc 6 — Clôture (1:20)

### Slide 15 — Bilan et perspectives
**Message** : deux gaps démontrés, un troisième au bilan nuancé, une méthode réutilisable.
**Durée** 1:20 · **Figures** `docs/figures/soutenance/s26_bilan_gaps.png` (reprise de la
slide 4, avec un statut par gap) puis `docs/figures/soutenance/perspectives_trois_axes.png`

- **Gap 1 ✔** — quatre familles validées en régime incrémental sur données industrielles
  réelles, sur PC et sur carte, avec des métriques CL complètes
- **Gap 2 ✔** — RAM mesurée composant par composant (105 300 B, 40 % de l'enveloppe), latences
  très en deçà de 100 ms, **parité de portage prouvée prédiction par prédiction**
- **Gap 3 ~** — QAT préserve la métrique côté PC ; la PTQ embarquée naïve s'est dégradée avant
  d'être récupérée par un noyau calibré ; **la rétropropagation quantifiée sur carte reste à
  faire**, et l'arbitrage énergétique reste ouvert

**Perspectives — un second visuel, en apparition sur la même slide.** Trois axes, chacun une
rangée de quatre jalons (*mesuré sur carte* → *verrou identifié* → *verrou levé* →
*déployable*), pleins jusqu'où le travail est allé :

- **décider seul quand se mettre à jour** — 97 % des mises à jour économisées, parité de
  verdict 1,000, +300 o ; **verrou : en auto-étiquetage, F1 0,504 au lieu de 0,889**
- **choisir seul les variables d'entrée** — F1 0,381 → 0,615, mais 105 036 → 184 864 o ;
  **verrou : arbitrage encore manuel, couple par couple**
- **mesurer l'énergie** — rangée entièrement creuse et grise : chaîne prête, sonde non posée,
  **aucun chiffre publié**

**Notes orateur** — Le schéma fait le travail que trois phrases juxtaposées ne font pas : il
montre que les trois axes **ne sont pas au même stade**. Deux sont instrumentés et butent sur
un verrou nommé ; le troisième n'a pas commencé à être mesuré. Dire la lecture à voix haute —
*« la marche entre le plein et le creux, c'est ce qu'il reste à faire »* — puis ne commenter
que la ligne rouge de chaque rangée.

Préciser, si on est poussé, que **les jalons sont un statut de projet, pas une grandeur
mesurée** — c'est écrit en pied de figure. Seuls les chiffres de la colonne de droite sont des
mesures.

Donner le 0,504 spontanément : c'est le verrou, et l'annoncer soi-même est ce qui rend crédible
tout le reste (désamorce Q-51). Le détail du gate autonome est en secours (**B9**), la détection
de dérive en **B10**.

Finir sur la contribution méthodologique, pas sur les chiffres :
*« au-delà du cas d'étude, ce qui se réutilise est une chaîne de portage à parité garantie par
construction et un protocole de mesure qui distingue systématiquement ce qui est mesuré de ce
qui est estimé. C'est cette discipline, autant que les résultats, qui donne sa portée à la
démonstration. »* Puis rendre la parole.

---

## Traçabilité — d'où vient chaque slide

Le plan précédent comptait 26 slides pour 31 min. Aucune n'a été effacée : chacune est soit
dans le fil, soit fusionnée, soit descendue en secours dans [`S03`](S03_slides_backup.md).

| Nouvelle | Ancienne(s) | Traitement |
|---|---|---|
| 1 | 1 | inchangée |
| 2 | 2 + 3 | fusion (problème + obstacles au réentraînement) |
| 3 | 4 + 9 | fusion (oubli catastrophique + EWC comme réponse) |
| 4 | 5 + 6 | fusion (état de l'art + triple gap), pivot |
| 5 | 7 + 8 | fusion (carte + jeux de données) |
| 6 | 10 + 11 | fusion (chaîne de portage + protocole) ; **12 repliée** en notes → **B16** |
| 7 | 13 | inchangée |
| 8 | 14 | recadrée « fil rouge EWC » |
| 9 | 15 | recadrée « seule slide comparative » ; 16 → **B13** |
| 10 | 17 + 18 | fusion (trois niveaux + total réel) ; 19 → **B14** |
| 11 | 20 | inchangée |
| 12 | 21 | inchangée |
| 13 | 23 | recadrée « fil rouge EWC » ; 22 déplacée en 14 |
| 14 | 22 + 24 | fusion (deux échelles de RAM + paradoxe de latence) |
| 15 | 26 + 25 | fusion (bilan + perspectives condensées en trois phrases) |

### Figures désormais hors du fil

Quatre figures du catalogue ne sont plus projetées. Aucune n'est à supprimer — le catalogue
les régénère, et elles servent ailleurs :

| Figure | Où elle vit maintenant |
|---|---|
| `s4_oubli_mesure.png` | **variante de la slide 3**, à réintroduire si la répétition la juge trop théorique (cf. encadré slide 3) |
| `s9_modeles.png` (tab. 4.2, les quatre familles) | absorbée par la slide 9, qui montre les quatre familles avec leurs résultats plutôt qu'en catalogue |
| `s7_budget_ram.png` | redondante avec la slide 10, qui donne les trois niveaux **et** le total |
| `s25_gate_economie.png` | slide de secours **B9**, qui portait déjà `b9_economie.png` et `b9_parite_gate.png` |
| `s12_gele_vs_en_ligne.png` · `s16_paderborn_ewc_seul.png` · `s19_pile_par_phase.png` | slides de secours **B16**, **B13**, **B14** |

---

## Répétition — points de contrôle

- [ ] Chronométrer une fois à voix haute : viser **15 min ± 30 s**. À ce budget il n'y a pas de
      marge — si ça déborde, le premier réflexe est de comprimer les slides **2, 5 et 11**,
      jamais les slides 4, 6, 9 et 12.
- [ ] Vérifier que le **fil EWC** s'entend : le modèle doit être nommé slides 3, 4, 8, 11, 13
      et 14. Si un auditeur ne peut pas dire à la fin « il a suivi un modèle », le principe 2
      n'est pas tenu.
- [ ] Vérifier que les slides **9, 10, 14** contiennent bien l'auto-annonce de la limite.
- [ ] Vérifier qu'aucun chiffre prononcé n'est absent du manuscrit (voir l'annexe de `S01`).
- [ ] Répéter spécifiquement les transitions 6 → 7, 9 → 10 et 12 → 13 : ce sont les changements
      de gap, et le jury doit les entendre.
- [ ] Vérifier la lisibilité des figures **projetées** : les slides 4, 10 et 14 portent
      désormais **deux figures chacune** — c'est le risque principal de ce format condensé,
      arbitrer la taille sur écran réel avant de figer le deck.

---

> **[`S05_script_oral.md`](S05_script_oral.md) est aligné sur ce plan** : même découpage en 15
> slides, même minutage, ≈ 2 350 mots. `S02` dit *quoi montrer*, `S05` dit *quoi dire* — toute
> modification de l'un doit être répercutée dans l'autre.
   
