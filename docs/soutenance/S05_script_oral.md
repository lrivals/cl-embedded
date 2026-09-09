# S05 — Script oral de la soutenance

**Le texte parlé intégral des 15 minutes**, slide par slide. Version détaillée de
[`S02_plan_presentation.md`](S02_plan_presentation.md), qui reste le plan de construction du deck :
`S02` dit *quoi montrer*, `S05` dit *quoi dire*.

**Ce document se répète à voix haute — il ne se lit pas en salle.** Le but est qu'après trois
lectures les phrases viennent seules ; les six formulations à savoir vraiment mot à mot sont
rassemblées en fin de document.

**Public** : un jury qui n'a lu que le manuscrit. Ni le dépôt, ni les JSON, ni les notebooks.
Chaque chiffre prononcé ici figure dans l'annexe « chiffres à connaître par cœur » de
[`S01`](S01_questions_jury.md) — donc dans le manuscrit, donc opposable. La seule exception est
signalée ⓘ slide 3.

> **Ce script est calibré sur 15 minutes**, conformément au format réel de la soutenance
> (15 min d'exposé + 15 min de questions). Il remplace une version à 30 min et 26 slides. Deux
> conséquences pour celui qui répète : **le débit ne peut plus absorber les approximations** — chaque
> phrase porte — et **l'exposé suit un seul modèle, EWC, de bout en bout**. Ce qui a quitté le fil
> vit dans [`S03`](S03_slides_backup.md) et se sort pendant les questions.

---

## Budget de mots — à lire avant la première répétition

Le script fait **2 630 mots prononcés**. Ce que cela vaut en minutes dépend entièrement du débit, et
il faut le savoir avant de répéter :

| Débit | Durée |
|---|---|
| 155 mots/min (français soutenu, posé) | **17:00** — trop long |
| 165 mots/min | **16:00** — encore long |
| **175 mots/min (débit soutenu d'un script répété)** | **15:00** — la cible |

**Autrement dit : ce script tient en quinze minutes à condition d'être répété.** Lu à froid, il en
fait dix-sept. Ce n'est pas une erreur de calibration, c'est un choix assumé : descendre à
2 300 mots obligeait à retirer des arguments — la limite annoncée slide 9, le diagnostic en trois
temps slide 13, le verrou du 0,504 slide 15 — et un exposé plus court qui a perdu ses arguments est
un moins bon exposé qu'un exposé dense bien répété.

| Bloc | Slides | Durée visée | Mots |
|---|---|---|---|
| 1 — Contexte : le problème et EWC | 1–3 | 2:50 | 497 |
| 2 — Positionnement et méthode | 4–6 | 3:20 | 592 |
| 3 — Gap 1 : données industrielles | 7–9 | 2:40 | 438 |
| 4 — Gap 2 : mesures sous contrainte | 10–12 | 2:25 | 425 |
| 5 — Gap 3 : quantification | 13–14 | 2:25 | 400 |
| 6 — Clôture | 15 | 1:20 | 278 |
| **Total** | **15** | **15:00** | **2 630** |

**Si la troisième répétition dépasse encore**, couper dans cet ordre, et pas dans un autre :

1. les trois paragraphes marqués ⏱ (slides 2, 5 et 11) — ≈ 60 mots ;
2. slide 4, la description des quatre travaux → n'en garder que deux, TinyOL et QLR-CL, les plus
   proches du sujet — ≈ 60 mots ;
3. slide 3, le détail des trois flèches du schéma de perte → « sans contrainte on sort du bassin A,
   avec EWC on rejoint B en y restant » — ≈ 40 mots ;
4. slide 14, la décomposition au cycle près → ne garder que la requantification, le poste dominant —
   ≈ 45 mots.

**Ne jamais couper** les slides 6, 9, 12 et 15 : elles portent la contribution méthodologique, la
limite annoncée, l'argument scientifique et la chute.

**Quatre règles de conduite**, héritées de `S02` :
un message par slide · **le fil rouge est un modèle, pas une liste de modèles** · la slide 4 pose le
triple gap, rappelé aux slides 7, 10 et 13 · **annoncer les limites avant que le jury ne les
trouve** — slides 9, 10 et 14, signalées ⚠ dans ce document.

---
# Bloc 1 — Contexte : le problème et EWC

## Slide 1 — Titre · 0:20 · aucune figure

> **Message** : poser la thèse en un souffle.

Merci de m'accueillir. Je suis Léonard Rivals ; je vous présente mon stage de master, mené à
l'ISAE-SUPAERO avec l'ENAC et Edge Spectrum, encadré par Arnaud Dion, Dorra Ben Khalifa et
Frédéric Zbierski. **Je vais vous parler de modèles qui continuent d'apprendre après leur déploiement,
sur un microcontrôleur à 256 Ko de mémoire — et surtout de ce qu'il en coûte, mesuré.**

**Transition** → Commençons par le problème.

---

## Slide 2 — La machine change, et on ne peut pas réentraîner · 1:10 · `s2_cas_usage.png` + `s3_cycle_de_vie.png`

> **Message** : un modèle déployé se dégrade parce que la machine change — et les trois portes de
> sortie habituelles sont fermées.

La situation type : une machine industrielle, des capteurs, un modèle de détection de panne au plus
près du capteur. Le jour du déploiement, il est bon — entraîné sur des données représentatives de la
machine telle qu'elle était alors.

Puis le temps passe. La machine vieillit, le capteur dérive, les conditions opératoires changent, et
la distribution qui arrive au modèle n'est plus celle de son entraînement. C'est la **dérive de
distribution**, graduelle ou abrupte, sur les seules variables d'entrée ou sur le domaine entier. Le
résultat : **la performance décroît sans que rien ne signale que quelque chose s'est cassé.**

Réentraîner suppose trois choses, et sur un microcontrôleur les trois font défaut. **La mémoire** :
le réentraînement rejoue l'historique complet, et il n'y a pas la place de le stocker. **La
connectivité** : un lien disponible, et l'accord de l'industriel pour que ses données sortent du
site.

⏱ *coupe si retard* — **La réactivité** : même quand la boucle centralisée existe, elle prend des
heures. Une dérive détectée le lundi est corrigée le vendredi.

Donc le modèle doit apprendre sur place, à partir du flux, sans revoir le passé.

**Transition** → Et c'est là qu'un obstacle propre à l'apprentissage apparaît.

---

## Slide 3 — L'obstacle et la réponse : oubli catastrophique → EWC · 1:20 · `KirkArticle_Plot.png` + `KirkArticle_SchemaLoss.png`

> **Message** : apprendre du nouveau détruit l'ancien — et EWC répond en rendant rigides les poids
> importants.

Quand un réseau apprend une nouvelle tâche, la descente de gradient optimise cette tâche-là et rien
d'autre : elle déplace les poids vers un minimum de l'erreur courante, sans aucune raison de
préserver les représentations des tâches précédentes. Elles sont écrasées — c'est **l'oubli
catastrophique**. Le point à retenir, parce qu'il fonde tout mon chapitre 5 : **ce n'est pas un bug,
c'est une conséquence de l'optimisation.**

La réponse que je retiens s'appelle **EWC** : une pénalité quadratique pondérée par la diagonale de
la matrice de Fisher, qui rend rigides les poids importants pour le passé. Je n'écris pas la
formule, ce schéma la remplace — il vient de l'article de Kirkpatrick, de 2017. Les deux ellipses
sont les régions de faible erreur des tâches A et B : sans contrainte, on sort du bassin A ; avec
une simple pénalité L2, on ne bouge plus du tout ; avec EWC, **on rejoint B en restant dans A**.

À gauche, la figure du même article : EWC y **tient** la tâche A. Je vous préviens — dans ma mesure,
il ne la tiendra pas. La différence tient au régime : chez Kirkpatrick, des tâches de même nature et
de même difficulté ; chez moi, un jeu de roulements où chaque tâche introduit des classes inédites.
**La régularisation suffit dans le régime de l'article ; sur nos données industrielles, elle ne
suffit pas — c'est un résultat du travail, pas un échec de mise en œuvre.**

> ⓘ **Si le jury pousse sur ce point** : l'oubli moyen mesuré vaut **0,858 sans EWC et 0,848 avec**.
> Ce sont les **seuls chiffres du script absents du manuscrit** (campagne `exp_S54_forgetting` du
> dépôt) — les annoncer comme une mesure complémentaire. Ne pas les prononcer spontanément : le
> chiffre d'oubli opposable arrive slide 8.
>
> ⓘ **Si le jury pousse sur l'écart à l'article** (« qu'avez-vous fait de plus que Kirkpatrick ? ») :
> tout est préparé dans l'**annexe EWC** en fin de document — le support d'exécution de l'article,
> les quatre éléments repris tels quels, le tableau des neuf écarts avec leur justification, et la
> déviation du proxy `w²` à annoncer soi-même.

**Transition** → Voyons ce que la littérature a résolu, et ce qu'elle laisse ouvert.

---

# Bloc 2 — Positionnement et méthode

## Slide 4 — L'état de l'art et le triple gap ★ · 1:15 · `s5_etat_de_lart.png` + `s6_triple_gap.png`

> **Message** : quatre travaux, chacun laisse un côté ouvert — trois lacunes qu'aucun ne comble
> ensemble.

Je ne lis pas ce tableau ; ce qui m'intéresse est la colonne de droite.

**TinyOL** apprend en ligne sur un Cortex-M4, pour environ 10 % de latence en plus — mais ne
décompose pas l'empreinte mémoire. **QLR-CL** quantifie le tampon de rejeu en 8 bits et divise
l'empreinte par 4 — mais l'entraînement reste en flottant. **Benatti** démontre un apprentissage sans
gradient sur une plateforme très basse consommation — hors de tout cadre de maintenance prédictive.
**LifeLearner** tient dans 212 Ko sur un Cortex-M7 — mais sans mesurer le surcoût de l'adaptation.

Mises bout à bout, ces colonnes dessinent **trois lacunes qu'à ma connaissance aucun travail ne
comble simultanément** : la validation sur **données industrielles temporelles réelles** avec un
protocole reproductible ; la démonstration **sous contrainte mémoire**, avec des chiffres mesurés
composant par composant et non une enveloppe annoncée ; la **quantification en 8 bits pendant
l'entraînement**, et non seulement à l'inférence.

Ces trois gaps sont le plan de mon exposé. **Et pour que quinze minutes suffisent, je les traite sur
un seul modèle — EWC — que je suis jusqu'au bout : ce qu'il oublie, ce qu'il coûte, ce qu'il devient
quantifié.** Les autres familles serviront une fois, à le situer. Enfin, je vous le dis plutôt que
de vous laisser le découvrir : **le bilan du troisième gap sera nuancé.**

**Transition** → Un mot sur l'enveloppe matérielle et les données.

---

## Slide 5 — La cible et le banc · 0:45 · `s7_fiche_carte.png` + `s8_jeux_donnees.png`

> **Message** : voilà l'enveloppe dans laquelle tout doit tenir.

Une carte **NUCLEO-F439ZI** : Cortex-M4 à 180 MHz, unité de calcul flottant, **256 Ko** de mémoire
vive, **pas d'accélérateur neuronal**. Budget que je me fixe : **100 ms** par inférence plus mise à
jour. La cible initiale du stage était une STM32N6 avec accélérateur ; elle n'était pas disponible,
et j'ai préféré du matériel réellement en main.

⏱ *coupe si retard* — Retenez que **cette unité de calcul flottant produira le résultat le plus
contre-intuitif de l'exposé.**

Côté données, six jeux industriels publics incarnant **deux régimes d'oubli**. L'**incrémental par
domaine** : les classes restent, la distribution change — une surveillance d'équipements découpée
par type de machine. L'**incrémental par classe** : de nouvelles classes apparaissent — Pronostia,
où la faute n'apparaît qu'après une phase normale, le régime le plus dur.

**Transition** → Le cœur méthodologique est ailleurs : dans le passage du PC à la carte.

---

## Slide 6 — La chaîne de portage et le protocole de mesure ★ · 1:20 · `s10_chaine_portage.png` + `s11_protocole_mesure.png`

> **Message** : la parité n'est pas espérée, elle est garantie par construction — et chaque grandeur
> a sa méthode de mesure.

C'est ma contribution méthodologique, et je la présente comme telle.

Quatre étapes : entraînement en Python, **export des poids vers des en-têtes C générés**,
compilation du firmware en flottant pour le Cortex-M4, liaison série entre l'hôte et la carte. Trois
invariants la tiennent : **les poids sont générés, jamais écrits à la main** ; **le format de trame
est figé**, toute évolution du firmware étant répercutée côté hôte dans le même mouvement ; **la
sélection des variables a une source unique** — les deux plateformes consomment littéralement les
mêmes colonnes. La parité est donc **attendue par construction** — mais je la vérifie quand même,
prédiction par prédiction.

Côté mesures, chaque grandeur a sa méthode : **la latence** par le compteur de cycles du cœur,
toujours séparée entre inférence seule et inférence plus mise à jour ; **la mémoire statique** par
les symboles de l'éditeur de liens ; **la pile** par peinture de pile — on remplit la zone libre d'un
motif connu, on relit jusqu'où il a été écrasé ; **l'intégrité** par un code de redondance cyclique,
zéro erreur. Ces méthodes donnent **ce qui s'est produit**, pas un pire cas théorique.

Enfin la règle de lecture, qui vaut pour toute la suite : mêmes poids, mêmes échantillons, même
ordre, même graine. **En régime gelé la parité doit être exacte ; en régime en ligne elle est
approchée, la carte calculant en 32 bits là où le PC calcule en 64.**

**Transition** → *(changement de gap — marquer la pause)* Passons au premier gap.

---

# Bloc 3 — Gap 1 : validation sur données industrielles

> Bandeau : **Gap 1 — données industrielles réelles**

## Slide 7 — L'accuracy est trompeuse · 0:40 · `s13_accuracy_trompeuse.png`

> **Message** : sur des jeux déséquilibrés, l'accuracy peut masquer un modèle qui ne détecte rien.

Premier enseignement, et il n'apparaît que sur données réelles.

Une cellule de ma grille : le détecteur de Mahalanobis sur CMAPSS. Sur carte, **0,853 d'accuracy** —
ce qui a l'air très bon — **pour un F1 de 0,214**. Les échantillons sains sont largement
majoritaires : un modèle qui prédit systématiquement « tout va bien » obtient une accuracy élevée
**tout en ne détectant aucune panne**.

D'où mon choix pour tout le reste de l'exposé : **le F1 de la classe fautif comme métrique de
référence**. Regardez les configurations au-delà de 0,85 d'accuracy — plusieurs ont un F1 nul.

**Transition** → La même exigence va faire apparaître bien pire, sur mon modèle.

---

## Slide 8 — L'oubli d'EWC, mesuré · 1:00 · `s14_oubli_bilan.png`

> **Message** : sur le modèle du fil rouge, la bonne métrique fait apparaître un effondrement.

Voici ce que le modèle que je suis depuis le début oublie réellement. Je vous le raconte comme une
enquête, parce que c'est ainsi que ça s'est déroulé.

Un modèle EWC multiclasse, entraîné en continu. **Première lecture** : la moyenne des F1 mesurés
juste après chaque tâche vaut **0,981**. Excellent — et parfaitement trompeur. **Deuxième lecture** :
je confronte le modèle final à toutes les tâches vues, ensemble. Le F1 tombe à **0,240**. L'écart
entre ces deux nombres, **c'est l'oubli** ; oubli moyen **0,847**. **Troisième lecture, sur carte, en
inférence gelée : 0,243.**

Et c'est là que le dispositif de parité prend toute sa valeur. Cette contre-performance embarquée
avait d'abord été suspectée d'être un bug de portage. La parité entre 0,240 et 0,243 a établi qu'il
n'en était rien : **la carte reproduisait fidèlement l'oubli du modèle de référence.** Sans ce
dispositif, j'aurais cherché un bug qui n'existait pas. En régime en ligne, la carte remonte à
0,507 ; et ce cas vient du jeu de roulements CWRU.

**Transition** → Situons maintenant EWC par rapport aux autres familles — une fois, et une seule.

---

## Slide 9 — EWC face aux trois autres familles ⚠ · 1:00 · `s15_grille_classement.png`

> **Message** : EWC domine, et le portage n'introduit pas de perte structurelle.

⚠ **Slide la plus sensible — les trois points, dans cet ordre, sans en sacrifier un.**

**La lecture.** F1 de la classe fautif, PC et carte côte à côte, pour les quatre familles. **EWC est
au-dessus sur les trois jeux**, et l'écart se creuse sur Pronostia, le scénario incrémental par
classe.

⚠ **La limite, que je préfère annoncer moi-même.** *« Les deux colonnes ne portent pas le même
effectif ni le même protocole — la colonne PC est l'évaluation d'apprentissage continu complète, la
colonne carte un flux rejoué en régime gelé sur un échantillon. Ce que ce tableau établit, c'est le
classement des modèles, pas une comparaison au centième entre plateformes. La comparaison appariée
rigoureuse, je vous la montre dans trois slides. »*

**Une réserve sur TinyOL** : ses deux colonnes ne portent pas la même architecture, l'écart n'est
pas imputable au seul portage.

Enfin le calcul hyperdimensionnel : sur carte, **F1 nul pour une accuracy de 0,867**. **Sans le F1,
ce tableau se lisait comme un succès.**

**Transition** → *(changement de gap — marquer la pause)* Voilà ce que ces modèles savent faire ;
voyons ce qu'ils coûtent.

---

# Bloc 4 — Gap 2 : mesures sous contrainte

> Bandeau : **Gap 2 — mesures sous contrainte**

## Slide 10 — La RAM : trois niveaux, et le total réel ⚠ · 1:00 · `s17_ram_trois_niveaux.png` + `s18_ram_totale_cascade.png`

> **Message** : je publie les trois chiffres, je dis lequel est le mien — et les globales seules
> sous-estiment.

⚠ Il existe trois chiffres de mémoire dans ce travail, et je les publie tous les trois. **Le noyau
minimal**, la tête EWC seule : environ **1 000 octets** — un apprentissage incrémental tient dans un
kilo-octet. **Le système multi-modèle par défaut** : **105 036 octets, 40,1 %** de l'enveloppe —
**c'est le chiffre de mon travail**. **Le pire cas mesuré**, CMAPSS avec ses 21 variables :
**183 936 octets, 70,2 %**.

Pourquoi les trois ? *« Publier uniquement le premier serait flatteur et malhonnête — c'est ce que
je reproche à la littérature, donc je m'y astreins. »*

Et même le deuxième était incomplet : les mesures antérieures ne remontaient que les variables
globales non initialisées, or cette section **exclut la pile**, là où vivent les gros tampons locaux.
La mesure complète additionne trois composantes : **460, plus 100 152, plus 4 688, égale 105 300
octets**. Mes mesures antérieures étaient donc optimistes de quatre kilo-octets et demi : les
conclusions tiennent, **mais il fallait le mesurer plutôt que le supposer.**

**Transition** → Le même raisonnement, appliqué au temps, donne un résultat plus marqué.

---

## Slide 11 — Latence : le surcoût de l'apprentissage en ligne · 0:35 · `s20_latence_surcout.png`

> **Message** : apprendre en ligne coûte un facteur 5 — et c'est ce facteur que personne ne publie.

Toujours sur la tête EWC : l'inférence seule prend entre **48 et 65 µs**, l'inférence **plus** la
mise à jour entre **239 et 340 µs**. Le budget était de 100 ms.

Je devance l'objection : oui, il est trivialement satisfait. *« Ce qui est intéressant n'est pas la
marge, c'est le facteur 5 — c'est lui qui dimensionnera un système à cadence plus élevée. »*

⏱ *coupe si retard* — La dispersion entre modèles couvre deux ordres de grandeur, de **5 µs** pour
Mahalanobis à **2,1 ms** pour le calcul hyperdimensionnel en 8 bits.

**Transition** → Tous ces chiffres reposent sur une hypothèse que je dois démontrer.

---

## Slide 12 — Parité PC ↔ carte ★ · 0:50 · `s21_parite_desaccords.png`

> **Message** : les chiffres mesurés sur carte décrivent bien le calcul de la référence.

⚠ **Ne jamais rogner cette slide.** Elle justifie tout ce qui précède : *« mesurer une latence sur un
portage infidèle ne démontrerait rien. »*

**En régime gelé, la parité est exactement de 1,000 — aucun désaccord, sur 7 534 à 7 672
échantillons.** **En régime en ligne, entre 0,963 et 0,989** : les divergences se concentrent sur les
frontières de décision, là où un écart d'arrondi entre 32 et 64 bits fait basculer une décision.
L'écart d'accuracy finale reste inférieur ou égal à **0,007**.

Je reprends la règle annoncée tout à l'heure : **un écart en régime gelé signale un bug de portage ;
un écart en régime en ligne, borné et concentré sur les frontières, est attendu.** Ici le premier est
nul, le second borné — c'est ce dispositif qui a tranché le cas de l'oubli, il y a quatre slides.

Une précision : **l'axe vertical est tronqué.** Lisez les valeurs, pas les hauteurs.

**Transition** → *(changement de gap — marquer la pause)* Reste le troisième gap, celui dont le
bilan sera nuancé.

---

# Bloc 5 — Gap 3 : quantification

> Bandeau : **Gap 3 — quantification pendant l'entraînement**

## Slide 13 — Effondrement, diagnostic, récupération · 1:10 · `s23_effondrement_recuperation.png`

> **Message** : la perte venait d'un seul choix d'implémentation, pas de la quantification.

C'est le même modèle depuis le début, cette fois en 8 bits. Trois temps, et le récit importe autant
que le résultat.

**Premier temps.** Côté PC, la quantification **pendant** l'entraînement préserve la métrique :
l'écart est inférieur ou égal à **0,006**.

**Deuxième temps.** Le même modèle, quantifié naïvement **après** entraînement, sur la carte,
**s'effondre** : le F1 tombe entre **0,07 et 0,15**, contre environ **0,92** en flottant. Un modèle
inutilisable.

**Troisième temps, le diagnostic.** Trois causes suspectées — l'accumulateur entier, l'ordre des
opérations, le choix de l'échelle — testées séparément. L'accumulateur, coupable évident, **n'apporte
rien**. La cause est unique : **une échelle fixe de 1/128, non calibrée sur la dynamique réelle des
poids**, qui ne remplissaient donc pas la plage disponible.

**Quatrième temps, la récupération.** Avec un noyau dont l'échelle est calibrée, la métrique est
intégralement récupérée — et **cette récupération est mesurée sur la carte, pas simulée**, avec une
parité exacte contre l'émulateur bit à bit.

Autrement dit : ce n'est pas la quantification qui dégradait le modèle, c'est un choix
d'implémentation dans la quantification.

**Transition** → Reste ce qu'elle rapporte vraiment — et là, la carte m'a donné tort.

---

## Slide 14 — Le gain est mémoire, pas temps ⚠ · 1:15 · `s22_ram_deux_echelles.png` + `s24_paradoxe_latence.png`

> **Message** : ÷4 sur les poids, RAM système inchangée, et sur un cœur à FPU l'INT8 est plus lent.

⚠ Deux résultats nuancés, et je donne les deux.

**Le premier concerne la mémoire, à deux échelles.** À l'échelle du modèle, le gain est réel et
mesuré : de **2,33 à 4 fois moins de mémoire**. À l'échelle du système, la mémoire totale est
**pratiquement inchangée** — les poids de la tête pèsent quelques centaines d'octets dans une
empreinte dominée par les tampons du pipeline multi-modèle. **Le gain est décisif sur le modèle,
marginal sur le système dans cette configuration.**

**Le second concerne le temps, et c'est le résultat le plus contre-intuitif de l'exposé.** En 8 bits,
l'inférence prend **74 µs** ; en flottant, **48 à 50** : la quantification rend le calcul **plus
lent** d'environ 50 %. Au cycle près : la multiplication-accumulation entière coûte environ **6 785
cycles** — elle n'avance pas sur le flottant ; la **requantification**, environ **3 777**, soit le
surcoût dominant, un arrondi flottant par neurone ; la déquantification, environ 200. *« L'unité de
calcul flottant exécute le flottant aussi vite que le cœur exécute l'entier, si bien que les deux
étages de conversion s'ajoutent en pure perte. »*

Sur un cœur **sans** unité flottante, la conclusion s'inverserait. Ce que je publie n'est donc pas
« la quantification en 8 bits est inutile », mais **« son bénéfice dépend de la cible et doit être
mesuré dessus »**.

**Transition** → Je conclus.

---

# Bloc 6 — Clôture

## Slide 15 — Bilan et perspectives · 1:20 · `s26_bilan_gaps.png` + `perspectives_trois_axes.png`

> **Message** : deux gaps démontrés, un troisième au bilan nuancé, une méthode réutilisable.

Je reviens à mes trois gaps, avec un statut pour chacun.

**Le premier est comblé** : quatre familles validées en régime incrémental sur données industrielles
réelles, sur PC et sur carte — et une leçon de méthode, l'accuracy seule aurait donné une lecture
fausse.

**Le deuxième est comblé** : mémoire mesurée composant par composant, 105 300 octets, 40 % de
l'enveloppe ; latences très en deçà des 100 ms ; et surtout **parité du portage prouvée prédiction
par prédiction**, pas supposée.

**Le troisième est au bilan nuancé**, comme annoncé : la quantification pendant l'entraînement
préserve la métrique côté PC, la quantification naïve après entraînement s'est effondrée sur la
carte avant d'être récupérée par un noyau calibré — mais **la rétropropagation entièrement quantifiée
sur carte reste à faire**, et l'arbitrage énergétique reste ouvert.

Trois prolongements. **Automatiser la décision de mise à jour** : un déclencheur autonome mesuré sur
carte économise **97 % des mises à jour** — mais en auto-étiquetage complet, **le F1 tombe à 0,504
sur Pronostia**, et c'est là qu'est le vrai travail restant. **Automatiser la sélection des
variables**. Et **mesurer l'énergie** : la chaîne est prête, la mesure n'a pas pu être réalisée —
vous ne trouverez donc **aucun chiffre d'énergie** ici, je préfère un manque assumé à une estimation.

Je voudrais finir sur autre chose que des chiffres. *« Au-delà du cas d'étude, ce qui se réutilise
est une chaîne de portage à parité garantie par construction, et un protocole de mesure qui
distingue systématiquement ce qui est mesuré de ce qui est estimé. C'est cette discipline, autant
que les résultats, qui donne sa portée à la démonstration. »*

Je vous remercie de votre attention.

---

# Annexe — EWC : l'article de 2017 et ce que j'en ai fait

**Hors script parlé.** Le budget de la slide 3 est de 1:20 et ne peut pas absorber ce qui suit.
Cette annexe arme la **phase de questions** : Q-05, Q-07, Q-10, et toute variante de *« qu'avez-vous
fait de plus que l'article ? »*. À réviser en compréhension, pas à réciter.

## Le support du modèle dans l'article

Kirkpatrick et al. démontrent EWC sur **deux supports, tous deux hors contrainte matérielle** :

- **MNIST permuté** — une suite de tâches obtenues en permutant les pixels, apprises par des
  réseaux entièrement connectés entraînés par SGD avec dropout ;
- **Atari 2600** — des agents d'apprentissage par renforcement profond (DQN) enchaînant dix jeux,
  donc des réseaux convolutifs de plusieurs ordres de grandeur au-dessus de ma tête.

Dans les deux cas : **GPU, hors ligne, accès complet aux données** de chaque tâche pour estimer la
Fisher, précision flottante, aucune contrainte de mémoire ni de latence. Sur MNIST, les **frontières
de tâche sont données** ; c'est seulement dans le volet Atari que les auteurs ajoutent un mécanisme
d'inférence de tâche en ligne — l'ancêtre le plus direct de mon gate de nouveauté.

**La phrase à avoir prête** : *« l'article démontre que la méthode fonctionne ; il ne dit rien de ce
qu'elle coûte quand on la met sur un microcontrôleur. C'est exactement l'espace que j'occupe. »*

## Ce que j'ai repris tel quel

Quatre choses, et il faut le dire sans détour — **le cœur de la méthode n'est pas de moi** :

1. **La pénalité elle-même**, à l'identique : `L(θ) = L_tâche(θ) + (λ/2) · Σᵢ Fᵢ (θᵢ − θ*ᵢ)²`
   (`ewc_mlp.py::ewc_loss`, et le même terme dans `ewc_head.c::ewc_sgd_step` sur la carte).
2. **L'approximation diagonale de la Fisher.** À souligner si on me la reproche : **c'est déjà
   l'approximation de l'article**, pas une simplification que la contrainte embarquée m'aurait
   imposée. La contrainte la rend seulement *non négociable* (cf. Q-05).
3. **L'estimation de la Fisher par le carré du gradient de la log-vraisemblance**, côté PC —
   `fisher.py::compute_fisher_diagonal`, sur 200 échantillons, en fin de tâche.
4. **Le régime d'optimisation** : SGD, λ fixe, ReLU, pas de BatchNorm, pas d'optimiseur adaptatif.

## Ce que j'ai changé, et pourquoi

| Élément | Dans l'article | Dans mon travail | Pourquoi |
|---|---|---|---|
| **Support d'exécution** | GPU, hors ligne | Cortex-M4 180 MHz, 256 Ko, rétropropagation **sur la carte** | C'est l'objet du stage : la méthode n'avait pas été mesurée sous cette contrainte |
| **Taille du réseau** | Réseaux profonds (DQN) | Tête `k → 32 → 16 → 2` | EWC stocke **trois copies** des poids (θ, F, θ*) : la RAM triple, le réseau doit rétrécir d'autant |
| **Nombre de pénalités** | **Une par tâche** — le coût mémoire croît avec le nombre de tâches | **EWC online** : une seule Fisher accumulée, un seul θ* (`Schwarz2018`) | Coût mémoire **constant**. Sur MCU, une structure qui grandit à chaque tâche est disqualifiante (Q-07) |
| **Régime d'apprentissage** | Par lots, plusieurs époques par tâche | **Un échantillon à la fois, une seule passe** | Aucun jeu de données ne tient en RAM ; le flux ne repasse pas |
| **Frontières de tâche** | Données (MNIST) ; inférées (Atari) | **Gate de nouveauté Mahalanobis + détecteur de dérive embarqué** | En production, personne ne signale au capteur qu'une tâche commence. C'est la perspective n° 1 de la slide 15 |
| **Estimation de la Fisher *sur la carte*** | grad² sur un échantillon de données | **Proxy `w²` en moyenne mobile** (`ewc_head.c::ewc_consolidate`) | Voir ci-dessous — c'est la déviation à assumer |
| **Données** | MNIST permuté, Atari | Séries temporelles industrielles réelles | **C'est le Gap 1** : la méthode n'avait jamais été validée sur ce type de signal |
| **Précision** | Flottante | FP32, puis INT8, Q15, sub-INT8 | **C'est le Gap 3** |
| **Tâche** | Multi-classe / renforcement | Détection binaire de panne | Cas d'usage maintenance prédictive |

## La déviation qu'il faut annoncer soi-même

Sur la carte, `ewc_consolidate` **n'estime pas la Fisher par le carré du gradient** : il utilise le
carré du poids, `w²`, en moyenne mobile exponentielle. C'est un proxy grossier — *un gros poids est
supposé important* — et il n'a pas la justification théorique de l'estimateur de l'article. La
raison est matérielle : consolider par grad² demanderait une **seconde passe** sur des échantillons
qu'il faudrait avoir gardés, donc un tampon que le budget RAM ne permet pas.

> **⚠ Le piège à désamorcer, et l'argument qui le désamorce.** Un jury peut tenter : *« votre EWC
> oublie parce que votre Fisher est un proxy. »* **Non** — et c'est vérifiable dans le dépôt : le
> chiffre d'oubli que j'annonce (0,858 sans EWC contre 0,848 avec) vient de `exp_S54_forgetting`,
> une campagne **PC**, sur `ewc_multiclass`, qui utilise la **Fisher exacte par grad²**. Le proxy
> `w²` n'est pas dans la boucle de cette mesure. Le résultat négatif tient au **régime de tâches**
> — des classes inédites à chaque tâche — pas à la qualité de l'estimateur.

C'est le bon ordre de réponse : d'abord reconnaître la déviation, ensuite montrer qu'elle
n'explique pas le résultat qu'on cherche à m'opposer.

## La synthèse en une phrase

*« Je n'ai pas modifié EWC : j'ai gardé sa pénalité et son approximation diagonale telles quelles.
J'ai changé tout ce qui l'entoure — la taille du réseau, le nombre de pénalités, le régime
d'apprentissage, la source des frontières de tâche, la précision et les données — parce que chacun
de ces choix est ce que la contrainte embarquée impose. Ma contribution n'est pas la méthode, c'est
la mesure de ce qu'elle coûte quand on la porte réellement. »*

---

# Table de chronométrage

À remplir en répétition. **Chronométrer bloc par bloc**, pas seulement le total : c'est le seul
moyen de savoir *où* le débit décroche.

| Bloc | Slides | Visé | Mots | Répétition 1 | Répétition 2 | Répétition 3 |
|---|---|---|---|---|---|---|
| 1 — Contexte et EWC | 1–3 | 2:50 | 497 | | | |
| 2 — Positionnement et méthode | 4–6 | 3:20 | 592 | | | |
| 3 — Gap 1 | 7–9 | 2:40 | 438 | | | |
| 4 — Gap 2 | 10–12 | 2:25 | 425 | | | |
| 5 — Gap 3 | 13–14 | 2:25 | 400 | | | |
| 6 — Clôture | 15 | 1:20 | 278 | | | |
| **Total** | | **15:00** | **2 630** | | | |

Si un bloc dépasse, revenir à la **liste de coupes ordonnée** du budget de mots, en tête de
document — et ne pas improviser une autre coupe en salle de répétition.

**Le contrôle propre au format court** : à la fin d'une répétition, se demander *« un auditeur
pourrait-il dire que j'ai suivi un seul modèle ? »* EWC doit être nommé slides 3, 4, 8, 11, 13
et 14. Si le fil ne s'entend pas, l'exposé redevient un catalogue — et c'est exactement ce que
quinze minutes ne permettent pas.

---

# Les six phrases à savoir mot à mot

Révision J-1. Le reste se révise en compréhension ; celles-ci se récitent.

1. **L'accroche (slide 1)** — « Je vais vous parler de modèles qui continuent d'apprendre après leur
   déploiement, sur un microcontrôleur à 256 Ko de mémoire — et surtout de ce qu'il en coûte,
   mesuré. »
2. **L'écart avec Kirkpatrick (slide 3)** — « La régularisation suffit dans le régime de l'article ;
   sur nos données industrielles, elle ne suffit pas — c'est un résultat du travail, pas un échec de
   mise en œuvre. »
3. **L'annonce du fil (slide 4)** — « Pour que quinze minutes suffisent, je traite les trois gaps sur
   un seul modèle — EWC — que je suis jusqu'au bout : ce qu'il oublie, ce qu'il coûte, ce qu'il
   devient quantifié. »
4. **La règle de lecture de la parité (slides 6 et 12)** — « Un écart en régime gelé signale un bug
   de portage ; un écart en régime en ligne, borné et concentré sur les frontières de décision, est
   attendu. »
5. **L'auto-annonce de la limite (slide 9)** — « Les deux colonnes ne portent pas le même effectif
   ni le même protocole […]. Ce que ce tableau établit, c'est le classement des modèles, pas une
   comparaison au centième entre plateformes. » → désamorce Q-01 et Q-02.
6. **La chute (slide 15)** — « Ce qui se réutilise est une chaîne de portage à parité garantie par
   construction, et un protocole de mesure qui distingue systématiquement ce qui est mesuré de ce
   qui est estimé. »

Deux autres se révisent aussi, mais pour la **phase de questions** : leurs slides ont quitté le fil.
« Publier uniquement le premier chiffre de RAM serait flatteur et malhonnête — c'est ce que je
reproche à la littérature, donc je m'y astreins » (slide 10, désamorce Q-25) ; et « Je n'ai pas de
modèle universel — sur CMAPSS, EWC plafonne à 0,456 » (slide de secours **B13**).

---

## Renvois

| Besoin | Document |
|---|---|
| Les 60 questions du jury et leurs réponses | [`S01_questions_jury.md`](S01_questions_jury.md) |
| Le plan du deck : figures, messages, minutage | [`S02_plan_presentation.md`](S02_plan_presentation.md) |
| Les 15 slides de secours et la table de routage question → slide | [`S03_slides_backup.md`](S03_slides_backup.md) |
| Les références et les crédits d'images à annoncer | [`S04_bibliographie.md`](S04_bibliographie.md) |
| Ce qui est repris de l'article EWC et ce qui en diverge | [Annexe EWC](#annexe--ewc--larticle-de-2017-et-ce-que-jen-ai-fait) (ce document) |
