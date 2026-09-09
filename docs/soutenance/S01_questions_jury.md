# S01 — Questions potentielles du jury et réponses préparées

> **Hypothèse de travail** : le jury n'a lu que le manuscrit. Il n'a ni le dépôt, ni les
> JSON d'expériences, ni les fiches de sprint, ni les notebooks. Toute question naîtra
> d'une **phrase, d'un chiffre ou d'une figure du manuscrit**, et toute réponse doit tenir
> **sans renvoyer au code**.

## Ce que le jury a effectivement sous les yeux

| Élément | Quantité | Détail |
|---|---|---|
| Chapitres | 8 + annexes | Intro, État de l'art, Problématique, Méthodo, Gap 1, Gap 2, Gap 3, Conclusion |
| Figures | **15** | 4 au ch. 5, 3 au ch. 6, 3 au ch. 7, **7 en annexe** |
| Tableaux | **5** | 4.1 jeux · 4.2 modèles · 5.1 grille 4×3 · A.1 jeux détail · A.2 grille 4×5 |
| Pages | 58 | |

**Point structurel à avoir en tête** : les chapitres 1 à 4 ne contiennent **aucune figure**.
Le jury n'a donc jamais vu de schéma de la chaîne de portage, de l'architecture firmware ni
du protocole UART. Toute question sur « comment ça marche concrètement » part de zéro —
d'où les slides de secours prévues en `S03`.

**Absent du manuscrit** (donc à ne pas amener soi-même, mais à garder en réserve si l'on est
poussé) : l'étude sub-INT8 (ternaire/binaire), les détecteurs de dérive S44/S45, les paires
de modèles et le méta-modèle, les mesures de courant du LPM01A, l'article standalone.

---

# Section 1 — Les trois questions critiques

Ces trois-là peuvent faire basculer la soutenance. Réponses à connaître **mot à mot**.

---

### Q-01 — « Vous annoncez une parité PC ↔ carte exacte de 1,000. Mais dans le tableau 5.1, EWC obtient 0,893 sur PC et 0,947 sur carte pour Monitoring. Comment la carte peut-elle faire mieux que le PC si elle calcule exactement la même chose ? »

**D'où elle vient** : tableau 5.1 croisé avec §6.3 (« la parité vaut exactement 1,000 »),
et surtout la phrase du §5.5 : *« le passage sur carte ne dégrade pas systématiquement les
chiffres — il les améliore même pour EWC sur Monitoring et Pronostia »*.

**Ce que le jury teste** : la solidité — c'est une contradiction apparente entre deux
chapitres. C'est *la* question qu'un rapporteur attentif posera.

**Réponse (à dire posément, sans défensive)** :

> « La question est juste, et la réponse est que ces deux chiffres ne mesurent pas la même
> chose. La parité de 1,000 est une mesure de **fidélité de portage** : mêmes poids, mêmes
> échantillons, même ordre, et on compare prédiction par prédiction. Elle répond à la
> question "la carte calcule-t-elle la même chose que le PC ?" — la réponse est oui, sans
> un seul désaccord.
>
> Les deux colonnes du tableau 5.1, elles, ne portent pas sur la même **population
> d'évaluation**. La colonne PC est l'évaluation d'apprentissage continu complète, sur
> l'intégralité du jeu de test après toutes les tâches. La colonne carte est un flux rejoué
> en régime gelé, sur un échantillon de ce jeu. L'écart entre 0,893 et 0,947 est donc un
> écart de population, pas un effet de la carte. Si l'on compare les deux plateformes sur
> les *mêmes* échantillons — ce que fait le chapitre 6 —, l'écart de performance finale est
> de 0,007 au maximum.
>
> Avec le recul, la formulation "il les améliore même" sur-interprète le tableau : elle
> devrait dire "le portage ne dégrade pas", ce qui est ce que la parité établit réellement.
> Le tableau aurait dû porter les effectifs de chaque colonne. »

**Si on insiste — « alors le tableau 5.1 ne démontre rien ? »** :

> « Il démontre le classement des modèles, qui est robuste : EWC au-dessus, HDC à zéro sur
> carte, Mahalanobis intermédiaire. Ce qu'il ne permet pas, c'est de lire les décimales
> comme une comparaison PC contre carte. La comparaison appariée rigoureuse, c'est le
> chapitre 6 — et c'est bien là que je place l'argument scientifique du portage. »

**Piège à éviter** : inventer une explication technique (« la carte arrondit
différemment », « float32 vs float64 »). Ce serait faux — en régime gelé la parité est
exacte, donc l'arithmétique n'explique rien. La seule explication vraie est la population
d'évaluation.

---

### Q-02 — « Sur combien d'échantillons vos chiffres de carte sont-ils calculés ? Quelle est votre barre d'erreur ? »

**D'où elle vient** : tableaux 5.1 et A.2, qui donnent trois décimales sans jamais donner
d'effectif ni d'intervalle de confiance. Aucun chiffre du manuscrit n'est assorti d'une
incertitude.

**Ce que le jury teste** : la rigueur statistique. C'est la faiblesse méthodologique la
plus visible du manuscrit.

**Réponse** :

> « Les cellules de la grille sont mesurées sur un flux de l'ordre de la centaine
> d'échantillons par cellule ; la comparaison appariée du chapitre 6 porte, elle, sur
> 7 534 à 7 672 échantillons. Sur une centaine d'échantillons avec une classe fautif
> minoritaire, l'intervalle de confiance sur un F1 est de l'ordre de ±0,05 à ±0,10. Un
> écart de 0,05 entre deux cellules n'est donc pas résoluble, et je ne le défends pas comme
> tel.
>
> Ce qui est résoluble et qui porte mes conclusions, ce sont les écarts d'un ordre de
> grandeur : EWC à 0,95 contre HDC à 0,00, ou une accuracy de 0,87 pour un F1 nul. Aucune
> de mes trois conclusions ne repose sur une troisième décimale.
>
> C'est une limite assumée : le protocole n'inclut ni répétitions multi-graines ni
> intervalles de confiance sur la grille complète. Chaque campagne de mesure sur carte
> demande un cycle entraînement–export–compilation–flash–streaming par cellule, ce qui a
> borné le nombre de répétitions dans le temps du stage. C'est la première chose que je
> reprendrais. »

**Piège à éviter** : prétendre que les chiffres sont significatifs, ou improviser une
valeur d'intervalle de confiance précise. Donner l'ordre de grandeur et assumer.

---

### Q-03 — « Votre problématique, page 3, annonce "un système de détection d'anomalies non supervisé". Vos trois modèles focus sont supervisés. Quelle est finalement votre contribution ? »

**D'où elle vient** : §1.4, dernier paragraphe, qui formule la problématique centrale en
termes non supervisés — alors que §1.3, §3.2 et tout le reste du manuscrit décrivent un
dispositif mixte à dominante supervisée.

**Ce que le jury teste** : la cohérence du cadrage. C'est une incohérence interne réelle,
repérable à la lecture.

**Réponse** :

> « Vous avez raison de relever la tension, et le manuscrit la documente au §3.2, "Écarts
> assumés vis-à-vis du rapport intermédiaire". Le cadre de départ était strictement non
> supervisé ; il a évolué vers un dispositif mixte pour deux raisons. D'une part, les jeux
> de données publics de maintenance prédictive fournissent des labels exploitables, qui
> rendent l'évaluation supervisée à la fois possible et informative — sans labels, je ne
> pouvais mesurer ni F1, ni oubli, ni transfert arrière, c'est-à-dire aucune des métriques
> d'apprentissage continu qui font l'objet du Gap 1. D'autre part, l'état de l'art embarqué
> auquel je me compare — TinyOL, QLR-CL, LifeLearner — est supervisé : rester non supervisé
> m'aurait privé de tout point de comparaison.
>
> Le non supervisé n'a pas disparu pour autant : Mahalanobis est conservé comme baseline et
> mesuré sur toute la grille, et le prototype de mise à jour autonome du chapitre 8 est
> précisément le retour au non supervisé — c'est un détecteur non supervisé qui décide à
> bord quand mettre à jour.
>
> La phrase du §1.4 n'a pas été réalignée sur le cadre final ; c'est un défaut de rédaction,
> et le cadre effectivement réalisé est celui du §1.3. »

**Piège à éviter** : nier l'incohérence, ou au contraire s'excuser longuement. Une phrase
de reconnaissance, puis la justification de fond, puis le pont vers les perspectives.

---

# Section 2 — Questions de compréhension

Le jury cherche à comprendre, pas à piéger. Réponses courtes, pédagogiques, sans jargon
inutile.

---

### Q-04 — « Expliquez-nous EWC simplement. »

**Réponse** : « Quand un réseau apprend une nouvelle tâche, la descente de gradient est
libre de modifier n'importe quel poids — y compris ceux qui portaient la tâche précédente.
EWC ajoute à la fonction de coût une pénalité quadratique qui tire chaque poids vers sa
valeur d'après la tâche précédente, avec une raideur proportionnelle à l'importance de ce
poids pour cette tâche. L'importance est estimée par la diagonale de la matrice
d'information de Fisher, c'est-à-dire, en pratique, la moyenne du carré du gradient. Les
poids importants deviennent rigides, les autres restent libres : c'est un compromis
stabilité–plasticité explicite. »

### Q-05 — « Pourquoi la diagonale de Fisher et pas la matrice complète ? »

**Réponse** : « La matrice complète est en O(n²) en nombre de paramètres. Pour ma tête à
705 paramètres, cela ferait un demi-million de coefficients, soit environ 2 Mo en float32 —
vingt fois la RAM totale que consomme mon firmware. La diagonale est en O(n), soit quelques
kilo-octets, et c'est l'approximation retenue par Kirkpatrick lui-même. La contrainte
embarquée rend ce choix non négociable. »

### Q-06 — « Comment choisissez-vous λ, le poids de la pénalité ? »

**Réponse** : « Par configuration, avec la valeur par défaut de la littérature, et fixée
par fichier YAML pour chaque couple modèle–jeu de données, de sorte que l'expérience soit
rejouable. Le manuscrit ne présente pas d'ablation sur λ : ce n'est pas l'objet du travail,
qui porte sur le portage et la mesure, pas sur l'optimisation de la méthode. C'est une
limite, et λ est un des premiers leviers si l'on voulait améliorer les résultats de la
grille — notamment sur CMAPSS. »

### Q-07 — « Que veut dire "EWC online" par rapport à EWC standard ? »

**Réponse** : « EWC standard accumule une pénalité et un jeu de paramètres de référence par
tâche : le coût mémoire croît avec le nombre de tâches. La variante en ligne maintient une
seule estimation de Fisher, mise à jour en continu, et un seul jeu de paramètres de
référence. Le coût mémoire devient constant, indépendant du nombre de tâches — c'est
exactement ce qu'il faut sur un microcontrôleur, où l'on ne peut pas laisser une structure
croître indéfiniment. »

### Q-08 — « Pourquoi cette carte ? Une NUCLEO-F439ZI, ce n'est pas ce qu'il y a de plus contraint. »

**Réponse** : « C'est un Cortex-M4 à 180 MHz avec 256 Ko de SRAM et sans accélérateur
neuronal — il tombe dans la définition même du TinyML donnée par Lin et al. : pas de DRAM,
pas de système d'exploitation, moins de 256 Ko de SRAM. La cible initiale du stage était une
STM32N6, avec un Cortex-M55 et un NPU, mais elle n'était pas disponible ; j'ai donc travaillé
sur une carte réellement en main plutôt que de simuler. Et ce choix s'est révélé
scientifiquement fécond, parce que la présence d'une FPU matérielle sur ce cœur est
exactement ce qui produit le résultat contre-intuitif du chapitre 7 : sur cette cible,
l'INT8 est plus lent que le flottant. »

### Q-09 — « Qu'est-ce que le compteur DWT, et pourquoi devrait-on lui faire confiance ? »

**Réponse** : « C'est un compteur de cycles intégré au cœur Cortex-M, incrémenté à chaque
cycle d'horloge. On le lit avant et après la section à mesurer et on convertit en
microsecondes par la fréquence d'horloge, ici 180 MHz. C'est la mesure la plus directe
possible : elle ne passe ni par un système d'exploitation, ni par un timer périphérique, ni
par une instrumentation logicielle qui s'ajouterait au temps mesuré. La résolution est le
cycle, soit 5,6 nanosecondes. »

### Q-10 — « Que sont `.bss` et `.data`, et pourquoi les distinguer ? »

**Réponse** : « Ce sont les deux sections de RAM que l'éditeur de liens réserve
statiquement. `.data` contient les variables globales initialisées à une valeur non nulle,
`.bss` celles qui démarrent à zéro. Je les mesure par différence de symboles du linker —
`_ebss` moins `_sbss` — donc sur le binaire réellement flashé, pas sur une estimation. Sur
mon build de référence, `.bss` vaut 105 036 octets et `.data` 460. Je les distingue parce
que ce sont deux allocations différentes, et parce que la somme des deux ne suffit pas :
il manque la pile. »

### Q-11 — « Le "stack painting", qu'est-ce que c'est, et pourquoi ne pas simplement calculer la pile ? »

**Réponse** : « On remplit au démarrage toute la zone de pile libre avec un motif connu.
Après exécution, on relit la zone : là où le motif a disparu, la pile est passée. La
frontière donne le point le plus profond réellement atteint. C'est une mesure, pas une
borne : un calcul statique de profondeur donnerait un pire cas théorique, souvent très
pessimiste, alors que le *watermark* donne ce qui s'est effectivement produit sur les
données réelles. C'est important ici parce que c'est ce qui m'a permis de montrer que la
mise à jour continue creuse la pile davantage que l'inférence seule — 4 688 octets contre
4 416 sur Monitoring. »

### Q-12 — « Pourquoi mesurer la pile ? `.bss` ne suffisait-il pas ? »

**Réponse** : « Non, et c'est une correction de méthode que le manuscrit assume
explicitement. `.bss` ne compte que les variables globales ; les gros tampons locaux vivent
sur la pile et lui échappent complètement. Mes mesures antérieures, qui ne remontaient que
`.bss`, étaient donc optimistes. La formule retenue est RAM totale égale `.data` plus `.bss`
plus le pic de pile. En pratique l'écart est modeste — environ 4,5 Ko — ce qui valide
rétrospectivement les conclusions antérieures, mais il fallait le mesurer plutôt que de le
supposer. »

### Q-13 — « Quelle est la différence entre domain-incremental et class-incremental sur vos jeux ? »

**Réponse** : « En domain-incremental, les classes restent les mêmes et c'est la
distribution des entrées qui change : sur Monitoring, je réapprends la même distinction
normal/fautif successivement sur une pompe, une turbine, puis un compresseur. En
class-incremental, de nouvelles classes apparaissent au fil des tâches : sur Pronostia, la
condition de faute n'existe qu'après une phase de fonctionnement normal. Le second est
nettement plus dur, parce que le modèle doit ajouter une classe sans avoir revu les
précédentes — c'est là que l'oubli se manifeste le plus violemment, et c'est là que l'écart
entre EWC et les baselines se creuse. »

### Q-14 — « Comment découpez-vous un jeu de données en tâches ? Ce découpage n'est-il pas arbitraire ? »

**Réponse** : « Il est fixé par une configuration YAML par jeu de données, et il suit une
variable métier, pas une coupure arbitraire : le type d'équipement pour Monitoring, le
sous-jeu FD001 à FD004 pour CMAPSS, la condition pour Pronostia, le degré de dommage pour
Paderborn. Le découpage est donc reproductible d'une exécution à l'autre et justifiable
physiquement. Il est vrai qu'un autre découpage donnerait d'autres chiffres — c'est une
propriété générale de l'évaluation en apprentissage continu, pas un choix de ma part. »

### Q-15 — « Pourquoi un CRC sur les trames UART ? »

**Réponse** : « Parce que sans lui je ne pourrais pas garantir que la prédiction que je
compare est bien celle qu'a produite la carte. Une trame corrompue passerait silencieusement
et polluerait la mesure de parité. Le taux d'erreur CRC est visé à zéro et rapporté à zéro
sur toutes les campagnes : c'est une condition de validité de tout le chapitre 6, pas un
détail d'ingénierie. »

### Q-16 — « QAT, PTQ, "fake quant" — pouvez-vous préciser ? »

**Réponse** : « La quantification post-entraînement, PTQ, prend un modèle déjà entraîné en
flottant et convertit ses poids en 8 bits : c'est simple, mais si l'échelle de conversion
est mal choisie par rapport à la dynamique réelle des poids, on perd de l'information. La
quantification consciente de l'entraînement, QAT, insère pendant l'apprentissage une
opération de *fake quant* : les valeurs sont arrondies sur la grille 8 bits mais restent
représentées en flottant, de sorte que les gradients continuent de circuler et que le modèle
apprend sous la contrainte de précision réduite. Dans mon travail, le QAT préserve la
métrique — l'écart avec le FP32 est inférieur à 0,006 — tandis que la PTQ naïve embarquée
s'effondrait, pour une raison d'échelle mal calibrée. »

### Q-17 — « Pourquoi une architecture k → 32 → 16 → 2 ? »

**Réponse** : « C'est un perceptron à deux couches cachées, dimensionné pour tenir
confortablement dans l'enveloppe mémoire tout en restant capable d'apprendre les frontières
de décision de ces jeux tabulaires — environ 700 paramètres pour k = 4. Toutes les
dimensions viennent d'une constante de configuration, jamais d'une valeur écrite dans le
code, ce qui permet de faire varier k et de mesurer l'effet. Je n'ai pas fait de recherche
d'architecture : ce n'est pas l'objet du travail, et une architecture fixe est nécessaire
pour que les comparaisons PC ↔ carte restent appariées. »

### Q-18 — « "Parité bit à bit" : au sens strict ? »

**Réponse** : « Au sens des décisions : je compare les prédictions une à une, et en régime
gelé il n'y a aucun désaccord sur 7 534 à 7 672 échantillons selon la cellule. Sur les
scores continus, l'écart maximal mesuré est de l'ordre de 10⁻⁶, ce qui est le bruit d'arrondi
du float32. Le terme est employé au sens fort parce que rien ne diffère au niveau de la
décision — mais la formulation exacte est "aucune divergence de prédiction", et c'est ce que
je défends. »

### Q-19 — « AF et BWT : quelles définitions exactes utilisez-vous ? »

**Réponse** : « L'oubli moyen est la moyenne, sur les tâches, de la chute entre la
performance maximale atteinte sur une tâche et sa performance finale. Le transfert arrière
est la différence moyenne entre la performance finale sur les tâches passées et leur
performance juste après apprentissage : négatif, il signale de l'oubli ; positif, il signale
que les tâches suivantes ont aidé les précédentes. Ce sont les définitions standard de la
littérature d'apprentissage continu. Sur mon cas EWC multiclasse, l'oubli moyen en F1 vaut
0,847 — c'est-à-dire un effondrement quasi complet. »

### Q-20 — « Pourquoi le F1 de la classe fautif plutôt que le F1 macro ou l'accuracy ? »

**Réponse** : « Parce que la seule chose qui compte industriellement, c'est de détecter la
panne. L'accuracy est portée par la classe majoritaire, qui est le fonctionnement normal :
un modèle qui ne détecte rien affiche une accuracy honorable. Le F1 macro moyenne les deux
classes et dilue partiellement le problème. Le F1 de la classe fautif est le seul qui
s'effondre à zéro quand le modèle cesse de détecter — c'est ce qui m'a permis de voir que HDC
embarqué prédisait systématiquement la classe majoritaire, avec 0,867 d'accuracy sur
Monitoring et un F1 nul. Sans cette métrique, ce tableau se lisait comme un succès. »

### Q-21 — « Que fait HDC concrètement ? »

**Réponse** : « L'encodage hyperdimensionnel projette chaque échantillon dans un espace de
très grande dimension — de l'ordre du millier — au moyen de vecteurs de base tirés
aléatoirement. Dans cet espace, l'apprentissage se réduit à additionner les hypervecteurs
d'une même classe pour former un prototype, et la prédiction à mesurer une similarité. Il n'y
a ni gradient ni rétropropagation, ce qui en fait un candidat naturel pour l'embarqué. Dans
mes mesures, c'est aussi le modèle le plus lent — l'encodage domine le coût — et le seul dont
le F1 embarqué s'effondre à zéro. »

### Q-22 — « Qu'est-ce que votre "émulateur bit-exact" et pourquoi vous croire ? »

**Réponse** : « C'est une réimplémentation en Python du noyau de calcul entier du firmware,
opération par opération, y compris les arrondis et les saturations. Son intérêt est de
permettre de tester une hypothèse — par exemple "et si l'accumulateur était en 32 bits ?" —
sans recompiler ni reflasher, ce qui rend l'ablation praticable. Sa validité ne repose pas
sur ma parole : elle est vérifiée contre la carte, et la parité entre l'émulateur et la carte
est exacte. C'est ce qui autorise à lire la figure d'ablation comme une mesure et non comme
une simulation. »

---

# Section 3 — Points attaquables

C'est ici que se joue la crédibilité. **Règle générale : concéder vite et précisément, puis
expliquer ce que le résultat vaut malgré tout.** Un point faible reconnu et cadré vaut mieux
qu'un point faible défendu.

---

### Q-23 — « La ligne TinyOL de votre tableau 5.1 compare deux architectures différentes. Pourquoi l'avoir laissée dans le tableau ? »

**Réponse** : « C'est une réserve que je porte moi-même sous les deux tableaux, parce que
laisser la ligne sans avertissement aurait été trompeur. Je l'ai gardée pour deux raisons.
La colonne carte est une mesure valide en elle-même : elle est vérifiée en parité exacte
contre sa propre référence PC, et elle décrit ce que le réseau réellement embarqué produit.
Et la retirer aurait laissé un trou dans une grille dont l'objet est justement de couvrir
les quatre familles. Ce que la ligne ne permet pas, c'est de conclure sur l'effet du
portage — et c'est écrit sous le tableau. Rétrospectivement, la présentation la plus honnête
aurait été de séparer visuellement cette ligne, par exemple par un trait, plutôt que de la
neutraliser par une note de bas de tableau. »

### Q-24 — « HDC donne un F1 nul sur les cinq jeux une fois embarqué. Ce n'est pas un bug ? »

**Réponse** : « Non, et c'est ce que la vérification de parité permet d'affirmer. Le modèle
ne plante pas et ne produit pas de valeurs aberrantes : il prédit systématiquement la classe
majoritaire, avec une accuracy correcte — 0,867 sur Monitoring, 0,900 sur Pronostia. C'est
un comportement cohérent, pas un dysfonctionnement.
>
La cause est identifiée : contrairement à EWC et Mahalanobis, dont les poids sont exportés
depuis le PC, la version embarquée de HDC construit sa projection à bord, avec une dimension
et une initialisation qui lui sont propres. Ce n'est donc pas le même modèle que celui du
PC, et la comparaison directe de la ligne HDC souffre de la même réserve que TinyOL — sauf
que sur HDC, cette réserve n'est pas explicitée dans le manuscrit, et elle aurait dû l'être.
>
Ce que je maintiens, c'est ce que la ligne démontre effectivement : un modèle embarqué peut
afficher 0,87 d'accuracy en ne détectant aucune panne. C'est le second contre-exemple,
indépendant de celui de Mahalanobis, qui justifie de raisonner en F1. »

**Piège à éviter** : présenter HDC comme un échec de la méthode HDC en général. Ce serait
injuste envers la littérature et faux : le PC atteint 0,955 sur CWRU.

### Q-25 — « Vous donnez trois chiffres de RAM : 1 000 octets, 105 036 octets, 183 936 octets. Lequel est votre résultat ? »

**Réponse** : « 105 300 octets, RAM totale — c'est le chiffre du système réellement déployé,
avec les quatre familles de modèles co-résidentes, pile comprise. Les deux autres bornent
l'interprétation, et le manuscrit les sépare explicitement pour cette raison. Le millier
d'octets est la tête EWC seule : il établit la faisabilité — l'apprentissage continu embarqué
*peut* tenir très en deçà de 100 Ko — mais ne décrit qu'un modèle isolé, et je me garde bien
de le présenter comme le chiffre du travail. Les 183 936 octets sont le pire cas, en
condition toutes variables sur CMAPSS avec 21 entrées : ils montrent que même la
configuration la plus défavorable reste dans l'enveloppe, à 70 %.
>
J'ai fait le choix de publier les trois plutôt qu'un seul, précisément parce que ne publier
que le plus flatteur est ce que je reproche à la littérature. »

### Q-26 — « Vous vous comparez à LifeLearner, 212 Ko sur Cortex-M7, contre vos 105 Ko. Mais leur tâche est bien plus complexe. La comparaison est-elle recevable ? »

**Réponse** : « Elle ne l'est pas comme comparaison de performance, et je ne la présente pas
ainsi. Ce que je retiens de LifeLearner, c'est un point de référence sur l'**enveloppe** :
c'est le travail du corpus qui se rapproche le plus des contraintes réelles d'un
microcontrôleur, et il frôle la limite des 256 Ko. Mon apport par rapport à lui n'est pas
"je consomme deux fois moins" — ce serait une comparaison de pommes et d'oranges — mais "je
mesure ce qu'ils ne mesurent pas" : la décomposition composant par composant, le coût
mémoire de la mise à jour en ligne, et le surcoût de latence de l'adaptation. LifeLearner
est évalué sur une tâche supervisée sans mesure du surcoût d'adaptation. C'est cette
absence-là qui définit mon Gap 2, pas le nombre d'octets. »

### Q-27 — « Votre budget de latence est de 100 ms et votre pire cas est de 2,1 ms. N'est-ce pas un critère trivialement satisfait ? »

**Réponse** : « Sur cette classe de capteurs, oui — et c'est en soi un résultat, parce que
ce n'était pas acquis pour de l'apprentissage *en ligne*, où la mise à jour continue s'ajoute
à l'inférence à chaque échantillon. Ce qui est intéressant n'est pas la marge, c'est ce que
la mesure révèle : l'adaptation en ligne multiplie la latence par cinq, de 48–65 µs à
239–340 µs. C'est ce facteur que la littérature ne publie pas, et c'est lui qui dimensionnera
un système à cadence plus élevée — en vibration à quelques kilohertz, la marge se réduit
nettement. Le budget de 100 ms est un critère de recevabilité industriel, pas une difficulté
scientifique ; la difficulté est de mesurer proprement le surcoût, et c'est ce que je fais. »

### Q-28 — « Chapitre 7 : l'INT8 est plus lent que le FP32, et vous montrez vous-même que la RAM du système est inchangée. Que reste-t-il du Gap 3 ? »

**Réponse** : « Il reste trois choses, et je les distingue.
>
D'abord un résultat positif net : la quantification consciente de l'entraînement préserve la
métrique, à moins de 0,006 d'écart. C'est la réponse directe à la lacune du corpus, où
l'entraînement reste systématiquement en flottant.
>
Ensuite un résultat de diagnostic : l'effondrement de la PTQ embarquée, de 0,92 à moins de
0,15 de F1, était imputable à un seul facteur — une échelle de quantification fixe, non
calibrée sur la dynamique réelle des poids — et non au principe de la quantification.
L'ablation le montre poste par poste, et le noyau calibré récupère intégralement la métrique
sur la cible matérielle.
>
Enfin deux résultats négatifs, que je publie comme tels : sur un cœur doté d'une FPU, l'INT8
n'accélère rien — il coûte 74 µs contre 48 à 50 —, et le gain mémoire de ÷4 porte sur les
poids du modèle, pas sur la RAM système, dominée par les tampons du pipeline multi-modèle.
>
Ma conclusion est donc que sur cette cible, l'INT8 est un gain de mémoire de modèle et rien
d'autre. C'est moins vendeur qu'un gain global, mais c'est ce que la mesure dit. »

### Q-29 — « Vous revendiquez la "rétropropagation quantifiée pendant l'adaptation" comme la lacune du corpus. Votre rétropropagation tourne-t-elle en quantifié sur la carte ? »

**Ce que le jury teste** : c'est la question la plus dangereuse du chapitre 7, parce qu'elle
touche à la revendication elle-même.

**Réponse** :

> « Non, et il faut le dire clairement. Sur la carte, l'inférence peut tourner en INT8, mais
> la mise à jour continue s'exécute en flottant. La quantification pendant l'entraînement,
> je la démontre côté PC, avec du *fake quant* dans la boucle d'apprentissage incrémental.
>
> Donc le Gap 3 est **partiellement** comblé, et la conclusion du manuscrit le formule ainsi
> — "le bilan est nuancé et assumé". Ce que j'établis, c'est qu'apprendre en continu sous
> contrainte de quantification ne coûte quasiment rien en qualité, ce qui lève le doute de
> principe. Ce que je n'établis pas, c'est une rétropropagation entière exécutée sur le
> microcontrôleur. Et le chapitre 7 donne d'ailleurs la raison pour laquelle ce ne serait pas
> rentable sur cette cible : les étages de conversion entier–flottant coûtent plus cher que
> ce que le calcul entier fait gagner. Sur un Cortex-M0+ sans FPU, la conclusion
> s'inverserait — et c'est la première expérience que je ferais pour aller au bout du Gap 3. »

### Q-30 — « Un travail sur l'embarqué contraint sans aucune mesure d'énergie ? »

**Réponse** : « C'est la dimension qui manque, et je la place explicitement en perspective
plutôt que de l'estimer. La chaîne d'instrumentation est en place — marqueurs matériels dans
le firmware pour délimiter les phases, routine de segmentation et d'intégration des mesures
d'un profileur de puissance, calcul d'autonomie à partir de la capacité batterie. Ce qui a
manqué, c'est la disponibilité de la sonde de mesure sur la durée du stage.
>
J'ai fait le choix de ne publier aucun chiffre d'énergie plutôt que d'en dériver un à partir
de la latence et d'un courant nominal de fiche technique. Une estimation de ce genre aurait
été présentable, et elle aurait été fausse : elle aurait notamment raté le fait que le gain
mémoire de l'INT8 ne se traduit pas en gain de temps, donc a priori pas non plus en gain
d'énergie. Sur ce point précis, la question reste ouverte, et je la donne comme ouverte. »

### Q-31 — « Pas d'intervalles de confiance, pas de répétitions. Vos conclusions sont-elles robustes ? »

Voir Q-02. Compléter par : « Les conclusions que je défends sont ordinales — le classement
des modèles, l'existence de l'oubli, la nullité du F1 de HDC, le signe du surcoût de latence
INT8 — et aucune ne repose sur une différence de quelques centièmes. »

### Q-32 — « Une seule carte, une seule architecture, un seul λ. Quelle est la validité externe ? »

**Réponse** : « Elle est limitée, et je la borne : mes conclusions valent pour un Cortex-M4
avec FPU, sur des jeux tabulaires ou de vibration agrégés en variables, avec une tête de
quelques centaines de paramètres. Ce qui se généralise n'est pas le chiffre, c'est la
**méthode** : la chaîne de portage à parité garantie par construction — poids générés,
protocole figé, source unique de sélection des variables — et le protocole de mesure qui
sépare systématiquement ce qui est mesuré de ce qui est estimé. C'est ce que je revendique
comme contribution réutilisable, et c'est ce que dit ma conclusion. »

### Q-33 — « CMAPSS est un problème de RUL, donc de régression. Vous en faites de la classification binaire. D'où sort votre seuil ? »

**Réponse** : « Le seuil est un paramètre de configuration, et il définit à partir de quelle
durée de vie résiduelle un échantillon est étiqueté fautif. J'ai mené une étude de
sensibilité à ce seuil, sur plusieurs valeurs et sur plusieurs jeux, en vérifiant que la
proportion de positifs varie de façon monotone. Elle n'est pas dans le manuscrit, faute de
place, ce qui est un manque puisque le tableau 5.1 dépend de ce choix. Le manuscrit conserve
par ailleurs CMAPSS comme référence de régression, avec le RMSE sur le RUL. Et je note que
CMAPSS est de toute façon la colonne la plus faible de ma grille pour tous les modèles :
c'est un jeu de séries longues, où une décision par échantillon isolé, sans fenêtre
temporelle, est structurellement désavantagée. »

### Q-34 — « Pourquoi Mahalanobis comme unique baseline non supervisée ? Un auto-encodeur ou un Isolation Forest auraient été plus forts. »

**Réponse** : « J'ai implémenté et évalué plusieurs détecteurs non supervisés — k-moyennes,
DBSCAN, k plus proches voisins, ACP, et Mahalanobis. Mahalanobis est le seul que j'ai porté,
pour une raison de contrainte embarquée : il n'a besoin que d'une moyenne et d'une inverse de
covariance, pas de rétropropagation, pas de stockage d'exemplaires — son inférence coûte
3 à 5 µs sur la carte, soit dix fois moins qu'EWC. Un auto-encodeur, lui, est présent dans le
dispositif : c'est TinyOL, dont la détection repose sur l'erreur de reconstruction. Ce que je
ne prétends pas, c'est que Mahalanobis soit l'état de l'art non supervisé — le manuscrit le
qualifie de baseline volontairement légère, et ses résultats sur les jeux vibratoires, 0,127
sur CWRU et 0,071 sur Paderborn, montrent bien qu'une gaussienne unique ne capture pas ces
distributions. »

### Q-35 — « Votre système n'est pas autonome : la mise à jour est déclenchée par l'utilisateur. »

**Réponse** : « Exact, dans le dispositif principal. Le déclencheur est externe, et c'est un
choix de méthode : je voulais que la comparaison PC ↔ carte soit appariée, donc que la
séquence de mises à jour soit strictement contrôlée. Un déclencheur autonome aurait rendu
les deux plateformes non comparables.
>
Le prototype autonome existe et il est mesuré sur la carte, au chapitre 8 : un détecteur non
supervisé plus un détecteur de dérive à fenêtre glissante décident à bord quand mettre à
jour. Le taux de mise à jour tombe à 0,025 contre 1,0, soit 97 % des mises à jour
économisées, et la parité de verdict carte ↔ PC vaut 1,000 — la décision prise à bord est
exactement celle que reconstruit la référence. Ce qui n'est pas résolu, c'est le label :
avec l'étiquette vraie, le F1 est préservé ; en pseudo-étiquetage entièrement autonome sur
Pronostia, il tombe à 0,504. C'est le verrou, et je le donne comme tel. »

### Q-36 — « Paderborn, mono-classe par tâche : n'est-ce pas un scénario construit pour avantager EWC ? »

**Réponse** : « Le découpage vient de la structure du jeu — un degré de dommage par
enregistrement —, il n'a pas été conçu pour la démonstration. Mais votre soupçon est
légitime, et je le retourne : c'est justement parce que ce régime est extrême que les autres
modèles s'y effondrent, et il serait malhonnête d'en tirer un classement général. C'est
pourquoi Paderborn est en annexe et non dans les jeux focus. Ce que la colonne établit, c'est
qu'il existe un régime où seule une méthode de régularisation tient — pas qu'EWC domine
partout. Sur CMAPSS, EWC plafonne à 0,456 : je n'ai pas de modèle universel. »

### Q-37 — « Votre démonstration de l'oubli catastrophique, au §5.3, porte sur CWRU — un jeu que vous reléguez en annexe. Pourquoi l'argument central du chapitre repose-t-il sur un jeu secondaire ? »

**D'où elle vient** : le §5.3 et la légende de la figure `ch5_oubli_catastrophique`
mentionnent CWRU, alors que le §4.1 annonce que le corps du manuscrit se concentre sur D2,
D4 et D5.

**Réponse** : « Parce que c'est sur CWRU que le phénomène est le plus lisible : c'est un
problème multiclasse à trois tâches, où l'écart entre la moyenne des F1 post-tâche, 0,981, et
le F1 du modèle final, 0,240, est spectaculaire. Sur un problème binaire à deux classes
partagées entre les domaines, l'oubli existe mais il est moins démonstratif. J'ai donc choisi
le cas le plus pédagogique pour établir le phénomène, quitte à ce qu'il vienne d'un jeu
annexe. C'est une incohérence de mise en page plus que de fond, et le §5.4 recadre ensuite
l'analyse sur les deux jeux focus. »

### Q-38 — « Comment le jury peut-il vérifier vos chiffres ? »

**Réponse** : « Chaque exécution produit un dossier horodaté contenant un instantané de la
configuration et les résultats en JSON, la graine aléatoire est fixée, et les poids C sont
générés par script et jamais édités à la main — donc n'importe quelle mesure est rejouable à
l'identique. Le manuscrit porte d'ailleurs, en commentaire de source, le chemin exact du
fichier de résultats derrière chaque chiffre du texte. Le dépôt n'accompagne pas le manuscrit,
mais il existe une version d'export documentée destinée à la publication, et je peux fournir
n'importe quelle campagne sur demande. »

### Q-39 — « Vos figures de RAM et de latence viennent-elles toutes de la même campagne ? »

**Réponse** : « Non, et c'est une chose que le manuscrit aurait dû dire plus nettement. Les
mesures de RAM totale, la comparaison appariée du chapitre 6, la grille du chapitre 5 et la
décomposition de latence INT8 sont quatre campagnes distinctes, chacune avec son protocole.
Elles sont cohérentes entre elles — le `.bss` de référence, 105 036 octets, est invariant
d'une campagne à l'autre, ce qui est un contrôle —, mais les effectifs et les régimes
diffèrent, et c'est ce qui explique la question sur le tableau 5.1 que vous m'avez posée
tout à l'heure. »

### Q-40 — « Le chapitre 6 dit "Gap 2 comblé". N'est-ce pas un peu péremptoire ? »

**Réponse** : « Le mot est fort, je l'accorde. Ce que j'établis précisément, c'est un
système d'apprentissage continu multi-modèle, à parité de prédiction vérifiée, mesuré
composant par composant, dans l'enveloppe d'un Cortex-M4 — ce qu'aucun des quatre travaux du
corpus n'établit conjointement. Ce que je n'établis pas, c'est une borne universelle : mes
chiffres valent pour ma tête, mes jeux et ma carte. "Comblé" veut dire "la démonstration
manquante a été faite une fois", pas "la question est close". »

---

# Section 4 — Positionnement scientifique

---

### Q-41 — « En une phrase : qu'y a-t-il de nouveau chez vous par rapport à TinyOL ? »

**Réponse** : « TinyOL démontre qu'on peut apprendre en ligne sur un Cortex-M4 pour un
surcoût de 10 % sur la latence, sur une tâche de vibration. Ce que j'ajoute : quatre familles
de méthodes comparées sur la même carte et le même protocole, une parité numérique avec la
référence PC prouvée prédiction par prédiction — ce qui permet d'affirmer que ce qu'on mesure
sur carte décrit bien le modèle de référence —, une décomposition mémoire qui inclut la pile
et le coût de l'adaptation, et l'axe quantification. TinyOL est une démonstration de
faisabilité ; mon travail est un protocole de mesure et de comparaison. »

### Q-42 — « "Aucun travail ne comble les trois lacunes simultanément" — comment vérifie-t-on une affirmation pareille ? »

**Réponse** : « On ne la vérifie pas au sens strict, et c'est la limite de toute
revendication de ce type. Ce que je peux défendre, c'est la vérification sur le corpus que
j'ai constitué : pour chacun des travaux recensés, j'indique laquelle des trois lacunes il
adresse et lesquelles il laisse ouvertes — QLR-CL quantifie mais garde l'entraînement en
flottant, LifeLearner tient l'enveloppe mémoire mais sur une tâche supervisée sans mesure du
surcoût d'adaptation, TinyOL fait de l'apprentissage en ligne mais ne publie pas de
décomposition mémoire. La formulation prudente serait "aucun des travaux recensés", et c'est
ainsi qu'il faut la lire. »

### Q-43 — « Le "triple gap" n'est-il pas un cadre construit après coup pour donner une unité au travail ? »

**Réponse** : « Il a été formulé après l'état de l'art et avant les expériences — c'est le
chapitre 3 du manuscrit, et il structure la feuille de route du stage, pas seulement sa
rédaction. Ce qui a bougé, c'est le poids relatif des trois : la quantification, initialement
secondaire, a pris une place centrale quand les premiers résultats se sont révélés
intéressants, et le manuscrit le documente au §3.2. Je concède que le cadre est un outil de
narration autant qu'un cadre scientifique. Sa vertu, c'est qu'il m'a obligé à publier le
bilan du troisième gap tel qu'il est — nuancé — au lieu de ne montrer que les deux qui
marchent. »

### Q-44 — « Ce travail est-il publiable ? Où ? »

**Réponse** : « Une partie l'est, et j'en ai rédigé la forme : un article court centré sur la
tête EWC en INT8 sur microcontrôleur, qui suit le fil parité FP32 mesurée → effondrement de
la PTQ naïve mesuré → récupération par noyau calibré, avec la distinction explicite entre ce
qui est mesuré sur carte et ce qui est émulé. Les cibles naturelles sont les ateliers TinyML
et embarqué — le type de venue où EMDL ou les workshops TinyML de MLSys publient. Ce qui
manquerait pour une conférence pleine : les répétitions statistiques, une seconde carte, et
les mesures d'énergie. »

### Q-45 — « Quelle serait la question de thèse qui prolonge ce stage ? »

**Réponse** : « Celle du verrou que je n'ai pas levé : *comment un capteur décide-t-il seul
qu'il doit apprendre, et de quoi ?* Mon prototype montre qu'on peut décider à bord quand
mettre à jour, avec une parité de décision parfaite et 97 % de mises à jour économisées.
Mais dès que le capteur fabrique lui-même son étiquette, la performance chute à 0,504 sur
Pronostia. Une thèse là-dessus articulerait trois choses : distinguer une dérive légitime
d'une anomalie à signaler, produire un signal d'apprentissage fiable sans supervision
humaine, et garantir qu'un modèle qui se met à jour tout seul ne diverge pas — c'est-à-dire
la question de la sûreté d'un modèle apprenant embarqué. »

### Q-46 — « Sur un microcontrôleur sans FPU, vos conclusions tiendraient-elles ? »

**Réponse** : « Une d'entre elles s'inverserait, et c'est ce qui la rend intéressante. Sur un
Cortex-M0+ sans unité flottante, chaque opération en virgule flottante est émulée en
logiciel, à un coût de plusieurs dizaines de cycles. Le calcul entier deviendrait alors
franchement gagnant, et le surcoût des deux étages de conversion — qui est ce qui plombe
l'INT8 chez moi — serait largement amorti. Mon résultat n'est donc pas "l'INT8 est inutile",
mais "le bénéfice de l'INT8 dépend de la présence d'une FPU, et il faut le mesurer sur la
cible plutôt que de le supposer". C'est une conclusion plus utile qu'un gain générique, et
c'est la première expérience que je ferais avec une seconde carte. »

### Q-47 — « Pourquoi EWC et pas Synaptic Intelligence, MAS ou Learning without Forgetting ? »

**Réponse** : « Les trois sont dans mon état de l'art, et elles se distinguent surtout par la
manière d'estimer l'importance des poids : accumulation du gradient le long de la trajectoire
pour SI, sensibilité de la sortie pour MAS, distillation depuis le modèle précédent pour LwF.
Sur mes contraintes, elles ont toutes le même profil mémoire — un scalaire d'importance par
paramètre —, sauf LwF qui demande de garder le modèle précédent, donc de doubler l'empreinte.
J'ai retenu EWC parce que c'est la référence de la famille, la plus documentée et la plus
comparable à la littérature. Comparer les estimateurs d'importance entre eux, sur carte,
serait une extension naturelle et peu coûteuse : le firmware ne changerait que sur le calcul
de l'importance. »

### Q-48 — « Vous écartez le replay, qui est pourtant la famille la plus efficace contre l'oubli. »

**Réponse** : « Je l'écarte pour une raison de contrainte, pas de performance, et le
manuscrit le dit : la taille du buffer pèse directement sur la RAM, ce qui en fait la famille
la plus difficile à porter. C'est d'ailleurs exactement le problème que QLR-CL attaque, en
quantifiant le buffer de rejeu latent pour en diviser l'empreinte par quatre. Un rejeu latent
quantifié serait la suite logique de mon travail — et il combinerait mes deux axes, puisqu'il
mettrait la quantification au service de l'empreinte du continual learning et non plus
seulement de celle des poids. »

### Q-49 — « Qu'apporterait CMSIS-NN, que vous citez sans l'utiliser ? »

**Réponse** : « CMSIS-NN exploite les instructions SIMD du Cortex-M4, qui permettent de
traiter quatre entiers 8 bits par instruction. Sur le poste qui domine mon surcoût — les
produits scalaires entiers, environ 6 785 cycles — on peut espérer un facteur proche de
quatre. En revanche, l'autre poste, la remise à l'échelle des sorties, à 3 777 cycles, est un
arrondi flottant par neurone qui ne se vectorise pas de la même façon. Donc CMSIS-NN
rapprocherait sans doute l'INT8 du FP32 sans nécessairement le dépasser sur cette cible.
C'est une piste ouverte que je n'ai pas explorée, et la décomposition au cycle près que je
publie est précisément ce qui permet d'en prédire le rendement avant de la tenter. »

### Q-50 — « Vos résultats se transposent-ils à d'autres domaines que la maintenance prédictive ? »

**Réponse** : « La méthode oui, les chiffres non. Ce qui se transpose, c'est la chaîne de
portage à parité garantie et le protocole de mesure — ils sont indépendants du domaine
applicatif. Ce qui ne se transpose pas, c'est la conclusion sur les modèles : elle dépend de
la dimension des entrées, du déséquilibre des classes et de la nature de la dérive. Un
domaine à entrées de haute dimension, comme l'image, sortirait immédiatement de l'enveloppe —
j'ai d'ailleurs une limite mesurée dans ce sens, où une matrice de covariance en O(k²) fait
déborder la SRAM à 128 variables. »

---

# Section 5 — Questions industrielles et applicatives

---

### Q-51 — « Sur le terrain, qui fournit l'étiquette "fautif" ? Sans opérateur, votre EWC n'apprend rien. »

**C'est la question la plus dure du bloc.** Ne pas la contourner ; donner le chiffre qui fait
mal avant qu'on ne le demande.

**Réponse** :

> « C'est le verrou principal, et je ne le résous pas. Trois régimes sont possibles, et je
> les ai mesurés. Avec une étiquette vraie fournie par un opérateur — un technicien qui
> confirme une intervention —, le modèle apprend et la performance est préservée : sur
> Monitoring, une politique de mise à jour déclenchée conserve un F1 de 0,919, soit
> exactement le niveau du modèle gelé, tout en économisant 97 % des mises à jour. C'est le
> régime d'apprentissage actif, et il est réaliste industriellement : on ne demande à
> l'opérateur que 2,5 % des échantillons.
>
> En pseudo-étiquetage entièrement autonome, où le capteur fabrique sa propre étiquette, la
> performance tient sur Monitoring mais chute à 0,504 sur Pronostia. L'économie de mises à
> jour se paie alors sur la qualité de l'étiquetage.
>
> Ma recommandation de déploiement est donc l'apprentissage actif, pas l'autonomie complète :
> le capteur décide *quand* demander, l'humain décide *quoi* apprendre. Rendre le troisième
> régime fiable, c'est le sujet de recherche qui prolonge ce travail. »

### Q-52 — « Que se passe-t-il si le modèle apprend une dérive qui était en réalité une panne ? »

**Réponse** : « C'est le risque central de ce type de système, et c'est exactement la
question posée dès le chapitre 2 : une déviation par rapport au normal appris est soit une
anomalie à signaler, soit un changement légitime auquel s'adapter. Confondre les deux, c'est
apprendre la panne comme un nouveau normal et cesser de l'alerter — une défaillance
silencieuse, la pire.
>
Mon dispositif ne tranche pas ce dilemme automatiquement, et c'est pour cela que je
recommande de garder l'humain dans la boucle sur le label. Les garde-fous que je vois : ne
jamais mettre à jour sur un échantillon dont le score d'anomalie est extrême, borner le
volume de mises à jour par unité de temps — ce que fait déjà mon gate, à 2,5 % —, et
conserver un modèle de référence gelé en parallèle pour détecter une divergence. Le premier
et le troisième ne sont pas implémentés ; le deuxième l'est. »

### Q-53 — « Combien coûte un déploiement ? »

**Réponse** : « Le composant lui-même est une carte à microcontrôleur de l'ordre de quelques
dizaines d'euros en unité, moins en volume — c'est le point de tout l'exercice : on met de
l'intelligence sur du matériel de commodité, sans passerelle ni serveur ni abonnement réseau.
Le coût réel n'est pas là : il est dans l'intégration au capteur existant, l'étalonnage
initial sur la machine, et la maintenance logicielle. Ce que ma mesure de RAM apporte à cette
discussion, c'est qu'à 105 Ko sur 256, il reste de la marge pour la logique applicative, ce
qui évite d'avoir à monter en gamme de composant. »

### Q-54 — « Comment met-on à jour le firmware d'un parc de capteurs déployés ? »

**Réponse** : « Ce n'est pas dans le périmètre du stage, et je ne l'ai pas traité. Ce que mon
architecture apporte à cette question, c'est une séparation nette entre le code et les
poids : les poids sont générés par script dans un fichier d'en-tête dédié, jamais écrits à la
main dans le code. Une mise à jour de modèle est donc une mise à jour de données, plus légère
et plus sûre qu'une mise à jour de code. C'est une base pour un mécanisme de mise à jour à
distance, pas un mécanisme en soi. »

### Q-55 — « Quelle autonomie sur batterie ? »

**Réponse** : « Je n'ai pas de chiffre à vous donner, et je préfère le dire que de vous en
donner un faux. La chaîne de calcul de l'autonomie est en place — elle part du courant moyen
mesuré et de la capacité de la batterie — mais la mesure de courant elle-même n'a pas pu être
faite dans le temps du stage. Ce que je peux dire qualitativement : à quelques centaines de
microsecondes de calcul pour une cadence de l'ordre du hertz, le calcul ne représente qu'une
fraction infime du cycle, et l'autonomie sera dominée par la consommation au repos et par
l'acquisition capteur, pas par le modèle. C'est d'ailleurs ce qui rend la question du gain
énergétique de l'INT8 douteuse a priori. »

### Q-56 — « Comment cela passe-t-il à l'échelle d'une usine, avec des centaines de capteurs ? »

**Réponse** : « Chaque capteur est indépendant, ce qui est la propriété intéressante : pas de
serveur central, pas de remontée de données brutes, donc pas de goulot ni de question de
confidentialité. Chaque capteur apprend sa machine, ce qui est précisément l'argument de
l'apprentissage sur l'appareil — deux pompes du même modèle n'ont ni le même vieillissement
ni le même régime. Ce que cela ne donne pas, c'est le partage d'apprentissage entre capteurs ;
mutualiser sans centraliser relèverait de l'apprentissage fédéré, qui est une autre question
et que je n'aborde pas. »

### Q-57 — « Comment certifie-t-on un modèle qui se modifie tout seul en exploitation ? »

**Réponse** : « C'est une objection de fond dans les secteurs réglementés, et je n'ai pas la
réponse complète. Deux éléments de mon travail y contribuent. D'abord la traçabilité : la
décision de mise à jour prise à bord est reproductible hors ligne — la parité de verdict
carte ↔ PC vaut 1,000 —, donc on peut auditer après coup pourquoi le modèle a appris. Ensuite
la bornitude : le coût mémoire et le coût en temps de la mise à jour sont mesurés et bornés,
ce qui est une condition nécessaire pour un système temps réel certifiable. Ce qui manque,
c'est une garantie sur la trajectoire du modèle — rien ne prouve qu'une séquence de mises à
jour ne le dégrade pas. En pratique, on déploierait avec un modèle de référence gelé en
parallèle et une bascule si les deux divergent trop. »

### Q-58 — « En quoi cela intéresse-t-il Edge Spectrum ? »

**Réponse** : « L'entreprise conçoit des capteurs intelligents ; le sujet de fond est de
savoir jusqu'où on peut pousser l'intelligence dans le capteur plutôt que dans la passerelle.
Ce que ce stage apporte concrètement, c'est un chiffrage : voilà ce que coûte, en RAM et en
microsecondes, un modèle qui continue d'apprendre sur un Cortex-M4 — et voilà une chaîne
d'outillage pour porter un modèle de PyTorch au firmware avec la garantie que le
comportement est identique. Cette chaîne est réutilisable indépendamment des modèles que j'ai
choisis. »

### Q-59 — « Pourquoi ne pas simplement mettre une passerelle plus puissante ? »

**Réponse** : « C'est une option légitime, et dans beaucoup de cas la bonne. Les trois raisons
de ne pas le faire sont celles du chapitre 1 : le coût et la disponibilité du lien réseau,
la confidentialité des données de production, et la latence de la boucle de décision. Ma
contribution ne dit pas "il faut faire de l'embarqué" ; elle dit "voilà le coût réel de
l'embarqué, mesuré", ce qui permet d'arbitrer sur des chiffres plutôt que sur une intuition.
Et sur cette carte, l'arbitrage est confortable : 40 % de la RAM et quelques centaines de
microsecondes. »

### Q-60 — « Sur quelle cadence d'échantillonnage votre système tient-il ? »

**Réponse** : « Le pire cas mesuré, avec mise à jour continue, est de l'ordre de 340 µs pour
EWC et de 2,1 ms pour HDC. En prenant une marge d'un facteur dix pour l'acquisition et le
reste de l'applicatif, cela situe la cadence soutenable autour de quelques centaines de hertz
pour EWC et de quelques dizaines pour HDC. C'est compatible avec de la supervision de
process — température, pression, vibration agrégée — mais pas avec de l'analyse de forme
d'onde vibratoire brute à plusieurs kilohertz, qui demanderait une extraction de variables en
amont. Mes jeux de vibration sont d'ailleurs traités en variables agrégées, pas en signal
brut. »

---

# Annexe — Chiffres à connaître par cœur

Ne jamais improviser un chiffre. Ceux-ci sont **dans le manuscrit**, donc opposables.

**Matériel**
- NUCLEO-F439ZI, Cortex-M4 @ 180 MHz, FPU simple précision, 256 Ko SRAM (192 + 64 CCM), 2 Mo Flash, pas de NPU

**RAM (ch. 6)**
- noyau minimal EWC seul : ≈ 1 000 B de `.bss`
- système multi-modèle de référence : `.bss` = 105 036 B — **40,1 %** de 256 Ko
- `.data` = 460 B
- RAM totale Monitoring : 460 + 100 152 + 4 688 = **105 300 B** (40,2 %)
- RAM totale Pronostia : **110 192 B** (42,0 %)
- pic de pile : 4 688 B en mise à jour contre 4 416 B en inférence (Monitoring)
- pire cas mesuré : 183 936 B (70,2 %), CMAPSS toutes variables, k = 21

**Latence (ch. 6 et 7)**
- budget : 100 ms
- EWC inférence seule : 48–65 µs · EWC inférence + MAJ CL : 239–340 µs (≈ ×5)
- Mahalanobis : ≈ 5 µs · pire cas mesuré : HDC INT8 ≈ 2,1 ms
- EWC INT8 : 74 µs contre 48–50 µs en FP32 (+24 à +26 µs)
- décomposition INT8 à 180 MHz : MAC ≈ 6 785 cycles · requantification ≈ 3 777 · déquantification ≈ 200

**Parité (ch. 6)**
- régime gelé : **1,000**, aucun désaccord, sur 7 534 à 7 672 échantillons
- régime en ligne : 0,963 à 0,989 · Δ acc_final ≤ **0,007**

**Gap 1 (ch. 5)**
- Mahalanobis × CMAPSS : accuracy 0,745 pour F1 0,269
- EWC multiclasse : moyenne des F1 post-tâche 0,981 → F1 du modèle final **0,240** · AF = 0,847
- carte : 0,243 en gelé, 0,507 en ligne
- tableau 5.1, F1 fautif PC | carte, condition 5feat :

| | Monitoring | CMAPSS | Pronostia |
|---|---|---|---|
| EWC | 0,893 \| 0,947 | 0,456 \| 0,381 | 0,930 \| 0,968 |
| HDC | 0,565 \| 0,000 | 0,000 \| 0,000 | 0,425 \| 0,000 |
| TinyOL | 0,754 \| 0,927 | 0,197 \| 0,300 | 0,285 \| 0,889 |
| Mahalanobis | 0,698 \| 0,710 | 0,269 \| 0,214 | 0,305 \| 0,273 |

- Paderborn : EWC 0,800 \| 0,800 · HDC 0,565 \| 0,000 · TinyOL 0,703 \| 0,000 · Maha 0,071 \| 0,113
- HDC sur carte : accuracy 0,867 (Monitoring) et 0,900 (Pronostia) pour F1 nul

**Gap 3 (ch. 7)**
- gain RAM INT8 : ×2,33 à ×4,0 selon modèle et jeu
- QAT côté PC : Δ ≤ **0,006** pour EWC
- PTQ naïve embarquée : F1 **0,07 à 0,15** contre ≈ 0,92 en FP32
- cause unique isolée par ablation : **échelle fixe 1/128 non calibrée**

**Perspectives (ch. 8)**
- gate autonome : taux de MAJ 0,025 contre 1,0 → **97 %** économisés
- latence 79–82 µs contre 238–251 µs · surcoût mémoire ≈ 300 B
- parité de verdict carte ↔ PC : **1,000**
- Monitoring pré-entraîné : F1 0,919 en gate déclenché = 0,919 en gelé
- Pronostia : 0,889 (étiquette vraie) contre 0,916 (gelé) · **0,504** en pseudo-étiquetage
- features : EWC × CMAPSS, F1 0,38 → 0,62 en passant de 5feat à all, mais `.bss` 104 956 → 183 936 B

---

# Checklist de couverture

Chaque figure et chaque tableau du manuscrit doit avoir au moins une question associée.

| Élément | Question(s) |
|---|---|
| Tab. 4.1 — jeux de données | Q-14, Q-33 |
| Tab. 4.2 — modèles | Q-34, Q-47 |
| Fig. ch5_accuracy_vs_f1 | Q-20 |
| Fig. ch5_oubli_catastrophique | Q-19, Q-37 |
| **Tab. 5.1 — grille 4×3** | **Q-01, Q-02**, Q-23, Q-24 |
| Fig. ch5_f1_grille_pc_board | Q-01, Q-23 |
| Fig. ch5_paderborn_ewc_seul | Q-36 |
| Fig. ch6_ram_trois_niveaux | Q-25, Q-26 |
| Fig. ch6_ram_totale_decomposee | Q-10, Q-12 |
| Fig. ch6_pic_pile_par_phase | Q-11 |
| Fig. ch6_latence_inference_vs_maj | Q-27, Q-60 |
| Fig. ch6_latence_par_modele | Q-21, Q-60 |
| Fig. ch6_parite_gele_vs_online | Q-01, Q-18 |
| Fig. ch7_ram_poids_vs_systeme | Q-28 |
| Fig. ch7_ablation_echelle | Q-22, Q-28 |
| Fig. ch7_noyau_v2_recuperation | Q-28 |
| Fig. ch7_moment_quantification | Q-16 |
| Fig. ch7_latence_int8_breakdown | Q-28, Q-29, Q-46, Q-49 |
| Tab. A.1 — jeux détail | Q-14 |
| Tab. A.2 — grille 4×5 | Q-02, Q-24, Q-34 |
| §A.3 — limites déclarées | Q-24, Q-50 |
