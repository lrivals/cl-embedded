# Présentation technique — Portage C & pipeline expérimental board (texte parlé)

> Texte parlé associé au plan [`03_structure_portage.md`](03_structure_portage.md).
> Registre : technique, public expert embarqué. **Les chiffres et codes hexadécimaux sont à prononcer
> exactement** — ils sont vérifiés sur `firmware/stm32f4_blink/inc/pipeline.h` et `scripts/sensor_stream.py`,
> ne pas improviser. Cette présentation explique **comment on traite les données et comment le code est
> porté** ; elle ne discute pas les résultats de performance (ils sont dans la présentation séminaire).

---

## Slide 1 — Le chemin d'un sample

« Dans la présentation séminaire, j'ai montré que nos quatre méthodes de Continual Learning marchent.
Aujourd'hui, je veux ouvrir le capot et vous montrer **comment elles vivent réellement sur un
microcontrôleur**. Le fil conducteur sera simple : on va suivre **un seul échantillon capteur**, depuis le
fichier de dataset sur mon PC, jusqu'à la ligne de résultat dans un fichier JSON — en passant physiquement
par la carte. À chaque étape, je m'intéresse au **traitement de la donnée** et au **portage du code**, pas
aux performances. Voici la carte du voyage. »

## Slide 2 — La cible NUCLEO-F439ZI

« La cible, c'est une NUCLEO-F439ZI : un Cortex-M4 à 180 MHz — fréquence que je mesure moi-même via le
compteur DWT —, 256 kilo-octets de SRAM, 2 méga-octets de Flash. Le point déterminant pour le portage,
c'est qu'elle a une **FPU matérielle** en simple précision : les opérations flottantes coûtent un cycle,
c'est ce qui rend les latences en microsecondes possibles. En revanche, **pas de NPU** : le forward pass
*et* la backpropagation s'exécutent tous les deux sur le CPU, en FP32. Et trois interdits qui structurent
tout le code : pas d'OS — on est en bare-metal, une boucle infinie ; **pas de `malloc`** — toute la mémoire
est statique ; pas de double précision. Note de méthode : j'écris tout pour tenir dans 64 ko, le budget de
notre Gap 2 et de la future STM32N6, même si la carte en offre 256. »

## Slide 3 — Toolchain & Makefile dual

« Côté outils : `arm-none-eabi-gcc` pour la compilation croisée, OpenOCD et le ST-LINK pour flasher, Renode
pour faire tourner la CI sans carte physique, et Unity pour les tests. Le mécanisme que je veux souligner,
c'est le **Makefile dual**. `make all` produit le binaire ARM. Mais `make test` recompile **exactement le
même code C** sur mon PC en x86, avec deux defines — `TEST_MODE` et `TEST_HOST` — qui désactivent la HAL
STM32 et simulent le DWT. Conséquence : mes tests unitaires couvrent le **vrai code de production**, pas une
copie. C'est ma première garantie de fidélité du portage. »

## Slide 4 — Architecture firmware

« À l'intérieur du binaire, l'organisation est volontairement simple. Un seul fichier orchestre tout,
`pipeline.c` : il reçoit la trame, vérifie le CRC, normalise, route vers le bon modèle, et émet la réponse.
Les modèles eux-mêmes sont des **têtes indépendantes** — `mahalanobis.c`, `ewc_head`, `hdc.c`, `tinyol.c`,
le méta-modèle —, chacune doublée d'une variante INT8 aux mêmes signatures. Le point clé, c'est la
**gestion mémoire en trois zones**. En Flash, immuable, les poids d'origine, les statistiques de
normalisation, la projection HDC. En `.bss`, la SRAM mutable : les **poids vivants** que le SGD modifie, la
matrice de Fisher, les métriques. Et la stack, pour les temporaires du forward. Au démarrage,
`pipeline_init()` fait un `memcpy` de la Flash vers le `.bss` : les poids **doivent** être recopiés dans une
zone modifiable, sinon on ne peut pas apprendre en ligne. Tout est dimensionné par des `#define`, zéro
allocation dynamique. »

## Slide 5 — Du PC au C : export des poids

« Ces poids d'origine, d'où viennent-ils ? Ils sont entraînés en Python — PyTorch ou scikit-learn — puis
**exportés** vers des headers C par une famille de scripts : `export_weights_c.py` pour Mahalanobis, l'EWC
binaire et le méta-modèle, et des variantes pour TinyOL, la régression RUL, le multiclasse. Le script génère
des tableaux `static const float` dans `model_weights.h` et ses cousins. **Règle absolue du projet** : on
n'édite **jamais** ces headers à la main — on régénère, pour la reproductibilité. Et la distinction
conceptuelle importante : la version en Flash est la **référence θ-étoile**, figée ; la copie en `.bss` est
celle que l'apprentissage embarqué fait évoluer. »

## Slide 6 — Pipeline de données côté PC

« Maintenant, qu'est-ce qu'on envoie comme donnée ? On part de six datasets industriels réels — CWRU,
Pronostia, CMAPSS, Paderborn, et deux datasets de maintenance. Sur chaque signal, on **extrait des features**
fréquentielles et temporelles : RMS, kurtosis, facteur de crête, skewness. On en **sélectionne les cinq
plus discriminantes**, hors ligne. Puis on **normalise en Z-score**. Et voilà le détail crucial pour la
suite : la normalisation utilise **exactement les mêmes** moyennes et écarts-types — `ZSCORE_MEAN`,
`ZSCORE_STD` — que ceux figés dans `model_weights.h`. Le PC et la carte normalisent avec les mêmes
constantes. Sans ça, pas de parité numérique. Le piège classique : si je change de dataset, je dois
recalculer ces statistiques et le seuil, et re-flasher. »

## Slide 7 — Trame UART, octet par octet

« Ces cinq flottants voyagent jusqu'à la carte sur l'UART, à 115 200 bauds, via le port COM virtuel du
ST-LINK. La communication est **binaire**, pas du texte. La trame est construite par `build_frame_v2`, et son
ordre réel — je le précise parce que d'anciennes docs le décrivaient différemment — est le suivant : MAGIC
sur deux octets, `0xAB 0xCD` ; la version ; l'identifiant de tâche ; le timestamp sur quatre octets ; le
nombre de features ; les features en flottants little-endian ; le label ; l'octet FLAGS ; et un CRC8. Le
header fait neuf octets, et **pour cinq features, la trame totale fait trente-deux octets**. Le CRC8 — polynôme
0x07, le même des deux côtés — couvre toute la charge utile ; s'il est faux, la trame est rejetée et la LED
passe au rouge. »

## Slide 8 — FLAGS : sélecteur de mode par nibble

« Et maintenant l'octet le plus intéressant : FLAGS. Un seul octet, mais il décide de tout le comportement.
Je le lis en **deux quartets**. Le quartet bas, ce sont des **actions combinables** : `0x01` UPDATE — fais un
pas de SGD avec ce sample ; `0x04` CONSOLIDATE — c'est une frontière de tâche, déclenche la consolidation
EWC ou la binarisation HDC ; `0x08` RESET ; `0x02` PROFILING. Le quartet haut, c'est le **mode**, une valeur
unique : `0x10` pour l'EWC, `0x20` HDC, `0x40` EWC INT8, `0x80` TinyOL. Puis des modes composites — `0x30`
multiclasse, `0x50` régression RUL, `0x60` HDC INT8, `0x70` le mode DUAL, `0xC0` TinyOL INT8. Et enfin les
ensembles : `0x90`, `0xA0`, `0xB0` pour les paires Mahalanobis plus un supervisé, `0xD0` et `0xE0` pour le
triple avec méta-modèle. Le détail qui m'a coûté du débogage : l'**ordre d'évaluation** compte. Comme `0x70`
contient les bits de `0x30`, il faut tester le mode DUAL **avant** le multiclasse. D'où cette cascade :
triple, puis paire, puis dual, puis multiclasse, RUL, les INT8, les modes simples, et par défaut
Mahalanobis. »

## Slide 9 — Cycle d'une inférence sur la carte

« Côté carte, `pipeline_run()` tourne en boucle, **bloquée sur la réception UART**. Quand une trame arrive,
le cycle se déroule en sept temps. Un : réception octet par octet jusqu'à la trame complète. Deux : décodage
et vérification du CRC — rejet si invalide. Trois : **normalisation Z-score**. Quatre : **routage** selon le
quartet haut, vers la bonne tête. Cinq : le **forward**, en FP32 sur la FPU, qui produit prédiction et
confiance. Six, si le flag UPDATE est là : la **mise à jour**, et elle diffère selon le modèle — pour l'EWC,
c'est un gradient cross-entropy plus la pénalité de Fisher, `lambda fois F fois (theta moins theta-étoile)` ;
pour le HDC, pas de gradient du tout, on **accumule** simplement l'hypervecteur ; pour Mahalanobis, on met à
jour la moyenne en EMA, façon Welford. Et si le flag CONSOLIDATE est présent — fin de tâche — on consolide.
Sept : on met à jour les **métriques en ligne** — accuracy, AUROC, oubli, RMSE, F1 — puis on émet. Tout se
passe en variables de stack, zéro malloc. Une nuance : TinyOL ne fait pas de backprop sur la carte, seule
sa petite tête OtO bouge. »

## Slide 10 — Réponses & parité board↔PC

« La carte répond, et la **longueur de la réponse identifie le mode** — c'est `sensor_stream.py` qui
désambiguïse. Quatorze octets pour le v2, vingt-trois pour le v3 qui ajoute accuracy, AUROC et oubli
**calculés à bord**, vingt-cinq pour le DUAL qui combine RUL et faute, vingt-deux pour une paire, vingt-sept
pour un triple avec le verdict du méta-modèle. Et voici mon **garde-fou méthodologique**, le plus important
de cette présentation : je vérifie systématiquement la **parité numérique entre la carte et le PC**. Même
entrée, même sortie — par exemple, la parité du méta-modèle est de 1,000 sur trois cents échantillons. C'est
ce qui me permet de distinguer un **bug de portage** d'une **limite du modèle**. Si la carte donne exactement
le même chiffre que le PC, alors mon portage C est fidèle — un point c'est tout. Si le modèle est mauvais,
c'est une question d'algorithme, pas de portage. »

## Slide 11 — Profiling : comment on mesure

« Comment j'obtiens les latences et la RAM ? Pour la **latence**, j'utilise le compteur de cycles matériel
du Cortex-M4, le DWT CYCCNT. `profiling_start` capture le compteur, `profiling_stop` calcule le delta, et je
divise par 180 — puisqu'à 180 MHz, une microseconde fait 180 cycles, avec une résolution d'environ 5,5
nanosecondes. Important : le **périmètre mesuré va de la réception à l'émission**, il inclut donc l'UART.
Pour la **RAM statique**, j'utilise les symboles du linker : `bss_bytes` égale `_ebss moins _sbss`, calculé
au runtime et encodé dans la réponse. Et je peux **vérifier indépendamment** avec `arm-none-eabi-size` sur
le binaire ELF. Pas d'instrument externe, tout est dans la carte. »

## Slide 12 — Mise en place d'une expérience

« Dernière étape : passer d'un sample à une **expérience complète**. C'est le rôle de
`board_experiment_recorder.py`. Il orchestre une **séquence multi-tâches** : pour chaque tâche, il envoie N
samples avec le flag PROFILING, plus UPDATE si on apprend, et il pose le flag CONSOLIDATE sur le **dernier
sample de chaque tâche** — c'est ce qui matérialise une frontière de domaine pour l'EWC. Détail pratique très
utile : le mode `--dry-run` rejoue toute l'orchestration **sans carte**, en loopback Python, pour valider le
protocole avant la session réelle. À la fin, tout atterrit dans `experiments/exp_*/` : le CSV des réponses
brutes, le `results.json` avec les métriques CL obligatoires, le snapshot de config, et le profiling. Pour
les paires et le triple, il y a `board_pair_recorder.py`. Et une règle d'or : on ne touche jamais au
protocole sans synchroniser **à la fois** `sensor_stream.py` et `pipeline.c`. »

« Voilà le voyage complet d'un échantillon : du dataset au JSON, tout est mesuré, reproductible, et validé
par parité. Je prends vos questions. »

---

## Notes anti-débordement

- Si le temps manque : **compresser les slides 2 et 3** (hardware + toolchain) — l'auditoire connaît le MCU.
- **Ne jamais sacrifier** les slides 7–10 (trame, FLAGS, cycle, parité) : c'est le cœur du « comment ».
- La note de parité board↔PC (slide 10) est le message à retenir — la garder même en version courte.
- L'annexe INT8 / `ewc_head.c` ne sort que sur question.
