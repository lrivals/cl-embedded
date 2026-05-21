# Skill : Analyse de Sprint

> **Usage** : Analyser un sprint complet à partir de ses fichiers de documentation.  
> **Déclencheur** : "analyse le sprint N" / "fais un résumé du sprint N" / "conclus le sprint N"

---

## Contexte à fournir

Indiquer le numéro du sprint cible. Si plusieurs sprints sont à comparer, les lister.

---

## Instructions pour Claude

### Étape 1 — Lecture des fichiers

Lis **tous** les fichiers correspondant au pattern :

```
docs/sprints/sprint_<N>/S*.md
```

Commence par le fichier master `S<N>00_*` (vue d'ensemble), puis lis chaque fichier tâche dans l'ordre numérique.

---

### Étape 2 — Extraction structurée

Présente l'analyse en sections suivantes :

#### 2.1 Contexte du sprint

| Champ | Valeur |
|-------|--------|
| Objectif principal | |
| Datasets / modèles | |
| Statut global | ✅ CLOSED / 🔴 EN COURS |
| Durée estimée | |
| Dépendances | Sprint(s) précédent(s) |

#### 2.2 Tâches

Tableau de toutes les tâches du sprint :

| ID | Livrable | Priorité | Statut |
|----|----------|:--------:|:------:|
| SN-01 | … | 🔴/🟡/🟢 | ✅/❌ |

#### 2.3 Expériences lancées

| Exp ID | Modèle | Dataset | Stratégie |
|--------|--------|---------|-----------|
| exp_XXX | … | … | refit / accumulate |

#### 2.4 Résultats — Métriques clés

Tableau par dataset et modèle (ne rapporter que les chiffres issus des fichiers) :

| Modèle | avg_AUROC | AF | RAM (Ko) | ≤ 64 Ko ? |
|--------|-----------|----|----------|:---------:|

Inclure une ligne de comparaison **refit vs accumulate** si les deux stratégies ont été testées.

#### 2.5 Points saillants

- Résultats contre-intuitifs (ex. ratio normal ≠ difficulté réelle)
- Décisions architecturales prises et leur justification
- Contraintes hardware impactées (RAM, latence, INT8)
- Limites ou questions ouvertes identifiées

---

### Étape 3 — Résumé / Conclusion

Rédige une conclusion **≤ 600 mots** en trois sections :

#### Bilan technique
Ce qui a été livré (loaders, expériences, tests, notebooks), les chiffres-clés, le statut final.

#### Points clés (3 maximum)
Les enseignements les plus importants pour la suite du projet.  
Pour chaque point, préciser s'il adresse **Gap 1** (données industrielles réelles), **Gap 2** (< 100 Ko RAM mesurés), ou **Gap 3** (quantification INT8) si applicable.

#### Recommandations
Ce que ce sprint implique pour les sprints suivants : priorités, configs à ajuster, modèles à écarter ou promouvoir pour le portage MCU.

---

## Règles de rédaction

- **Factuel** : cite les métriques exactes issues des fichiers, jamais de chiffres inventés
- **Concis** : max 600 mots pour la conclusion (Étape 3)
- **Tableaux markdown** pour toutes les métriques
- **Pas de paraphrase** des titres de tâche : synthétise, ne recopie pas
- **Lier aux gaps** quand pertinent (`Gap 1 ✅`, `Gap 2 ✅`, `Gap 3 ❌`)
