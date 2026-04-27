# Skill : Mise à jour du Diagramme Pipeline

> **Usage** : Synchroniser `pipeline_network_diagram.html` avec l'état réel du dépôt.
> **Déclencheur** : "mets à jour le diagramme pipeline" / "synchronise le diagramme" / "update pipeline diagram"

---

## Procédure (exécuter dans l'ordre)

### Étape 1 — Lire l'état actuel du diagramme

Lire `pipeline_network_diagram.html` lignes 571–900 pour extraire :
- Tous les `id` existants du tableau `nodes`
- Tous les paires `{source, target}` du tableau `links`

### Étape 2 — Scanner le projet

Lister les fichiers réels dans ces dossiers et mapper leur type de nœud :

| Dossier | Filtre | Type nœud |
|---------|--------|-----------|
| `configs/` | `*.yaml` | `config` |
| `src/data/` | `*.py` (loaders) | `data_loader` |
| `src/models/` | `*.py` (classes) | `ml` |
| `src/training/` | `*.py` | `training` |
| `src/evaluation/` | `*.py` | `evaluation` |
| `src/utils/` | `*.py` | `utils` |
| `scripts/` | `*.py` | `main` |
| `experiments/exp_*/` | répertoires | `output` |
| `data/raw/` | `*.csv` | `data` |

### Étape 3 — Identifier les deltas

Comparer la liste des fichiers réels avec les IDs existants et produire trois listes :

- **À ajouter** : fichiers présents dans le dépôt, absents du tableau `nodes`
- **À modifier** : `file` ou `description` d'un nœud existant qui ne correspond plus
- **À supprimer** : nœuds dont le `file` n'existe plus dans le dépôt

### Étape 4 — Proposer les changements à l'utilisateur

Présenter les deltas sous forme de tableau concis **avant d'écrire quoi que ce soit**.
Attendre confirmation. Exemple :

```
Ajouts (3) :
  + hdc_anomaly_config  configs/hdc_anomaly_detection_config.yaml  [config, ml_cl, equipment_monitoring]
  + anomaly_metrics     src/evaluation/anomaly_metrics.py           [evaluation, eval_exp, equipment_monitoring]
  + tinyol_anomaly      src/models/tinyol/tinyol_anomaly_detector.py [ml, ml_cl, equipment_monitoring]

Suppressions (0) :
  aucune

Modifications (0) :
  aucune
```

### Étape 5 — Appliquer les changements dans le HTML

Pour les **nœuds à ajouter** :
- Insérer dans la section commentée correspondante du tableau `nodes`
  (ex. `// ── Configs ──`, `// ── Modèles ML ──`)
- Utiliser le schéma nœud ci-dessous
- Respecter l'ordre lexicographique dans la section

Pour les **liens à ajouter** :
- Identifier les dépendances logiques (quel script importe quel module ?)
- Insérer dans le tableau `links` dans la section commentée appropriée
- Vérifier que `source` et `target` sont des IDs existants

Pour les **nœuds à supprimer** :
- Supprimer l'entrée du tableau `nodes`
- Supprimer tous les liens dont `source` ou `target` valent cet ID

---

## Schémas de référence

### Nœud

```js
{
  id: "snake_case_unique_id",       // dérivé du nom de fichier sans extension
  name: "Label court affiché",      // 2-4 mots, ce qui sera visible dans le diagramme
  type: "...",                       // voir Types ci-dessous
  category: "...",                   // voir Catégories ci-dessous
  experiment: "...",                 // voir Expériences ci-dessous
  importance: 0,                     // 1 = visible en mode Simplifié, 0 = Full seulement
  file: "chemin/relatif/depuis/root",
  description: "Ligne 1\nLigne 2\nLigne 3"  // \n pour les sauts de ligne
}
```

### Types de nœuds (`type`)

| Valeur | Couleur | Usage |
|--------|---------|-------|
| `data` | rouge `#ef4444` | Fichiers CSV bruts |
| `config` | gris `#64748b` | Fichiers YAML de configuration |
| `data_loader` | ambre `#f59e0b` | Loaders et feature engineering |
| `ml` | vert `#22c55e` | Classes de modèles ML/CL |
| `training` | bleu `#3b82f6` | Boucles d'entraînement, scénarios |
| `evaluation` | violet `#a855f7` | Métriques, profiler mémoire |
| `utils` | cyan `#06b6d4` | Helpers, quantization, reproducibility |
| `main` | orange `#f97316` | Scripts CLI (`scripts/*.py`) |
| `output` | teal `#14b8a6` | Répertoires de sorties, figures |

### Catégories (`category`)

| Valeur | Description |
|--------|-------------|
| `data_eda` | Exploration de données, visualisation |
| `memory` | Profiling RAM, contraintes embarquées |
| `eval_exp` | Évaluation des expériences CL |
| `plot` | Génération de figures |
| `ml_cl` | Modèles et entraînement CL supervisés |
| `ml_unsup` | Baselines non-supervisées |

### Expériences (`experiment`)

| Valeur | Description |
|--------|-------------|
| `single_task` | Baselines single-task (référence) |
| `industrial_pump` | Dataset 1 — Large Industrial Pump |
| `equipment_monitoring` | Dataset 2 — Industrial Equipment Monitoring |
| `all` | Commun aux deux datasets |

### Types de liens (`type`)

| Valeur | Couleur | Sémantique |
|--------|---------|-----------|
| `import` | bleu | Dépendance Python (`import`) |
| `reads` | rouge | Lecture de fichier / dataset |
| `output` | vert | Génère un fichier de sortie |
| `generates` | ambre | Génère des figures/artefacts |
| `uses_config` | gris | Utilise un fichier de configuration |

---

## Repères dans le HTML

- **Tableau `nodes`** : commence ligne ~571, se termine par `];` ligne ~819
- **Tableau `links`** : commence ligne ~821, se termine par `];` ~25 lignes plus bas
- Les sections du tableau `nodes` sont séparées par des commentaires `// ── Nom ──────`
- Nouvelles sections à créer si la catégorie n'existe pas encore

---

## Checklist avant de valider

- [ ] Chaque nouveau nœud a un `id` unique (pas de collision avec l'existant)
- [ ] Chaque nouveau lien a `source` ET `target` présents dans `nodes`
- [ ] Les `importance: 1` sont réservés aux nœuds structurants (datasets, scripts principaux, modèles clés)
- [ ] Le `file` correspond au chemin réel depuis la racine du dépôt
- [ ] Les `description` sont au format `"Ligne1\nLigne2"` (pas de backticks, pas de guillemets non échappés)
- [ ] Le fichier HTML s'ouvre correctement dans le navigateur après modification
