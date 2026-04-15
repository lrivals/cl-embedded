# Plan d'implémentation — Outils HTML autonomes pour tout projet

> **Objectif :** Créer, dans n'importe quel projet, les deux outils HTML suivants à partir de zéro :
> - **`results_explorer.html`** — explorateur de fichiers interactif auto-contenu
> - **`pipeline_network_diagram.html`** — diagramme réseau D3.js de l'architecture du projet

---

## Contexte

Les deux fichiers sont des **outils HTML autonomes** (single-file, 0 dépendance locale) qui permettent de naviguer et comprendre un projet analytique/ML :

| Fichier | Taille | Génération | Bibliothèque |
|---|---|---|---|
| `results_explorer.html` | ~9 MB | Script Python (`update_results_explorer.py`) | Aucune (vanilla JS) |
| `pipeline_network_diagram.html` | ~80 KB | Fichier statique à écrire à la main | D3.js v7 (CDN) |

---

## Partie 1 — `results_explorer.html`

### Qu'est-ce que c'est ?

Un **explorateur de fichiers web** qui :
- Scanne récursivement des dossiers du projet (`Plots/`, `res/`, `data/`, etc.)
- Embarque les métadonnées et le contenu textuel (CSV, TXT ≤ 100 KB) dans un blob JSON injecté dans le HTML
- Affiche un arbre cliquable à gauche + un viewer à droite (images, CSV tabulé, texte brut)
- Fonctionnalités : recherche, navigation PREV/NEXT, zoom image, redimensionnement sidebar, breadcrumb, copie chemin

### Architecture du fichier généré

```
results_explorer.html
├── <head>  → CSS (dark theme, Inter + JetBrains Mono via Google Fonts)
├── <body>  → structure HTML fixe (header, sidebar, resize-handle, viewer-panel)
└── <script>
    ├── const resultTree = { ...JSON injecté... }   ← données
    ├── const statsData  = { ...JSON injecté... }   ← compteurs
    ├── buildTree() / renderTree()                  ← arbre sidebar
    ├── selectFile() / renderImageViewer()
    │   renderCsvViewer() / renderTextViewer()      ← viewer
    ├── setupImageInteraction()                     ← zoom/pan image
    ├── resize-handle drag                          ← sidebar resizable
    └── keyboard shortcuts (←/→/Esc)
```

### Structure du JSON embarqué

```json
{
  "Plots": {
    "icon": "🖼", "label": "Plots", "desc": "Visualisations",
    "children": {
      "Boxplot": {
        "icon": "📦", "label": "Boxplot", "desc": "Boxplots",
        "children": { ... },
        "fileList": [
          {
            "name": "boxplot_m5C.png",
            "path": "Plots/Boxplot/boxplot_m5C.png",
            "ext": "png",
            "size": 54321,
            "content": null
          }
        ],
        "files": "png",
        "count": 12
      }
    }
  },
  "res": { ... },
  "data": { ... }
}
```

### Étape 1 — Créer le script générateur Python

**Fichier :** `src/reporting/update_results_explorer.py`

```python
#!/usr/bin/env python3
"""Génère results_explorer.html à partir des fichiers réels du projet."""

import argparse, json, os, sys, time, webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_HTML  = PROJECT_ROOT / "results_explorer.html"

# ── À ADAPTER selon votre projet ──────────────────────────────────────────────
SCAN_ROOTS = ["Plots", "res", "data"]          # dossiers à scanner

SKIP_DIRS  = {"__pycache__", ".git", "venv", "node_modules"}
SKIP_FILES = {".gitignore", "results_explorer.html", "pipeline_network_diagram.html"}
SKIP_EXTS  = {".pyc", ".zip", ".html"}

TEXT_EMBED_MAX_BYTES = 102_400  # 100 KB — seuil pour embarquer CSV/TXT

# Règles d'enrichissement (icon + label) pour les dossiers reconnus
ENRICHMENT_RULES = [
    # (fragment_chemin, icone, description)
    ("random_forest",    "🌲", "Résultats Random Forest"),
    ("Boxplot",          "📦", "Boxplots"),
    ("metrics",          "📊", "Métriques"),
    ("data",             "📂", "Données source"),
    # ... ajouter selon votre projet
]
# ──────────────────────────────────────────────────────────────────────────────

def read_text_content(path: Path, max_bytes: int = TEXT_EMBED_MAX_BYTES):
    try:
        if path.stat().st_size > max_bytes:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

def scan_directory(base: Path, rel: Path, embed_text: bool = True) -> dict:
    full = base / rel
    node: dict = {}
    try:
        entries = sorted(os.listdir(full))
    except PermissionError:
        return node

    dirs  = [e for e in entries if (full / e).is_dir()  and e not in SKIP_DIRS and not e.startswith(".")]
    files = [e for e in entries if (full / e).is_file() and e not in SKIP_FILES
             and not e.startswith(".") and Path(e).suffix.lower() not in SKIP_EXTS]

    if dirs:
        node["children"] = {}
        for d in dirs:
            child = scan_directory(base, rel / d, embed_text=embed_text)
            if child or (base / rel / d).is_dir():
                node["children"][d] = child

    if files:
        file_list = []
        for f in files:
            fpath = rel / f
            fpath_full = base / fpath
            ext  = Path(f).suffix.lstrip(".").lower()
            size = 0
            try:
                size = fpath_full.stat().st_size
            except OSError:
                pass
            content = None
            if embed_text and ext in ("txt", "csv"):
                content = read_text_content(fpath_full)
            file_list.append({"name": f, "path": fpath.as_posix(),
                               "ext": ext, "size": size, "content": content})
        node["fileList"] = file_list
        node["files"]    = ",".join(sorted({e["ext"] for e in file_list if e["ext"]}))
        node["count"]    = len(file_list)
    return node

def enrich_node(rel_path: Path, node: dict) -> dict:
    parts_str  = "/".join(rel_path.parts)
    icon, desc = "📁", rel_path.name
    for fragment, rule_icon, rule_desc in ENRICHMENT_RULES:
        if fragment.lower() in parts_str.lower():
            icon, desc = rule_icon, rule_desc
            break
    node["icon"]  = icon
    node["label"] = rel_path.name
    node["desc"]  = desc
    if "children" in node:
        for name, child in node["children"].items():
            enrich_node(rel_path / name, child)
    return node

def build_result_tree(project_root: Path, embed_text: bool = True) -> dict:
    tree = {}
    for root_name in SCAN_ROOTS:
        root_path = project_root / root_name
        if not root_path.exists():
            continue
        node = scan_directory(project_root, Path(root_name), embed_text=embed_text)
        enrich_node(Path(root_name), node)
        tree[root_name] = node
    return tree

def count_stats(tree: dict) -> dict:
    stats = {"png": 0, "csv": 0, "txt": 0, "other": 0, "total": 0}
    def walk(node):
        for entry in node.get("fileList", []):
            ext = entry.get("ext", "")
            stats[ext if ext in stats else "other"] += 1
            stats["total"] += 1
        for child in node.get("children", {}).values():
            walk(child)
    for n in tree.values():
        walk(n)
    return stats

def generate_html(tree: dict, stats: dict) -> str:
    tree_json  = json.dumps(tree,  ensure_ascii=False, separators=(",", ":"))
    stats_json = json.dumps(stats, ensure_ascii=False, separators=(",", ":"))
    return HTML_TEMPLATE \
        .replace("/* __RESULT_TREE_DATA__ */null", tree_json) \
        .replace("/* __STATS_DATA__ */null",       stats_json)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--serve",    action="store_true")
    parser.add_argument("--watch",    action="store_true")
    parser.add_argument("--open",     action="store_true")
    parser.add_argument("--port",     type=int, default=8080)
    parser.add_argument("--output",   type=Path, default=None)
    args = parser.parse_args()

    output = args.output or OUTPUT_HTML

    tree  = build_result_tree(PROJECT_ROOT)
    stats = count_stats(tree)

    if args.dry_run:
        print(json.dumps({"tree": tree, "stats": stats}, ensure_ascii=False, indent=2))
        return

    html = generate_html(tree, stats)
    output.write_text(html, encoding="utf-8")
    print(f"[OK] {output}  ({len(html) // 1024} KB)")

    if args.serve:
        # ... (http.server.HTTPServer sur PROJECT_ROOT, port args.port)
        pass
    if args.watch:
        # ... (polling os.walk + re-génération si mtime change)
        pass

if __name__ == "__main__":
    main()
```

### Étape 2 — Créer le template HTML

Le script a besoin d'une variable `HTML_TEMPLATE` (string Python) contenant le fichier HTML avec deux placeholders :

```python
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MON PROJET — Results Explorer</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    /* ... CSS complet (voir section Design ci-dessous) ... */
  </style>
</head>
<body>
  <div class="header">
    <div class="header-title">MON PROJET <span>— Results Explorer</span></div>
    <div class="stats-bar" id="statsBar"></div>
  </div>
  <div class="layout">
    <div class="sidebar" id="sidebar">
      <div class="sidebar-top">
        <div class="search-container">
          <span class="search-icon">🔍</span>
          <input class="search-input" id="searchInput" placeholder="Rechercher un fichier...">
        </div>
      </div>
      <div class="sidebar-actions">
        <button class="action-btn" onclick="collapseAll()">Réduire tout</button>
        <button class="action-btn" onclick="expandToLevel(1)">Développer</button>
      </div>
      <div class="tree-root" id="treeRoot"></div>
    </div>
    <div class="resize-handle" id="resizeHandle"></div>
    <div class="viewer-panel" id="viewerPanel">
      <div class="viewer-placeholder">
        <div class="placeholder-icon">📂</div>
        <div class="placeholder-text">Sélectionnez un fichier</div>
      </div>
    </div>
  </div>

  <script>
    const resultTree = /* __RESULT_TREE_DATA__ */null;
    const statsData  = /* __STATS_DATA__ */null;
    // ... JS complet (voir section Logique JS ci-dessous) ...
  </script>
</body>
</html>"""
```

> **Note :** Les deux placeholders `/* __RESULT_TREE_DATA__ */null` et `/* __STATS_DATA__ */null` sont remplacés par le script Python avec les JSON réels.

### Étape 3 — CSS du template (dark theme)

Points clés du design :

```css
:root {
  --bg-main: #0a0a0a;
  --bg-sidebar: rgba(20,20,20,0.8);
  --bg-viewer: #050505;
  --glass-bg: rgba(40,40,40,0.4);
  --glass-border: rgba(255,255,255,0.12);
  --text-main: #d4d4d4;
  --text-dim: #737373;
  --text-bright: #ffffff;
  --sidebar-width: 320px;
}
body {
  font-family: 'Inter', system-ui, sans-serif;
  background-color: var(--bg-main);
  color: var(--text-main);
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.layout { flex: 1; display: flex; overflow: hidden; }
.sidebar {
  width: var(--sidebar-width);
  min-width: 260px; max-width: 50%;
  backdrop-filter: blur(8px);
  background: var(--bg-sidebar);
  border-right: 1px solid var(--glass-border);
  display: flex; flex-direction: column;
}
.resize-handle { width: 4px; cursor: col-resize; background: transparent; }
.viewer-panel { flex: 1; overflow: hidden; display: flex; flex-direction: column; }
/* Tree */
.tree-file.active { background: rgba(255,255,255,0.1); color: var(--text-bright); }
.tree-file.hidden { display: none; }
/* Viewer */
.img-container { overflow: hidden; display: flex; justify-content: center; align-items: center; height: 100%; }
.img-container img { max-width: 100%; cursor: grab; transform-origin: center center; }
.csv-table { border-collapse: collapse; font-size: 0.78rem; }
.csv-table th, .csv-table td { border: 1px solid var(--glass-border); padding: 6px 10px; }
.csv-table th { background: rgba(255,255,255,0.06); color: var(--text-bright); }
.text-box { padding: 24px; white-space: pre-wrap; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }
```

### Étape 4 — Logique JavaScript

Fonctions clés à implémenter :

| Fonction | Rôle |
|---|---|
| `renderStats()` | Affiche les compteurs (N images, N CSV...) dans le header |
| `buildTree(name, node, depth)` | Construit récursivement les div `.tree-dir` / `.tree-file` |
| `renderTree()` | Vide le `#treeRoot` et appelle `buildTree` sur chaque clé de `resultTree` |
| `selectFile(file, siblings)` | Met à jour l'état et appelle le bon viewer |
| `renderImageViewer(file)` | Inject `<img src="file.path">` + toolbar dans `#viewerPanel` |
| `renderCsvViewer(file)` | Parse `file.content` (`;` ou `,`) → tableau HTML |
| `renderTextViewer(file)` | Affiche `file.content` dans `<pre>` |
| `navFile(dir)` | Navigation PREV (-1) / NEXT (+1) parmi les fichiers du même type |
| `setupImageInteraction()` | `wheel` = zoom, `mousedown/move/up` = pan |
| `collapseAll() / expandToLevel(n)` | Toggle classes `.open` sur `.tree-children` |
| Resize handle | `mousedown` sur `#resizeHandle` → drag → `sidebar.style.width = w + 'px'` |
| Keyboard | `←/→` = navFile, `Esc` = collapseAll |

### Étape 5 — Lancement

```bash
# Génération simple
python src/reporting/update_results_explorer.py

# Ouvrir dans le navigateur
python -m http.server 8080
# puis aller sur http://localhost:8080/results_explorer.html

# Ou avec le mode serve intégré
python src/reporting/update_results_explorer.py --serve --open

# Auto-régénération si les fichiers changent
python src/reporting/update_results_explorer.py --watch
```

### Points d'adaptation pour un nouveau projet

| Variable | Valeur à changer |
|---|---|
| `SCAN_ROOTS` | Dossiers à scanner (`["output", "reports", "figures"]`) |
| `SKIP_DIRS` | Dossiers à ignorer |
| `SKIP_EXTS` | Extensions à ignorer |
| `ENRICHMENT_RULES` | Icônes et labels pour vos dossiers |
| `TEXT_EMBED_MAX_BYTES` | Taille max pour embarquer du texte (défaut 100 KB) |
| Titre dans `HTML_TEMPLATE` | Nom du projet |

---

## Partie 2 — `pipeline_network_diagram.html`

### Qu'est-ce que c'est ?

Un **diagramme réseau interactif** basé sur D3.js v7 (force simulation) qui visualise :
- Les **nœuds** : scripts, fichiers de données, outputs, apps
- Les **liens** : imports, lecture de données, écriture de sorties, lancement de sous-processus
- **Filtrage** par catégorie, deux modes de vue (simplifié / complet), tooltip et panneau d'info au clic

Ce fichier est **entièrement statique** — il se rédige à la main une fois l'architecture connue.

### Architecture du fichier

```
pipeline_network_diagram.html
├── <head>
│   ├── D3.js v7 via CDN (cdnjs.cloudflare.com)
│   └── CSS : dark theme navy (#0f172a), layout fixe
├── <body>
│   ├── <h1> titre
│   ├── .controls  → boutons de filtre + toggle vue
│   ├── .legend    → légende nœuds + liens
│   ├── .info-panel → panneau détail au clic
│   ├── <svg id="network"> → rendu D3
│   └── .tooltip   → tooltip au survol
└── <script>
    ├── const nodes = [ ... ]   ← définition des nœuds
    ├── const links = [ ... ]   ← définition des liens
    ├── D3 force simulation
    ├── filterByType(type)
    ├── setViewMode('simplified'|'full')
    ├── resetView() / toggleLabels()
    └── gestion tooltip + info-panel
```

### Étape 1 — Palette de couleurs et types

```javascript
// Couleurs par type de nœud
const typeColors = {
  data:         "#ef4444",   // rouge    — fichiers de données
  main:         "#f59e0b",   // orange   — scripts CLI / entry points
  ml:           "#22c55e",   // vert     — modules ML core
  utils:        "#3b82f6",   // bleu     — utilitaires
  analysis:     "#a855f7",   // violet   — scripts d'analyse
  preprocessing:"#a855f7",   // violet   — preprocessing
  cross_dataset:"#8b5cf6",   // violet clair
  reporting:    "#f97316",   // orange foncé
  app:          "#06b6d4",   // cyan     — apps interactives
  output:       "#14b8a6",   // teal     — dossiers de sortie
};

// Couleurs par type de lien
const linkColors = {
  import:  "#3b82f6",   // bleu   — imports Python
  reads:   "#ef4444",   // rouge  — lecture de données
  output:  "#22c55e",   // vert   — écriture de sorties
  launches:"#f59e0b",   // orange — lancement de sous-processus
};
```

### Étape 2 — Définir les nœuds

Chaque nœud est un objet JavaScript :

```javascript
const nodes = [
  // Fichiers de données
  {
    id: "data_main",
    name: "data.csv",           // label affiché
    type: "data",               // → couleur via typeColors
    category: "data",           // → filtre par bouton
    importance: 1,              // 1 = nœud principal (rayon plus grand)
    file: "data/data.csv",      // chemin affiché dans le tooltip
    description: "Dataset principal\n240 échantillons, 39 features"
  },

  // Scripts ML
  {
    id: "main_classifier",
    name: "main_classifier.py",
    type: "main",
    category: "ml",
    importance: 1,
    file: "src/ml/main_classifier.py",
    description: "Point d'entrée classification\n9 classifieurs supportés"
  },

  // Dossiers de sortie
  {
    id: "out_results",
    name: "res/",
    type: "output",
    category: "output",
    file: "res/",
    description: "Résultats de classification"
  },
  // ...
];
```

### Étape 3 — Définir les liens

```javascript
const links = [
  // { source: id_nœud, target: id_nœud, type: "import"|"reads"|"output"|"launches" }
  { source: "main_classifier",  target: "ml_pipeline",  type: "import"  },
  { source: "main_classifier",  target: "data_main",    type: "reads"   },
  { source: "ml_pipeline",      target: "out_results",  type: "output"  },
  { source: "rf_launcher",      target: "rf_tree_viewer", type: "launches" },
  // ...
];
```

### Étape 4 — Mise en place de D3 force simulation

```javascript
const width  = window.innerWidth;
const height = window.innerHeight - 60;

const svg = d3.select("#network").attr("width", width).attr("height", height);
const g   = svg.append("g");  // groupe transformable (zoom)

// Zoom pan
const zoom = d3.zoom().scaleExtent([0.2, 5])
  .on("zoom", e => g.attr("transform", e.transform));
svg.call(zoom);

// Simulation de forces
const simulation = d3.forceSimulation(nodes)
  .force("link",    d3.forceLink(links).id(d => d.id).distance(120).strength(0.5))
  .force("charge",  d3.forceManyBody().strength(-400))
  .force("center",  d3.forceCenter(width / 2, height / 2))
  .force("collision", d3.forceCollide().radius(d => nodeRadius(d) + 10));

// Rendu des liens
const link = g.selectAll(".link")
  .data(links).enter().append("path")
  .attr("class", "link")
  .attr("stroke", d => linkColors[d.type])
  .attr("stroke-width", 1.5);

// Rendu des nœuds (groupes <g>)
const node = g.selectAll(".node")
  .data(nodes).enter().append("g")
  .attr("class", "node")
  .call(d3.drag()
    .on("start", dragstarted)
    .on("drag",  dragged)
    .on("end",   dragended));

node.append("circle")
  .attr("r",    d => nodeRadius(d))
  .attr("fill", d => typeColors[d.type] || "#64748b");

node.append("text")
  .attr("class", "node-label")
  .attr("dy", d => nodeRadius(d) + 12)
  .text(d => d.name);

// Tick
simulation.on("tick", () => {
  link.attr("d", d => `M${d.source.x},${d.source.y} L${d.target.x},${d.target.y}`);
  node.attr("transform", d => `translate(${d.x},${d.y})`);
});

function nodeRadius(d) {
  return d.importance === 1 ? 20 : 14;
}
```

### Étape 5 — Filtre par catégorie et mode de vue

```javascript
// Mode simplifié : masquer les nœuds non-importance et leurs liens
let currentMode = "simplified";
let showLabels  = true;

function setViewMode(mode) {
  currentMode = mode;
  // Toggle visibilité des nœuds selon mode
  node.style("opacity", d => {
    if (mode === "simplified" && !d.importance) return 0.15;
    return 1;
  });
}

function filterByType(type) {
  node.style("opacity", d => {
    if (type === "all") return 1;
    return d.category === type ? 1 : 0.1;
  });
  link.style("opacity", d => {
    if (type === "all") return 0.4;
    return (d.source.category === type || d.target.category === type) ? 0.8 : 0.05;
  });
}

function toggleLabels() {
  showLabels = !showLabels;
  d3.selectAll(".node-label").style("opacity", showLabels ? 1 : 0);
}

function resetView() {
  svg.transition().duration(500)
    .call(zoom.transform, d3.zoomIdentity);
}
```

### Étape 6 — Tooltip et panneau info

```javascript
// Tooltip (au survol)
const tooltip = d3.select("#tooltip");

node.on("mouseover", (event, d) => {
  tooltip.style("opacity", 1)
    .style("left", (event.clientX + 16) + "px")
    .style("top",  (event.clientY - 10) + "px")
    .html(`<strong>${d.name}</strong>
           <span class="tt-type">${d.type}</span>
           <span class="tt-file">${d.file || ""}</span>`);
})
.on("mousemove", (event) => {
  tooltip.style("left", (event.clientX + 16) + "px")
         .style("top",  (event.clientY - 10) + "px");
})
.on("mouseout",  () => tooltip.style("opacity", 0));

// Panneau info (au clic)
node.on("click", (event, d) => {
  event.stopPropagation();
  document.getElementById("infoTitle").textContent = d.name;
  document.getElementById("infoFile").textContent  = d.file || "";
  document.getElementById("infoDesc").textContent  = d.description || "";
  // Calculer les connexions entrantes/sortantes
  const incoming = links.filter(l => l.target.id === d.id);
  const outgoing  = links.filter(l => l.source.id === d.id);
  document.getElementById("infoConnections").innerHTML =
    `Entrants : <span>${incoming.length}</span>  |  Sortants : <span>${outgoing.length}</span>`;
  document.getElementById("infoPanel").classList.add("visible");
});

function closeInfo() {
  document.getElementById("infoPanel").classList.remove("visible");
}
svg.on("click", () => closeInfo());
```

### Étape 7 — CSS du pipeline diagram

Points clés :

```css
body {
  background: #0f172a;
  color: #e2e8f0;
  font-family: 'Segoe UI', system-ui, sans-serif;
  overflow: hidden;
}
svg {
  display: block;
  width: 100vw;
  height: calc(100vh - 60px);
  background: radial-gradient(ellipse at center, #1e293b 0%, #0f172a 70%);
}
.controls, .legend, .info-panel {
  position: fixed;
  background: rgba(15, 23, 42, 0.92);
  border: 1px solid #334155;
  border-radius: 12px;
  backdrop-filter: blur(8px);
  z-index: 50;
}
.controls { top: 70px; left: 16px; padding: 14px; }
.legend   { bottom: 16px; right: 16px; padding: 14px 18px; }
.info-panel { top: 70px; right: 16px; max-width: 340px; padding: 18px;
              opacity: 0; transition: opacity 0.25s; border-color: #38bdf8; }
.info-panel.visible { opacity: 1; }
.tooltip {
  position: fixed;
  background: rgba(15,23,42,0.96);
  border: 1px solid #38bdf8;
  border-radius: 10px;
  padding: 14px 18px;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.2s;
  max-width: 360px;
  z-index: 100;
}
.btn.active { background: #38bdf8; color: #0f172a; font-weight: 600; }
```

---

## Récapitulatif — Checklist d'implémentation

### Pour `results_explorer.html`

- [ ] Créer `src/reporting/update_results_explorer.py` avec les constantes adaptées au projet
- [ ] Écrire la variable `HTML_TEMPLATE` dans ce script (HTML + CSS + JS)
- [ ] Implémenter `scan_directory()`, `enrich_node()`, `build_result_tree()`, `count_stats()`, `generate_html()`
- [ ] Implémenter le JS : `buildTree()`, `selectFile()`, les 3 viewers, zoom/pan, resize, keyboard
- [ ] Tester : `python src/reporting/update_results_explorer.py && python -m http.server 8080`

### Pour `pipeline_network_diagram.html`

- [ ] Lister tous les scripts/modules/données du projet → liste de nœuds
- [ ] Identifier les dépendances entre modules → liste de liens
- [ ] Choisir une palette de couleurs par type de nœud
- [ ] Créer le HTML avec D3.js CDN + CSS dark theme
- [ ] Implémenter la force simulation avec drag, zoom, tooltip, info-panel
- [ ] Ajouter les contrôles : filtre par catégorie, mode simplifié/complet, toggle labels
- [ ] Tester en ouvrant directement dans le navigateur (fichier statique, pas de serveur requis)

---

## Commandes de vérification

```bash
# Vérifier que results_explorer.html est généré et non vide
python src/reporting/update_results_explorer.py
ls -lh results_explorer.html

# Servir les deux fichiers localement
python -m http.server 8080
# → http://localhost:8080/results_explorer.html
# → http://localhost:8080/pipeline_network_diagram.html

# Pour le pipeline diagram : ouvrir directement (fichier statique)
xdg-open pipeline_network_diagram.html    # Linux
open pipeline_network_diagram.html        # macOS
```
