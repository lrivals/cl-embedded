# S12-11 — MAJ `roadmap_phase1.md` + table cross-dataset CWRU vs PRONOSTIA

| Champ | Valeur |
|-------|--------|
| **ID** | S12-11 |
| **Sprint** | Sprint 12 |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 1h |
| **Dépendances** | S12-05 à S12-09 (expériences + notebooks terminés) |
| **Fichier cible** | `docs/roadmap_phase1.md` |

---

## Objectif

Clôturer le sprint 12 dans la documentation de suivi : marquer S12-01 à S12-11 comme ✅ dans `roadmap_phase1.md` et ajouter la table comparative CWRU vs PRONOSTIA (et les autres datasets Phase 1) pour préparer la section de synthèse du manuscrit.

---

## Modifications à apporter à `roadmap_phase1.md`

### 1. Marquer S12-01 à S12-11 comme terminées

Dans la section sprint 12, passer chaque tâche de ⬜ à ✅.

### 2. Ajouter la table comparative cross-dataset (5 datasets Phase 1)

```markdown
## Synthèse cross-dataset — Phase 1

| Dataset | Type | N échantillons | N features | Scénario CL | Modèles | Exp | Statut |
|---------|------|---------------|-----------|-------------|---------|-----|--------|
| Equipment Monitoring | Tabulaire | ~10 000 | 4 | by_equipment (3 tâches) | EWC, HDC, TinyOL, KMeans, Maha, DBSCAN | exp_030–043 | ✅ |
| PRONOSTIA | Séries temporelles | ~87 000 | 8 | by_location (3 tâches) | EWC, HDC, TinyOL, KMeans, Maha, DBSCAN | exp_044–055 | ✅ |
| Battery RUL | Séries temporelles | ~variable | 5 | by_battery (N tâches) | EWC, HDC, TinyOL, KMeans, Maha, DBSCAN | exp_056–067 | ✅ |
| CWRU Bearing | Tabulaire (features stat.) | 2 299 | 9 | by_fault_type + by_severity | EWC, HDC, TinyOL, KMeans, Maha, DBSCAN | exp_068–085 | ✅ |
```

### 3. Ajouter la table comparative CWRU vs PRONOSTIA (métriques CL)

```markdown
## Comparaison CWRU vs PRONOSTIA — métriques CL

| Métrique | Scénario | EWC | HDC | TinyOL | KMeans | Maha | DBSCAN |
|---------|---------|-----|-----|--------|--------|------|--------|
| AA ↑ | PRONOSTIA by_location | — | — | — | — | — | — |
| AF ↓ | PRONOSTIA by_location | — | — | — | — | — | — |
| BWT | PRONOSTIA by_location | — | — | — | — | — | — |
| AA ↑ | CWRU by_fault_type | — | — | — | — | — | — |
| AF ↓ | CWRU by_fault_type | — | — | — | — | — | — |
| BWT | CWRU by_fault_type | — | — | — | — | — | — |
| AA ↑ | CWRU by_severity | — | — | — | — | — | — |
| AF ↓ | CWRU by_severity | — | — | — | — | — | — |
| BWT | CWRU by_severity | — | — | — | — | — | — |
```

> Les valeurs `—` sont à remplir depuis les `metrics_cl.json` des expériences correspondantes après exécution.

### 4. Mettre à jour `docs/context/datasets.md`

Ajouter une ligne CWRU dans la table des datasets :

```markdown
| CWRU Bearing | Tabulaire (9 features stat.) | 2 299 fenêtres | Binaire {0,1} | by_fault_type + by_severity | `data/raw/CWRU Bearing Dataset/` |
```

---

## Critères d'acceptation

- [ ] `roadmap_phase1.md` : S12-01 à S12-11 tous marqués ✅
- [ ] Table comparative 5 datasets présente dans `roadmap_phase1.md`
- [ ] Table CWRU vs PRONOSTIA (métriques CL) présente, même si les valeurs sont `—` en attente d'exécution
- [ ] `docs/context/datasets.md` : ligne CWRU ajoutée
- [ ] `FIXME(gap1)` dans les notebooks Phase 1 pointe vers exp_074–085 en complément de exp_050–055

## Statut

⬜ Non démarré
