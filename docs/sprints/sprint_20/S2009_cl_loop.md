# S2009 — (Optionnel) Online CL loop : changement TASK_ID automatique

| Champ | Valeur |
|-------|--------|
| **Sprint** | 20 |
| **Priorité** | 🟢 Nice-to-have |
| **Statut** | ✅ Terminé |
| **Durée estimée** | 3h |
| **Dépendances** | S2002 (protocol v3), S2005 (exp EWC complète) |
| **Fichiers cibles** | `scripts/sensor_stream.py` |
| **Référence** | `docs/sprints/sprint_18/S1802_sensor_stream.md` |

---

## Contexte

`sensor_stream.py` envoie des frames avec un `TASK_ID` fixé manuellement.
Pour un vrai scénario CL domain-incremental, le `TASK_ID` doit changer automatiquement à chaque frontière de tâche, et déclencher `ewc_consolidate()` côté firmware via un flag dans les FLAGS du header UART.

---

## Ce qu'il faut ajouter

### Option `--cl-sequence` dans `sensor_stream.py`

```python
# Séquence CL : pump (task 0) → turbine (task 1) → compressor (task 2)
python scripts/sensor_stream.py \
    --port /dev/ttyACM0 \
    --config configs/board_ewc.yaml \
    --cl-sequence pump:167,turbine:167,compressor:166 \
    --consolidate-on-task-change  # envoie FLAGS=0x02 au dernier sample de chaque tâche
```

### Mécanisme côté firmware

Bit FLAGS dans le header UART v2/v3 :
- `FLAGS & 0x01` : demande de mise à jour (déjà défini)
- `FLAGS & 0x02` : frontière de tâche → appeler `ewc_consolidate()` après ce sample

```c
if (frame.flags & 0x02) {
    ewc_consolidate(&g_ewc, EWC_FISHER_DECAY);
    g_current_task_id = frame.task_id;
}
```

### Enregistrement automatique de l'expérience

À chaque fin de tâche, `sensor_stream.py` appelle `board_experiment_recorder.py` pour sauvegarder les métriques intermédiaires :
- `exp_S20_XX/task_0_metrics.json`, `task_1_metrics.json`, `task_2_metrics.json`
- Permet de tracer la courbe `acc_task_0` au fil des tâches (visualisation du forgetting)

---

## Vérification

- [ ] `--cl-sequence` respecte l'ordre et le nombre de samples par tâche
- [ ] Le flag `0x02` déclenche bien `ewc_consolidate()` côté firmware (test Unity `test_pipeline_consolidate_flag`)
- [ ] Les métriques intermédiaires sont sauvegardées dans `experiments/`
- [ ] Courbe acc_task_0 décroissante avec λ=0, stable avec λ=400

---

## Questions ouvertes

- `TODO(arnaud)` : Fréquence d'échantillonnage : 10 Hz (actuel) ou augmenter pour simuler un vrai capteur industriel ?
- `TODO(fred)` : Format des données Edge Spectrum compatible avec `--cl-sequence` ? (ADC continu vs frames UART)
