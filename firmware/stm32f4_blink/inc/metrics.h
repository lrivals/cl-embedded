#pragma once
#include <stdint.h>

/* ── Métriques CL on-board : accuracy, AUROC approximé, forgetting ─────────
 *
 * Toutes les structures sont statiques (pas de malloc).
 * Compatibles streaming (O(1) ou fenêtre bornée).
 *
 * Référence : src/evaluation/online_metrics.py (équivalent Python)
 */

/* ── Accuracy online ────────────────────────────────────────────────────── */

/* MEM: 8 B @ uint32 en .bss */
typedef struct {
    uint32_t n_correct;
    uint32_t n_total;
} OnlineAccuracy;

void    acc_init(OnlineAccuracy *a);
void    acc_update(OnlineAccuracy *a, int pred, int true_label);
float   acc_compute(const OnlineAccuracy *a);   /* retourne 0.0 si n_total==0 */

/* ── AUROC approché (fenêtre glissante bornée) ──────────────────────────── */

#define AUROC_WINDOW 50U   /* MEM: 50×(1B+4B) = 250 B @ FP32+u8 en .bss */

/* MEM: 258 B @ FP32+uint8 en .bss */
typedef struct {
    float    scores[AUROC_WINDOW];   /* MEM: 200 B @ FP32 */
    uint8_t  labels[AUROC_WINDOW];   /* MEM:  50 B @ uint8 */
    uint32_t head;                   /* Index écriture (circulaire) */
    uint32_t count;                  /* Éléments valides (≤ AUROC_WINDOW) */
} OnlineAUROC;

void  auroc_init(OnlineAUROC *a);
void  auroc_update(OnlineAUROC *a, float score, int label);
float auroc_compute(const OnlineAUROC *a);   /* Wilcoxon-Mann-Whitney, O(W²) */

/* ── Forgetting tracker (par tâche, max 4 tâches) ──────────────────────── */

#define MAX_TASKS 4U

/* MEM: 36 B @ FP32+uint8 en .bss */
typedef struct {
    float    peak_acc[MAX_TASKS];     /* MEM: 16 B @ FP32 — meilleure accuracy observée par tâche */
    float    current_acc[MAX_TASKS];  /* MEM: 16 B @ FP32 — accuracy courante par tâche */
    uint8_t  seen[MAX_TASKS];         /* MEM:  4 B @ uint8 — tâche déjà observée */
} ForgettingTracker;

void  fgt_init(ForgettingTracker *f);
void  fgt_update(ForgettingTracker *f, uint8_t task_id, float acc);
float fgt_avg_forgetting(const ForgettingTracker *f);   /* AF = mean(peak - current) */
float fgt_backward_transfer(const ForgettingTracker *f);

/* ── Snapshot compact pour transmission UART ────────────────────────────── */

/* MEM: 12 B en .bss (3 × float) */
typedef struct {
    float accuracy;    /* accuracy courante globale */
    float auroc;       /* AUROC fenêtre glissante */
    float forgetting;  /* AF moyen */
} MetricsSnapshot;

/* Encode [acc:f32][auroc:f32][forgetting:f32] = 12 B dans buf */
void metrics_encode_snapshot(const MetricsSnapshot *s, uint8_t *buf);
#define METRICS_SNAPSHOT_SIZE 12U
