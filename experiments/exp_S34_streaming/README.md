# exp_S34_streaming — Balayage débit/buffer board (Sprint 34 S3403)

Board réelle **NUCLEO-F439ZI** (`/dev/ttyACM0`), dataset CWRU (5 features), protocole v3.

## Contenu

- `mahalanobis_rate{R}_stride{S}_w5.json` / `ewc_rate{R}_stride{S}_w5.json` — une config par
  fichier : latence DWT, drops, timeouts, erreurs CRC, débit analytique (S3401), saturation.
- `summary_mahalanobis_w5.json` / `summary_ewc_w5.json` — agrégats par modèle.
- `bss_by_window.json` — `.bss` mesuré (`arm-none-eabi-size`) par taille de fenêtre W
  (rebuild `make all STREAM_BUF_W=W`), W ∈ {5, 10, 25, 50}.

## Résultats clés (mesurés, non inventés)

- **Latence DWT** : Mahalanobis **5 µs**, EWC **50 µs** — invariantes au rate/stride,
  **≪ 100 ms (Gap 2 ✅)** sur toute la plage 50–5000 Hz.
- **0 drop / 0 timeout / 0 erreur CRC** sur toutes les configs.
- **Pas de saturation par dépassement de buffer** : le protocole UART est **synchrone
  requête/réponse** → le PC attend chaque réponse avant d'émettre la suivante, donc il ne
  peut pas saturer la board (auto-throttling). La borne réelle est le **round-trip**
  (UART + inférence), pas l'overrun board. Le `debit_max = 1/latence_inf` (S3401) reste la
  borne théorique : Maha 200 000 Hz, EWC 20 000 Hz, tous deux ≫ `debit_streaming` testé
  (≤ 5000 Hz) → marge temps-réel positive partout.
- **`.bss` du buffer de streaming** = `W × PROTO_MAX_N(16) × 4 B` : 320 B (W=5) → 3200 B
  (W=50), linéaire et borné, négligeable devant la SRAM (0 malloc).

> Le multi-stream concurrent (plusieurs flux simultanés) reste une étude analytique
> (cf. notebook `notebooks/cl_eval/streaming/comparison.ipynb`, `TODO(arnaud)`).
