"""
Génération des figures SCHÉMATIQUES pour la présentation « Portage C & pipeline
expérimental board » (NUCLEO-F439ZI, Sprints 16-31).

Contrairement à scripts/generate_presentation_plots.py (orienté résultats), ce
script produit uniquement des SCHÉMAS d'architecture et de flux de données :
architecture firmware, trame UART octet par octet, dispatch par nibble, flux de
données bout-en-bout, workflow d'export des poids, et mécanique de profiling.

Toutes les valeurs (offsets d'octets, codes de mode, tailles de réponse) sont
calquées sur firmware/stm32f4_blink/inc/pipeline.h et scripts/sensor_stream.py.

Usage:
    python scripts/generate_portage_plots.py
    python scripts/generate_portage_plots.py --output docs/figures/presentation_board/
    python scripts/generate_portage_plots.py --show
"""

import argparse
import os

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "figure.dpi": 150,
})

COLORS = {
    "mahal": "#2196F3",
    "ewc": "#4CAF50",
    "tinyol": "#FF9800",
    "hdc": "#9C27B0",
    "meta": "#00897B",
    "budget": "#F44336",
    "board": "#1565C0",
    "pc": "#546E7A",
    "flash": "#8D6E63",
    "bss": "#5C6BC0",
    "stack": "#26A69A",
    "ok": "#43A047",
    "warn": "#FFA000",
    "bad": "#E53935",
    "header": "#37474F",
    "feat": "#1E88E5",
    "tail": "#6D4C41",
}


def save_or_show(fig, name: str, out_dir, show: bool):
    if show:
        plt.show()
    if out_dir:
        path = os.path.join(out_dir, f"{name}.png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  ✅ {path}")
    plt.close(fig)


def _box(ax, x, y, w, h, text, color, fontsize=10, text_color="white",
         alpha=1.0, fontweight="bold", round_pad=0.02):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad={round_pad}",
                         facecolor=color, edgecolor="white",
                         linewidth=1.2, alpha=alpha, zorder=3)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=text_color, fontsize=fontsize, fontweight=fontweight,
            zorder=4)
    return box


def _arrow(ax, x0, y0, x1, y1, color="#37474F", lw=1.8, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw, zorder=2))


# ---------------------------------------------------------------------------
# 1. Architecture firmware — pipeline.c routeur + têtes modèles + mémoire
# ---------------------------------------------------------------------------
def plot_firmware_arch(out_dir, show):
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("Architecture firmware NUCLEO-F439ZI — pipeline.c orchestre, têtes modèles exécutent",
                 fontsize=13, fontweight="bold")

    # Entrée UART
    _box(ax, 0.3, 6.6, 2.1, 1.0, "UART RX\n(USART3 115200)", COLORS["pc"], fontsize=9)
    # Orchestrateur
    _box(ax, 3.2, 6.4, 3.0, 1.4, "pipeline.c\nrouteur (FLAGS → mode)\nRX·CRC·norm·TX", COLORS["board"], fontsize=10)
    _arrow(ax, 2.4, 7.1, 3.2, 7.1)
    # Sortie UART
    _box(ax, 7.0, 6.6, 2.1, 1.0, "UART TX\nréponse 14–27 B", COLORS["pc"], fontsize=9)
    _arrow(ax, 6.2, 7.1, 7.0, 7.1)
    # profiling
    _box(ax, 9.6, 6.6, 2.1, 1.0, "profiling.c\nDWT + .bss", COLORS["ok"], fontsize=9)
    _arrow(ax, 4.7, 6.4, 4.7, 5.7)

    # Têtes modèles (rangée)
    heads = [
        ("mahalanobis.c\nEMA-Welford", COLORS["mahal"]),
        ("ewc_head*.c\nMLP + Fisher SGD", COLORS["ewc"]),
        ("hdc.c\naccum. ±1", COLORS["hdc"]),
        ("tinyol.c\nautoencodeur", COLORS["tinyol"]),
        ("meta_head.c\nstacking", COLORS["meta"]),
    ]
    x = 0.4
    for name, color in heads:
        _box(ax, x, 4.3, 2.05, 1.2, name, color, fontsize=8.5)
        _arrow(ax, 4.7, 5.7, x + 1.0, 5.5, color="#90A4AE", lw=1.2)
        x += 2.3

    ax.text(6.0, 3.85, "INT8 : ewc_head_int8.c · hdc_int8.c · tinyol_int8.c (mêmes signatures)",
            ha="center", fontsize=8.5, style="italic", color="#455A64")

    # Zones mémoire
    _box(ax, 0.4, 1.7, 3.4, 1.6,
         "Flash (.rodata)\nimmuable\n• poids *_INIT\n• Z-score, proj. HDC", COLORS["flash"], fontsize=9)
    _box(ax, 4.2, 1.7, 3.4, 1.6,
         ".bss (SRAM)\nmutable — zéro malloc\n• poids vivants (SGD)\n• Fisher, métriques", COLORS["bss"], fontsize=9)
    _box(ax, 8.0, 1.7, 3.4, 1.6,
         "Stack\ntemporaires forward\n• h1[32], h2[16]\n• hv[D]…", COLORS["stack"], fontsize=9)
    ax.text(6.0, 1.25, "memcpy Flash → .bss au boot (pipeline_init) : les poids doivent être modifiables pour l'update en ligne",
            ha="center", fontsize=8.5, color="#455A64")

    ax.text(6.0, 0.5, "Allocation 100 % statique · tailles via #define · bare-metal (pas d'OS)",
            ha="center", fontsize=9, fontweight="bold", color=COLORS["board"])

    save_or_show(fig, "portage_01_firmware_arch", out_dir, show)


# ---------------------------------------------------------------------------
# 2. Trame UART requête octet par octet + tailles de réponse
# ---------------------------------------------------------------------------
def plot_uart_frame(out_dir, show):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 33)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Trame requête PC → carte (32 B pour N=5) — ordre réel de sensor_stream.py",
                 fontsize=13, fontweight="bold")

    # Champs : (label court, label dessous, n_octets, couleur)
    fields = [
        ("MAGIC", "AB CD", 2, COLORS["header"]),
        ("VER", "02/03", 1, COLORS["header"]),
        ("TASK", "id", 1, COLORS["header"]),
        ("TIMESTAMP", "u32 LE", 4, COLORS["header"]),
        ("N", "=5", 1, COLORS["header"]),
        ("features  f32 × N  (little-endian, déjà normalisées Z-score)", "", 20, COLORS["feat"]),
        ("L", "lbl", 1, COLORS["tail"]),
        ("F", "flg", 1, COLORS["bad"]),
        ("C", "crc", 1, COLORS["tail"]),
    ]
    x = 0.5
    scale = 0.95
    offset = 0
    for top, bot, n, color in fields:
        w = n * scale
        label = f"{top}\n{bot}" if bot else top
        _box(ax, x, 5.4, w, 1.7, label, color,
             fontsize=8.5 if n > 2 else 7.5, round_pad=0.01)
        ax.text(x, 5.1, f"+{offset}", ha="left", va="top", fontsize=7, color="#607D8B")
        x += w + 0.12
        offset += n
    ax.text(x - 0.05, 6.25, f"= {offset} B", ha="left", va="center", fontsize=9.5,
            fontweight="bold", color=COLORS["board"])

    ax.text(16.5, 8.0, "header  struct.pack(\"<HBBIB\", …) = 9 B   +   features 4·N   +   label·flags 2 B   +   CRC8 1 B",
            ha="center", fontsize=8.5, color="#455A64")

    # Tailles de réponse
    ax.text(0.5, 4.3, "Réponses (la longueur identifie le mode — désambiguïsée par sensor_stream.py) :",
            fontsize=10, fontweight="bold")
    resp = [
        ("v2 — 14 B", "<BfIHHB", COLORS["pc"]),
        ("v3 — 23 B", "<BfIHfff  (+ACC/AUROC/FORGET)", COLORS["board"]),
        ("DUAL — 25 B", "<BffIfff  (RUL + faute)", COLORS["ewc"]),
        ("PAIR — 22 B", "<BfBfIff  (Maha + supervisé)", COLORS["mahal"]),
        ("TRIPLE — 27 B", "<BfBfIffBf  (PAIR + méta)", COLORS["meta"]),
    ]
    y = 3.5
    for name, fmt, color in resp:
        _box(ax, 0.5, y, 4.0, 0.55, name, color, fontsize=9)
        ax.text(5.0, y + 0.28, fmt, va="center", fontsize=9, family="monospace", color="#37474F")
        y -= 0.68

    save_or_show(fig, "portage_02_uart_frame", out_dir, show)


# ---------------------------------------------------------------------------
# 3. Dispatch par nibble — table des modes + ordre d'évaluation
# ---------------------------------------------------------------------------
def plot_mode_dispatch(out_dir, show):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("FLAGS = sélecteur de mode — nibble bas (action) + nibble haut (modèle/mode)",
                 fontsize=13, fontweight="bold")

    # Nibble bas
    ax.text(0.3, 9.3, "Nibble bas — actions (combinables)", fontsize=10.5, fontweight="bold", color=COLORS["board"])
    low = [
        ("0x01  UPDATE", "1 pas SGD avec ce sample"),
        ("0x04  CONSOLIDATE", "frontière de tâche → Fisher / binarize"),
        ("0x08  RESET", "ré-init poids + métriques"),
        ("0x02  PROFILING", "inclut métriques DWT"),
    ]
    y = 8.7
    for code, desc in low:
        _box(ax, 0.3, y, 2.9, 0.55, code, COLORS["bad"], fontsize=8.5)
        ax.text(3.4, y + 0.27, desc, va="center", fontsize=8.5, color="#37474F")
        y -= 0.7

    # Nibble haut
    ax.text(0.3, 5.5, "Nibble haut — mode (valeur unique, exact-match)", fontsize=10.5, fontweight="bold", color=COLORS["board"])
    modes = [
        ("0x10", "EWC binaire", COLORS["ewc"]),
        ("0x20", "HDC", COLORS["hdc"]),
        ("0x40", "EWC INT8", COLORS["ewc"]),
        ("0x80", "TinyOL", COLORS["tinyol"]),
        ("0x30", "EWC multiclasse", COLORS["ewc"]),
        ("0x50", "EWC régression (RUL)", COLORS["ewc"]),
        ("0x60", "HDC INT8", COLORS["hdc"]),
        ("0x70", "DUAL (RUL+faute)", COLORS["board"]),
        ("0xC0", "TinyOL INT8", COLORS["tinyol"]),
        ("0x90", "PAIR Maha+EWC", COLORS["mahal"]),
        ("0xA0", "PAIR Maha+HDC", COLORS["mahal"]),
        ("0xB0", "PAIR Maha+TinyOL", COLORS["mahal"]),
        ("0xD0", "TRIPLE Maha+EWC+méta", COLORS["meta"]),
        ("0xE0", "TRIPLE Maha+HDC+méta", COLORS["meta"]),
    ]
    x0, y0 = 0.3, 4.6
    col_w, row_h = 3.95, 0.6
    for i, (code, name, color) in enumerate(modes):
        col = i % 3
        row = i // 3
        x = x0 + col * col_w
        y = y0 - row * (row_h + 0.12)
        _box(ax, x, y, 1.0, row_h, code, color, fontsize=8.5)
        ax.text(x + 1.15, y + row_h / 2, name, va="center", fontsize=8.5, color="#37474F")

    # Ordre de dispatch
    ax.text(0.3, 0.95, "Ordre d'évaluation (exact-match avant subset, sinon collisions de bits) :",
            fontsize=9, fontweight="bold", color=COLORS["bad"])
    ax.text(0.3, 0.45,
            "TRIPLE → PAIR → DUAL(0x70) → MULTICLASS(0x30) → RUL(0x50) → HDC_INT8(0x60) → "
            "TINYOL_INT8(0xC0) → EWC/HDC/INT8/TINYOL → défaut Mahalanobis",
            fontsize=8, family="monospace", color="#455A64")

    save_or_show(fig, "portage_03_mode_dispatch", out_dir, show)


# ---------------------------------------------------------------------------
# 4. Flux de données bout-en-bout
# ---------------------------------------------------------------------------
def plot_dataflow(out_dir, show):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    ax.set_title("Flux de données bout-en-bout — du dataset au JSON d'expérience",
                 fontsize=13, fontweight="bold")

    # Côté PC (haut)
    ax.text(0.2, 7.0, "PC (Python)", fontsize=10, fontweight="bold", color=COLORS["pc"])
    pc_steps = [
        ("Dataset\nCWRU/CMAPSS…", COLORS["pc"]),
        ("Features\nRMS·kurtosis·crête", COLORS["feat"]),
        ("Top-5 +\nZ-score", COLORS["feat"]),
        ("build_frame_v2\n+ CRC8", COLORS["board"]),
    ]
    x = 0.3
    for name, color in pc_steps:
        _box(ax, x, 5.3, 2.4, 1.1, name, color, fontsize=8.5)
        if x > 0.4:
            _arrow(ax, x - 0.25, 5.85, x, 5.85)
        x += 2.85

    # Lien UART descendant
    _arrow(ax, 11.05, 5.3, 11.05, 4.4, color=COLORS["bad"], lw=2.2)
    ax.text(11.2, 4.85, "UART", fontsize=8.5, color=COLORS["bad"], rotation=90, va="center")

    # Côté carte (bas)
    ax.text(0.2, 4.0, "Carte (C)", fontsize=10, fontweight="bold", color=COLORS["board"])
    board_steps = [
        ("RX + CRC\n+ normalise", COLORS["board"]),
        ("Routage\n(nibble haut)", COLORS["board"]),
        ("Forward\n+ update SGD", COLORS["ewc"]),
        ("Métriques\nen ligne", COLORS["ok"]),
        ("TX réponse\n+ profiling", COLORS["pc"]),
    ]
    x = 0.3
    for name, color in board_steps:
        _box(ax, x, 2.5, 2.1, 1.1, name, color, fontsize=8.5)
        if x > 0.4:
            _arrow(ax, x - 0.22, 3.05, x, 3.05)
        x += 2.3

    # Remontée vers JSON
    _arrow(ax, 11.0, 2.5, 11.0, 1.75, color=COLORS["bad"], lw=2.2)
    _arrow(ax, 11.0, 1.75, 10.4, 1.5, color=COLORS["bad"], lw=2.2)
    _box(ax, 0.6, 0.45, 10.8, 1.05,
         "experiments/exp_*/  →  dataset.csv · results.json · config_snapshot.yaml · profiling.json",
         COLORS["ok"], fontsize=9.5)

    save_or_show(fig, "portage_04_dataflow", out_dir, show)


# ---------------------------------------------------------------------------
# 5. Workflow d'export des poids Python → C
# ---------------------------------------------------------------------------
def plot_weights_export(out_dir, show):
    fig, ax = plt.subplots(figsize=(11.5, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Export des poids : Python → header C (jamais édité à la main)",
                 fontsize=13, fontweight="bold")

    _box(ax, 0.4, 4.6, 2.6, 1.3, "Entraînement\nPyTorch / sklearn\n→ .pt / .pkl", COLORS["pc"], fontsize=9)
    _box(ax, 4.0, 4.6, 3.2, 1.3,
         "export_weights_*.py\nc · tinyol · ewc_rul\newc_multiclass · --meta", COLORS["board"], fontsize=9)
    _box(ax, 8.2, 4.6, 3.4, 1.3,
         "model_weights*.h\nmeta_weights.h\n(arrays static const)", COLORS["flash"], fontsize=9)
    _arrow(ax, 3.0, 5.25, 4.0, 5.25)
    _arrow(ax, 7.2, 5.25, 8.2, 5.25)

    # vers la carte
    _box(ax, 8.2, 2.4, 3.4, 1.1, "make all / flash\n→ Flash (.rodata)", COLORS["flash"], fontsize=9)
    _arrow(ax, 9.9, 4.6, 9.9, 3.5)
    _box(ax, 4.0, 2.4, 3.2, 1.1, "pipeline_init()\nmemcpy Flash → .bss", COLORS["bss"], fontsize=9)
    _arrow(ax, 8.2, 2.95, 7.2, 2.95)
    _box(ax, 0.4, 2.4, 2.6, 1.1, "poids vivants\n(SGD en ligne)", COLORS["ewc"], fontsize=9)
    _arrow(ax, 4.0, 2.95, 3.0, 2.95)

    ax.text(6.0, 1.3,
            "Règle projet : interdit d'éditer model_weights.h à la main — toujours régénérer via le script.",
            ha="center", fontsize=9.5, fontweight="bold", color=COLORS["bad"])
    ax.text(6.0, 0.6,
            "Flash = immuable (référence θ*) · .bss = copie modifiable mise à jour par l'apprentissage embarqué.",
            ha="center", fontsize=9, color="#455A64")

    save_or_show(fig, "portage_05_weights_export", out_dir, show)


# ---------------------------------------------------------------------------
# 6. Profiling — DWT (latence) + symboles linker (.bss)
# ---------------------------------------------------------------------------
def plot_profiling(out_dir, show):
    fig, ax = plt.subplots(figsize=(11.5, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Profiling embarqué — mesurer la latence et la RAM sans instrument externe",
                 fontsize=13, fontweight="bold")

    # Latence DWT
    ax.text(0.3, 6.4, "Latence — compteur de cycles DWT CYCCNT", fontsize=10.5, fontweight="bold", color=COLORS["board"])
    _box(ax, 0.3, 5.4, 1.9, 0.8, "profiling_start\nt0 = CYCCNT", COLORS["ok"], fontsize=8.5)
    _box(ax, 2.5, 5.4, 6.4, 0.8, "RX → CRC → normalise → forward → update → métriques → TX", COLORS["board"], fontsize=8.5)
    _box(ax, 9.2, 5.4, 2.4, 0.8, "profiling_stop\nΔ = CYCCNT − t0", COLORS["ok"], fontsize=8.5)
    _arrow(ax, 2.2, 5.8, 2.5, 5.8)
    _arrow(ax, 8.9, 5.8, 9.2, 5.8)
    ax.text(6.0, 4.7, "latence_µs = Δcycles / 180   (180 MHz ⇒ 1 µs = 180 cycles · résolution ≈ 5.5 ns)",
            ha="center", fontsize=9.5, family="monospace", color="#37474F")
    ax.text(6.0, 4.2, "périmètre mesuré = RX → TX (inclut UART) · variantes : 130 µs inférence / 403 µs inférence+update (exp_S26)",
            ha="center", fontsize=8.5, color="#455A64")

    # RAM .bss
    ax.text(0.3, 3.3, "RAM statique — symboles du linker", fontsize=10.5, fontweight="bold", color=COLORS["board"])
    _box(ax, 0.3, 2.3, 2.6, 0.8, "_sbss", COLORS["bss"], fontsize=9)
    _box(ax, 3.1, 2.3, 5.6, 0.8, ".bss (poids vivants · Fisher · métriques)", COLORS["bss"], fontsize=9, alpha=0.85)
    _box(ax, 8.9, 2.3, 2.6, 0.8, "_ebss", COLORS["bss"], fontsize=9)
    ax.text(6.0, 1.55, "bss_bytes = (_ebss − _sbss)   — calculé au runtime, encodé dans la réponse UART",
            ha="center", fontsize=9.5, family="monospace", color="#37474F")
    ax.text(6.0, 1.0, "vérif externe : arm-none-eabi-size build/*.elf   ·   exemples mesurés : .bss 66.7 Ko (S26) → 104.6 Ko (S31)",
            ha="center", fontsize=8.5, color="#455A64")

    save_or_show(fig, "portage_06_profiling", out_dir, show)


def main():
    parser = argparse.ArgumentParser(description="Figures schématiques portage C board")
    parser.add_argument("--output", default="docs/figures/presentation_board/",
                        help="répertoire de sortie des PNG")
    parser.add_argument("--show", action="store_true", help="affiche au lieu de sauvegarder")
    args = parser.parse_args()

    out_dir = None if args.show else args.output
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("Génération des figures schématiques portage…")
    plot_firmware_arch(out_dir, args.show)
    plot_uart_frame(out_dir, args.show)
    plot_mode_dispatch(out_dir, args.show)
    plot_dataflow(out_dir, args.show)
    plot_weights_export(out_dir, args.show)
    plot_profiling(out_dir, args.show)
    print("Terminé.")


if __name__ == "__main__":
    main()
