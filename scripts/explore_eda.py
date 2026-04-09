"""explore_eda.py — Génère toutes les figures EDA pour les deux datasets.

Usage :
    python scripts/explore_eda.py --dataset equipment
    python scripts/explore_eda.py --dataset pump
    python scripts/explore_eda.py --dataset all

Sorties (notebooks/figures/eda/) :
    equipment_monitoring/
        boxplots_by_faulty.png
        histograms_by_faulty.png
        violin_by_faulty.png
        kde_by_faulty.png
        pairplot_by_faulty.png
        label_distribution.png
        boxplots_by_equipment_faulty.png
        violin_by_equipment_faulty.png
        kde_by_equipment_faulty.png
    pump_maintenance/
        boxplots_by_maintenance.png
        histograms_by_maintenance.png
        violin_by_maintenance.png
        kde_by_maintenance.png
        pairplot_by_maintenance.png
        temporal_by_maintenance.png
        label_distribution.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ajout du répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.evaluation.eda_plots import (
    plot_boxplots_by_group_and_label,
    plot_boxplots_by_label,
    plot_histograms_by_label,
    plot_kde_by_group_and_label,
    plot_kde_by_label,
    plot_label_distribution,
    plot_pairplot_by_label,
    plot_temporal_by_label,
    plot_violin_by_group_and_label,
    plot_violin_by_label,
)
from src.evaluation.plots import save_figure
from src.utils.config_loader import load_config

# ---------------------------------------------------------------------------
# Constantes datasets
# ---------------------------------------------------------------------------

EQUIPMENT_FEATURE_COLS = ["temperature", "pressure", "vibration", "humidity"]
EQUIPMENT_LABEL_COL = "faulty"
EQUIPMENT_LABEL_NAME = "Faulty"

PUMP_COL_RENAME = {
    "Temperature": "temperature",
    "Vibration": "vibration",
    "Pressure": "pressure",
    "Flow_Rate": "flow_rate",
    "RPM": "rpm",
    "Operational_Hours": "operational_hours",
    "Maintenance_Flag": "maintenance_required",
}
PUMP_FEATURE_COLS = ["temperature", "vibration", "pressure", "flow_rate", "rpm"]
PUMP_LABEL_COL = "maintenance_required"
PUMP_LABEL_NAME = "Maintenance"
PUMP_TIME_COL = "operational_hours"


def run_equipment_eda(csv_path: Path, out_dir: Path) -> None:
    """Génère toutes les figures EDA pour le Dataset 2 (Equipment Monitoring)."""
    print(f"[EDA] Chargement Dataset 2 : {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  → {df.shape[0]} lignes, {df.shape[1]} colonnes")

    out_dir.mkdir(parents=True, exist_ok=True)

    print("[EDA] Distribution du label (par équipement)...")
    fig = plot_label_distribution(
        df,
        label_col=EQUIPMENT_LABEL_COL,
        label_name=EQUIPMENT_LABEL_NAME,
        group_col="equipment",
        title="Taux de défaut par type d'équipement — Dataset 2",
    )
    save_figure(fig, out_dir / "label_distribution.png")

    print("[EDA] Boxplots par faulty...")
    fig = plot_boxplots_by_label(
        df,
        feature_cols=EQUIPMENT_FEATURE_COLS,
        label_col=EQUIPMENT_LABEL_COL,
        label_name=EQUIPMENT_LABEL_NAME,
        title="Boxplots par état (faulty) — Dataset 2 Equipment Monitoring",
    )
    save_figure(fig, out_dir / "boxplots_by_faulty.png")

    print("[EDA] Histogrammes + KDE par faulty...")
    fig = plot_histograms_by_label(
        df,
        feature_cols=EQUIPMENT_FEATURE_COLS,
        label_col=EQUIPMENT_LABEL_COL,
        label_name=EQUIPMENT_LABEL_NAME,
        title="Histogrammes + KDE par état (faulty) — Dataset 2",
    )
    save_figure(fig, out_dir / "histograms_by_faulty.png")

    print("[EDA] Violin plots par faulty...")
    fig = plot_violin_by_label(
        df,
        feature_cols=EQUIPMENT_FEATURE_COLS,
        label_col=EQUIPMENT_LABEL_COL,
        label_name=EQUIPMENT_LABEL_NAME,
        title="Violin plots par état (faulty) — Dataset 2",
    )
    save_figure(fig, out_dir / "violin_by_faulty.png")

    print("[EDA] Densités KDE par faulty...")
    fig = plot_kde_by_label(
        df,
        feature_cols=EQUIPMENT_FEATURE_COLS,
        label_col=EQUIPMENT_LABEL_COL,
        label_name=EQUIPMENT_LABEL_NAME,
        title="Densités KDE par état (faulty) — Dataset 2",
    )
    save_figure(fig, out_dir / "kde_by_faulty.png")

    print("[EDA] Pairplot par faulty...")
    fig = plot_pairplot_by_label(
        df,
        feature_cols=EQUIPMENT_FEATURE_COLS,
        label_col=EQUIPMENT_LABEL_COL,
        label_name=EQUIPMENT_LABEL_NAME,
        title="Scatter matrix par état (faulty) — Dataset 2",
    )
    save_figure(fig, out_dir / "pairplot_by_faulty.png")

    print("[EDA] Boxplots par équipement et faulty...")
    fig = plot_boxplots_by_group_and_label(
        df,
        feature_cols=EQUIPMENT_FEATURE_COLS,
        label_col=EQUIPMENT_LABEL_COL,
        group_col="equipment",
        label_name=EQUIPMENT_LABEL_NAME,
        title="Boxplots par équipement et état (faulty) — Dataset 2",
    )
    save_figure(fig, out_dir / "boxplots_by_equipment_faulty.png")

    print("[EDA] Violin plots par équipement et faulty...")
    fig = plot_violin_by_group_and_label(
        df,
        feature_cols=EQUIPMENT_FEATURE_COLS,
        label_col=EQUIPMENT_LABEL_COL,
        group_col="equipment",
        label_name=EQUIPMENT_LABEL_NAME,
        title="Violin plots par équipement et état (faulty) — Dataset 2",
    )
    save_figure(fig, out_dir / "violin_by_equipment_faulty.png")

    print("[EDA] Densités KDE par équipement et faulty...")
    fig = plot_kde_by_group_and_label(
        df,
        feature_cols=EQUIPMENT_FEATURE_COLS,
        label_col=EQUIPMENT_LABEL_COL,
        group_col="equipment",
        label_name=EQUIPMENT_LABEL_NAME,
        title="Densités KDE par équipement et état (faulty) — Dataset 2",
    )
    save_figure(fig, out_dir / "kde_by_equipment_faulty.png")

    print(f"[EDA] Dataset 2 terminé → {out_dir}")


def run_pump_eda(csv_path: Path, out_dir: Path) -> None:
    """Génère toutes les figures EDA pour le Dataset 1 (Pump Maintenance)."""
    print(f"[EDA] Chargement Dataset 1 : {csv_path}")
    df = pd.read_csv(csv_path).rename(columns=PUMP_COL_RENAME)
    df = df.sort_values(PUMP_TIME_COL).reset_index(drop=True)
    print(f"  → {df.shape[0]} lignes, {df.shape[1]} colonnes")

    out_dir.mkdir(parents=True, exist_ok=True)

    print("[EDA] Distribution du label...")
    fig = plot_label_distribution(
        df,
        label_col=PUMP_LABEL_COL,
        label_name=PUMP_LABEL_NAME,
        title="Distribution du label maintenance — Dataset 1 Pump",
    )
    save_figure(fig, out_dir / "label_distribution.png")

    print("[EDA] Boxplots par maintenance...")
    fig = plot_boxplots_by_label(
        df,
        feature_cols=PUMP_FEATURE_COLS,
        label_col=PUMP_LABEL_COL,
        label_name=PUMP_LABEL_NAME,
        title="Boxplots par état (maintenance) — Dataset 1 Pump",
    )
    save_figure(fig, out_dir / "boxplots_by_maintenance.png")

    print("[EDA] Histogrammes + KDE par maintenance...")
    fig = plot_histograms_by_label(
        df,
        feature_cols=PUMP_FEATURE_COLS,
        label_col=PUMP_LABEL_COL,
        label_name=PUMP_LABEL_NAME,
        title="Histogrammes + KDE par état (maintenance) — Dataset 1",
    )
    save_figure(fig, out_dir / "histograms_by_maintenance.png")

    print("[EDA] Violin plots par maintenance...")
    fig = plot_violin_by_label(
        df,
        feature_cols=PUMP_FEATURE_COLS,
        label_col=PUMP_LABEL_COL,
        label_name=PUMP_LABEL_NAME,
        title="Violin plots par état (maintenance) — Dataset 1",
    )
    save_figure(fig, out_dir / "violin_by_maintenance.png")

    print("[EDA] Densités KDE par maintenance...")
    fig = plot_kde_by_label(
        df,
        feature_cols=PUMP_FEATURE_COLS,
        label_col=PUMP_LABEL_COL,
        label_name=PUMP_LABEL_NAME,
        title="Densités KDE par état (maintenance) — Dataset 1",
    )
    save_figure(fig, out_dir / "kde_by_maintenance.png")

    print("[EDA] Pairplot par maintenance...")
    fig = plot_pairplot_by_label(
        df,
        feature_cols=PUMP_FEATURE_COLS,
        label_col=PUMP_LABEL_COL,
        label_name=PUMP_LABEL_NAME,
        title="Scatter matrix par état (maintenance) — Dataset 1",
    )
    save_figure(fig, out_dir / "pairplot_by_maintenance.png")

    print("[EDA] Évolution temporelle par maintenance...")
    fig = plot_temporal_by_label(
        df,
        feature_cols=PUMP_FEATURE_COLS,
        label_col=PUMP_LABEL_COL,
        time_col=PUMP_TIME_COL,
        label_name=PUMP_LABEL_NAME,
        title="Évolution temporelle par état (maintenance) — Dataset 1",
    )
    save_figure(fig, out_dir / "temporal_by_maintenance.png")

    print(f"[EDA] Dataset 1 terminé → {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Génération des figures EDA pour CL-Embedded")
    parser.add_argument(
        "--dataset",
        choices=["equipment", "pump", "all"],
        default="all",
        help="Dataset(s) à traiter (défaut : all)",
    )
    parser.add_argument(
        "--config-equipment",
        default="configs/ewc_config.yaml",
        help="Config YAML pour le dataset Equipment Monitoring",
    )
    parser.add_argument(
        "--config-pump",
        default="configs/tinyol_config.yaml",
        help="Config YAML pour le dataset Pump Maintenance",
    )
    parser.add_argument(
        "--out-dir",
        default="notebooks/figures/eda",
        help="Répertoire racine de sortie (défaut : notebooks/figures/eda)",
    )
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    out_base = root / args.out_dir

    if args.dataset in ("equipment", "all"):
        cfg = load_config(root / args.config_equipment)
        csv_path = root / cfg["data"]["csv_path"]
        run_equipment_eda(csv_path, out_base / "equipment_monitoring")

    if args.dataset in ("pump", "all"):
        cfg = load_config(root / args.config_pump)
        csv_path = root / cfg["data"]["csv_path"]
        run_pump_eda(csv_path, out_base / "pump_maintenance")


if __name__ == "__main__":
    main()
