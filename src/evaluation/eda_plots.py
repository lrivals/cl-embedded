"""Visualisations EDA pour l'exploration des deux datasets CL-Embedded.

Fonctions de visualisation orientées analyse de la séparation normal/faulty,
utilisées dans notebooks/01_data_exploration.ipynb et scripts/explore_eda.py.

Contrainte : ce module ne doit être importé que depuis src/evaluation/ et notebooks/.
Ne jamais importer matplotlib/seaborn dans les modules modèles ou d'entraînement.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend non-interactif (serveurs, CI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns

    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False

# Style cohérent avec plots.py
FIGURE_DPI: int = 150
FIGURE_FONT_SIZE: int = 11
COLORS_LABEL: dict[int, str] = {0: "#4CAF50", 1: "#F44336"}  # vert=normal, rouge=faulty
LABEL_NAMES_DEFAULT: dict[int, str] = {0: "Normal", 1: "Faulty"}


def _label_palette(label_col: str, df: pd.DataFrame) -> dict:
    """Retourne un dict {valeur_label: couleur} pour seaborn."""
    vals = sorted(df[label_col].unique())
    return {v: COLORS_LABEL.get(int(v), "#999999") for v in vals}


def plot_boxplots_by_label(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    label_name: str = "Label",
    title: str | None = None,
) -> plt.Figure:
    """Boxplots de chaque feature colorés par valeur du label.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset complet (toutes lignes).
    feature_cols : list[str]
        Colonnes numériques à visualiser.
    label_col : str
        Colonne label (ex. "faulty", "maintenance_required").
    label_name : str
        Nom affiché dans la légende.
    title : str | None
        Titre de la figure.

    Returns
    -------
    plt.Figure
    """
    n = len(feature_cols)
    fig, axes = plt.subplots(1, n, figsize=(max(4 * n, 12), 5))
    if n == 1:
        axes = [axes]

    label_vals = sorted(df[label_col].unique())
    label_labels = [f"{label_name}={int(v)}" for v in label_vals]

    for ax, feat in zip(axes, feature_cols):
        data_by_label = [df[df[label_col] == v][feat].dropna().values for v in label_vals]

        if _HAS_SEABORN:
            plot_df = df[[feat, label_col]].copy()
            plot_df[label_col] = plot_df[label_col].astype(str)
            palette = {str(v): COLORS_LABEL.get(int(v), "#999999") for v in label_vals}
            sns.boxplot(
                data=plot_df,
                x=label_col,
                y=feat,
                palette=palette,
                ax=ax,
                linewidth=1.2,
                flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
            )
            ax.set_xticklabels(label_labels, fontsize=FIGURE_FONT_SIZE - 1)
        else:
            bp = ax.boxplot(
                data_by_label,
                patch_artist=True,
                medianprops={"color": "black", "linewidth": 2},
                flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
            )
            for patch, v in zip(bp["boxes"], label_vals):
                patch.set_facecolor(COLORS_LABEL.get(int(v), "#999999"))
                patch.set_alpha(0.8)
            ax.set_xticklabels(label_labels, fontsize=FIGURE_FONT_SIZE - 1)

        ax.set_title(feat, fontsize=FIGURE_FONT_SIZE)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        title or f"Boxplots par {label_name}",
        fontsize=FIGURE_FONT_SIZE + 2,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_histograms_by_label(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    label_name: str = "Label",
    bins: int = 40,
    title: str | None = None,
) -> plt.Figure:
    """Histogrammes avec KDE overlay par feature, colorés par label.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
    label_col : str
    label_name : str
    bins : int
        Nombre de bins pour l'histogramme.
    title : str | None

    Returns
    -------
    plt.Figure
    """
    n = len(feature_cols)
    fig, axes = plt.subplots(1, n, figsize=(max(4 * n, 12), 5))
    if n == 1:
        axes = [axes]

    label_vals = sorted(df[label_col].unique())

    for ax, feat in zip(axes, feature_cols):
        if _HAS_SEABORN:
            plot_df = df[[feat, label_col]].copy()
            plot_df[label_col] = plot_df[label_col].astype(str)
            palette = {str(v): COLORS_LABEL.get(int(v), "#999999") for v in label_vals}
            for v in label_vals:
                subset = df[df[label_col] == v][feat].dropna()
                sns.histplot(
                    subset,
                    ax=ax,
                    kde=True,
                    color=COLORS_LABEL.get(int(v), "#999999"),
                    alpha=0.4,
                    bins=bins,
                    label=f"{label_name}={int(v)}",
                    stat="density",
                )
        else:
            for v in label_vals:
                subset = df[df[label_col] == v][feat].dropna()
                ax.hist(
                    subset,
                    bins=bins,
                    alpha=0.4,
                    color=COLORS_LABEL.get(int(v), "#999999"),
                    density=True,
                    label=f"{label_name}={int(v)}",
                )

        ax.set_title(feat, fontsize=FIGURE_FONT_SIZE)
        ax.set_xlabel("")
        ax.set_ylabel("Densité" if feat == feature_cols[0] else "")
        ax.legend(fontsize=FIGURE_FONT_SIZE - 2)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        title or f"Histogrammes + KDE par {label_name}",
        fontsize=FIGURE_FONT_SIZE + 2,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_violin_by_label(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    label_name: str = "Label",
    title: str | None = None,
) -> plt.Figure:
    """Violin plots par feature, colorés par label.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
    label_col : str
    label_name : str
    title : str | None

    Returns
    -------
    plt.Figure
    """
    n = len(feature_cols)
    fig, axes = plt.subplots(1, n, figsize=(max(4 * n, 12), 5))
    if n == 1:
        axes = [axes]

    label_vals = sorted(df[label_col].unique())
    label_labels = [f"{label_name}={int(v)}" for v in label_vals]

    for ax, feat in zip(axes, feature_cols):
        if _HAS_SEABORN:
            plot_df = df[[feat, label_col]].copy()
            plot_df[label_col] = plot_df[label_col].astype(str)
            palette = {str(v): COLORS_LABEL.get(int(v), "#999999") for v in label_vals}
            sns.violinplot(
                data=plot_df,
                x=label_col,
                y=feat,
                palette=palette,
                ax=ax,
                inner="quartile",
                linewidth=1.0,
            )
            ax.set_xticklabels(label_labels, fontsize=FIGURE_FONT_SIZE - 1)
        else:
            data_by_label = [df[df[label_col] == v][feat].dropna().values for v in label_vals]
            parts = ax.violinplot(data_by_label, positions=range(len(label_vals)), showmedians=True)
            for i, (pc, v) in enumerate(zip(parts["bodies"], label_vals)):
                pc.set_facecolor(COLORS_LABEL.get(int(v), "#999999"))
                pc.set_alpha(0.7)
            ax.set_xticks(range(len(label_vals)))
            ax.set_xticklabels(label_labels, fontsize=FIGURE_FONT_SIZE - 1)

        ax.set_title(feat, fontsize=FIGURE_FONT_SIZE)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        title or f"Violin plots par {label_name}",
        fontsize=FIGURE_FONT_SIZE + 2,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_kde_by_label(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    label_name: str = "Label",
    title: str | None = None,
) -> plt.Figure:
    """Courbes de densité KDE superposées par feature — normal vs faulty.

    Montre la séparation de distribution entre les deux classes.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
    label_col : str
    label_name : str
    title : str | None

    Returns
    -------
    plt.Figure
    """
    n = len(feature_cols)
    fig, axes = plt.subplots(1, n, figsize=(max(4 * n, 12), 4))
    if n == 1:
        axes = [axes]

    label_vals = sorted(df[label_col].unique())

    for ax, feat in zip(axes, feature_cols):
        for v in label_vals:
            subset = df[df[label_col] == v][feat].dropna()
            color = COLORS_LABEL.get(int(v), "#999999")
            label = f"{label_name}={int(v)} (n={len(subset)})"
            if _HAS_SEABORN:
                sns.kdeplot(subset, ax=ax, color=color, fill=True, alpha=0.3, label=label)
            else:
                # Estimation KDE manuelle via numpy
                from scipy.stats import gaussian_kde  # type: ignore

                kde = gaussian_kde(subset)
                x_range = np.linspace(subset.min(), subset.max(), 200)
                ax.fill_between(x_range, kde(x_range), alpha=0.3, color=color, label=label)
                ax.plot(x_range, kde(x_range), color=color, lw=1.5)

        ax.set_title(feat, fontsize=FIGURE_FONT_SIZE)
        ax.set_xlabel("")
        ax.set_ylabel("Densité" if feat == feature_cols[0] else "")
        ax.legend(fontsize=FIGURE_FONT_SIZE - 2)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        title or f"Densités KDE par {label_name}",
        fontsize=FIGURE_FONT_SIZE + 2,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_pairplot_by_label(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    label_name: str = "Label",
    sample_n: int = 2000,
    title: str | None = None,
) -> plt.Figure:
    """Scatter matrix (pairplot) colorée par label.

    Diagonale : KDE par label. Hors-diagonale : scatter.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
    label_col : str
    label_name : str
    sample_n : int
        Nombre de points max (subsample pour rapidité).
    title : str | None

    Returns
    -------
    plt.Figure
    """
    plot_df = df[feature_cols + [label_col]].dropna()
    if len(plot_df) > sample_n:
        plot_df = plot_df.sample(n=sample_n, random_state=42)

    label_vals = sorted(plot_df[label_col].unique())
    palette = {v: COLORS_LABEL.get(int(v), "#999999") for v in label_vals}
    label_map = {v: f"{label_name}={int(v)}" for v in label_vals}

    if _HAS_SEABORN:
        plot_df = plot_df.copy()
        plot_df["_label"] = plot_df[label_col].map(label_map)
        palette_str = {label_map[v]: c for v, c in palette.items()}
        pg = sns.pairplot(
            plot_df,
            vars=feature_cols,
            hue="_label",
            palette=palette_str,
            diag_kind="kde",
            plot_kws={"alpha": 0.3, "s": 8, "linewidths": 0},
            diag_kws={"fill": True, "alpha": 0.4},
            corner=False,
        )
        pg.fig.suptitle(
            title or f"Pairplot par {label_name}",
            fontsize=FIGURE_FONT_SIZE + 2,
            fontweight="bold",
            y=1.02,
        )
        return pg.fig
    else:
        n = len(feature_cols)
        fig, axes = plt.subplots(n, n, figsize=(max(3 * n, 10), max(3 * n, 10)))
        for i, feat_y in enumerate(feature_cols):
            for j, feat_x in enumerate(feature_cols):
                ax = axes[i, j]
                if i == j:
                    for v in label_vals:
                        subset = plot_df[plot_df[label_col] == v][feat_x].dropna()
                        ax.hist(subset, bins=25, alpha=0.4, color=palette[v], density=True)
                else:
                    for v in label_vals:
                        mask = plot_df[label_col] == v
                        ax.scatter(
                            plot_df.loc[mask, feat_x],
                            plot_df.loc[mask, feat_y],
                            c=palette[v],
                            alpha=0.3,
                            s=5,
                            linewidths=0,
                            label=label_map[v],
                        )
                if i == n - 1:
                    ax.set_xlabel(feat_x, fontsize=8)
                if j == 0:
                    ax.set_ylabel(feat_y, fontsize=8)
                ax.tick_params(labelsize=7)

        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[v], markersize=7,
                       label=label_map[v])
            for v in label_vals
        ]
        fig.legend(handles=handles, loc="upper right", fontsize=FIGURE_FONT_SIZE - 1)
        fig.suptitle(
            title or f"Pairplot par {label_name}",
            fontsize=FIGURE_FONT_SIZE + 2,
            fontweight="bold",
        )
        fig.tight_layout()
        return fig


def plot_label_distribution(
    df: pd.DataFrame,
    label_col: str,
    label_name: str = "Label",
    group_col: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Barplot de la distribution du label, optionnellement par groupe.

    Parameters
    ----------
    df : pd.DataFrame
    label_col : str
    label_name : str
    group_col : str | None
        Si fourni, affiche le taux de label=1 par groupe (ex. "equipment").
    title : str | None

    Returns
    -------
    plt.Figure
    """
    if group_col is not None:
        stats = df.groupby(group_col)[label_col].agg(["mean", "sum", "count"])
        stats.columns = ["rate", "n_pos", "n_total"]

        fig, ax = plt.subplots(figsize=(max(5, len(stats) * 1.5), 4))
        colors = [COLORS_LABEL.get(1, "#F44336")] * len(stats)
        bars = ax.bar(stats.index, stats["rate"], color=colors, alpha=0.8, edgecolor="white")

        # Annotations
        for bar, (_, row) in zip(bars, stats.iterrows()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{row['rate']:.1%}\n(n={int(row['n_pos'])})",
                ha="center",
                va="bottom",
                fontsize=FIGURE_FONT_SIZE - 1,
            )

        global_rate = df[label_col].mean()
        ax.axhline(global_rate, color="black", linestyle="--", linewidth=1.2,
                   label=f"Global ({global_rate:.1%})")
        ax.set_ylabel(f"Taux {label_name}=1", fontsize=FIGURE_FONT_SIZE)
        ax.set_xlabel(group_col, fontsize=FIGURE_FONT_SIZE)
        ax.set_ylim(0, min(1.0, stats["rate"].max() * 1.3))
        ax.legend(fontsize=FIGURE_FONT_SIZE - 1)
        ax.grid(True, axis="y", alpha=0.3)
    else:
        counts = df[label_col].value_counts().sort_index()
        label_vals = counts.index.tolist()
        colors = [COLORS_LABEL.get(int(v), "#999999") for v in label_vals]

        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(
            [f"{label_name}={int(v)}" for v in label_vals],
            counts.values,
            color=colors,
            alpha=0.8,
            edgecolor="white",
        )
        for bar, count in zip(bars, counts.values):
            pct = count / len(df)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + len(df) * 0.01,
                f"{count}\n({pct:.1%})",
                ha="center",
                va="bottom",
                fontsize=FIGURE_FONT_SIZE - 1,
            )
        ax.set_ylabel("Nombre d'échantillons", fontsize=FIGURE_FONT_SIZE)
        ax.set_xlabel("")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        title or f"Distribution du label {label_name}",
        fontsize=FIGURE_FONT_SIZE + 2,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_boxplots_by_group_and_label(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    group_col: str,
    label_name: str = "Label",
    title: str | None = None,
) -> plt.Figure:
    """Boxplots en grille (groupes × features), colorés par label.

    Chaque ligne correspond à une valeur unique de group_col (ex. "equipment"),
    chaque colonne à une feature. Utile pour visualiser les différences inter-domaines
    en scénario domain-incremental.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
    label_col : str
    group_col : str
        Colonne de groupement (ex. "equipment").
    label_name : str
    title : str | None

    Returns
    -------
    plt.Figure
    """
    groups = sorted(df[group_col].unique())
    n_groups = len(groups)
    n_feats = len(feature_cols)
    label_vals = sorted(df[label_col].unique())
    palette = {str(v): COLORS_LABEL.get(int(v), "#999999") for v in label_vals}

    fig, axes = plt.subplots(
        n_groups, n_feats, figsize=(4 * n_feats, 3.5 * n_groups), squeeze=False
    )

    for row_idx, grp in enumerate(groups):
        df_grp = df[df[group_col] == grp]
        for col_idx, feat in enumerate(feature_cols):
            ax = axes[row_idx][col_idx]
            if _HAS_SEABORN:
                plot_df = df_grp[[feat, label_col]].copy()
                plot_df[label_col] = plot_df[label_col].astype(str)
                sns.boxplot(
                    data=plot_df,
                    x=label_col,
                    y=feat,
                    palette=palette,
                    ax=ax,
                    linewidth=1.0,
                    flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
                )
                ax.set_xticklabels(
                    [f"{label_name}={int(v)}" for v in label_vals],
                    fontsize=FIGURE_FONT_SIZE - 2,
                )
            else:
                data_by_label = [df_grp[df_grp[label_col] == v][feat].dropna().values for v in label_vals]
                bp = ax.boxplot(
                    data_by_label,
                    patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.5},
                    flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
                )
                for patch, v in zip(bp["boxes"], label_vals):
                    patch.set_facecolor(COLORS_LABEL.get(int(v), "#999999"))
                    patch.set_alpha(0.8)
                ax.set_xticks(range(1, len(label_vals) + 1))
                ax.set_xticklabels(
                    [f"{label_name}={int(v)}" for v in label_vals],
                    fontsize=FIGURE_FONT_SIZE - 2,
                )

            # Titres colonnes (première ligne seulement)
            if row_idx == 0:
                ax.set_title(feat, fontsize=FIGURE_FONT_SIZE)
            # Labels lignes (première colonne seulement)
            if col_idx == 0:
                ax.set_ylabel(str(grp), fontsize=FIGURE_FONT_SIZE, fontweight="bold")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        title or f"Boxplots par {group_col} et {label_name}",
        fontsize=FIGURE_FONT_SIZE + 2,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_violin_by_group_and_label(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    group_col: str,
    label_name: str = "Label",
    title: str | None = None,
) -> plt.Figure:
    """Violin plots en grille (groupes × features), colorés par label.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
    label_col : str
    group_col : str
    label_name : str
    title : str | None

    Returns
    -------
    plt.Figure
    """
    groups = sorted(df[group_col].unique())
    n_groups = len(groups)
    n_feats = len(feature_cols)
    label_vals = sorted(df[label_col].unique())
    palette = {str(v): COLORS_LABEL.get(int(v), "#999999") for v in label_vals}

    fig, axes = plt.subplots(
        n_groups, n_feats, figsize=(4 * n_feats, 3.5 * n_groups), squeeze=False
    )

    for row_idx, grp in enumerate(groups):
        df_grp = df[df[group_col] == grp]
        for col_idx, feat in enumerate(feature_cols):
            ax = axes[row_idx][col_idx]
            if _HAS_SEABORN:
                plot_df = df_grp[[feat, label_col]].copy()
                plot_df[label_col] = plot_df[label_col].astype(str)
                sns.violinplot(
                    data=plot_df,
                    x=label_col,
                    y=feat,
                    palette=palette,
                    ax=ax,
                    inner="quartile",
                    linewidth=0.8,
                )
                ax.set_xticklabels(
                    [f"{label_name}={int(v)}" for v in label_vals],
                    fontsize=FIGURE_FONT_SIZE - 2,
                )
            else:
                data_by_label = [df_grp[df_grp[label_col] == v][feat].dropna().values for v in label_vals]
                parts = ax.violinplot(data_by_label, positions=range(len(label_vals)), showmedians=True)
                for pc, v in zip(parts["bodies"], label_vals):
                    pc.set_facecolor(COLORS_LABEL.get(int(v), "#999999"))
                    pc.set_alpha(0.7)
                ax.set_xticks(range(len(label_vals)))
                ax.set_xticklabels(
                    [f"{label_name}={int(v)}" for v in label_vals],
                    fontsize=FIGURE_FONT_SIZE - 2,
                )

            if row_idx == 0:
                ax.set_title(feat, fontsize=FIGURE_FONT_SIZE)
            if col_idx == 0:
                ax.set_ylabel(str(grp), fontsize=FIGURE_FONT_SIZE, fontweight="bold")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        title or f"Violin plots par {group_col} et {label_name}",
        fontsize=FIGURE_FONT_SIZE + 2,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_kde_by_group_and_label(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    group_col: str,
    label_name: str = "Label",
    title: str | None = None,
) -> plt.Figure:
    """Courbes KDE en grille (groupes × features), colorées par label.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
    label_col : str
    group_col : str
    label_name : str
    title : str | None

    Returns
    -------
    plt.Figure
    """
    groups = sorted(df[group_col].unique())
    n_groups = len(groups)
    n_feats = len(feature_cols)
    label_vals = sorted(df[label_col].unique())

    fig, axes = plt.subplots(
        n_groups, n_feats, figsize=(4 * n_feats, 3 * n_groups), squeeze=False
    )

    for row_idx, grp in enumerate(groups):
        df_grp = df[df[group_col] == grp]
        for col_idx, feat in enumerate(feature_cols):
            ax = axes[row_idx][col_idx]
            for v in label_vals:
                subset = df_grp[df_grp[label_col] == v][feat].dropna()
                color = COLORS_LABEL.get(int(v), "#999999")
                lbl = f"{label_name}={int(v)}" if row_idx == 0 and col_idx == 0 else None
                if _HAS_SEABORN:
                    sns.kdeplot(subset, ax=ax, color=color, fill=True, alpha=0.3, label=lbl)
                else:
                    from scipy.stats import gaussian_kde  # type: ignore

                    kde = gaussian_kde(subset)
                    x_range = np.linspace(subset.min(), subset.max(), 200)
                    ax.fill_between(x_range, kde(x_range), alpha=0.3, color=color, label=lbl)
                    ax.plot(x_range, kde(x_range), color=color, lw=1.5)

            if row_idx == 0:
                ax.set_title(feat, fontsize=FIGURE_FONT_SIZE)
            if col_idx == 0:
                ax.set_ylabel(str(grp), fontsize=FIGURE_FONT_SIZE, fontweight="bold")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            ax.grid(True, alpha=0.3)

    # Légende unique dans le premier subplot
    axes[0][0].legend(fontsize=FIGURE_FONT_SIZE - 2)

    fig.suptitle(
        title or f"Densités KDE par {group_col} et {label_name}",
        fontsize=FIGURE_FONT_SIZE + 2,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_temporal_by_label(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    time_col: str,
    label_name: str = "Label",
    rolling_window: int = 200,
    title: str | None = None,
) -> plt.Figure:
    """Évolution temporelle de chaque feature, colorée par label.

    Une ligne par feature, scatter des points colorés par label +
    moyenne glissante par classe.

    Parameters
    ----------
    df : pd.DataFrame
        Doit être trié par time_col.
    feature_cols : list[str]
    label_col : str
    time_col : str
        Colonne temporelle (ex. "operational_hours").
    label_name : str
    rolling_window : int
        Fenêtre pour la moyenne glissante.
    title : str | None

    Returns
    -------
    plt.Figure
    """
    n = len(feature_cols)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    label_vals = sorted(df[label_col].unique())

    for ax, feat in zip(axes, feature_cols):
        for v in label_vals:
            mask = df[label_col] == v
            color = COLORS_LABEL.get(int(v), "#999999")
            ax.scatter(
                df.loc[mask, time_col],
                df.loc[mask, feat],
                c=color,
                alpha=0.15,
                s=4,
                linewidths=0,
                label=f"{label_name}={int(v)}",
            )
            # Moyenne glissante sur les points de cette classe
            roll = df.loc[mask, feat].rolling(rolling_window, min_periods=1).mean()
            ax.plot(df.loc[mask, time_col], roll, color=color, lw=1.5, alpha=0.85)

        ax.set_ylabel(feat, fontsize=FIGURE_FONT_SIZE - 1)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=FIGURE_FONT_SIZE - 2, loc="upper right")

    axes[-1].set_xlabel(time_col, fontsize=FIGURE_FONT_SIZE)
    fig.suptitle(
        title or f"Évolution temporelle par {label_name}",
        fontsize=FIGURE_FONT_SIZE + 2,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# S5-15 : Fonctions EDA affinées — Dataset 1 (Pump) + Dataset 2 (Equipment)
# ---------------------------------------------------------------------------


def _save_fig(fig: plt.Figure, save_path: Path | None) -> None:
    """Sauvegarde la figure si save_path est fourni, en créant les dossiers parents."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")


# ── Dataset 1 — Pump : analyse par Pump_ID ──────────────────────────────────


def plot_boxplots_by_pump_id(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    pump_col: str = "Pump_ID",
    label_name: str = "Maintenance",
    save_path: Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Boxplots des features par Pump_ID, avec hue = label (normal vs faulty).

    Permet de comparer la distribution de chaque feature entre les pompes
    et de repérer les pompes à comportement anormal.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset Pump (doit contenir pump_col, feature_cols et label_col).
    feature_cols : list[str]
        Colonnes numériques à visualiser (ex. Temperature, Vibration, …).
    label_col : str
        Colonne label binaire (ex. "Maintenance_Flag").
    pump_col : str
        Colonne identifiant la pompe (ex. "Pump_ID").
    label_name : str
        Nom affiché dans la légende.
    save_path : Path | None
        Si fourni, sauvegarde la figure au chemin indiqué.
    title : str | None
        Titre de la figure.

    Returns
    -------
    plt.Figure
    """
    n = len(feature_cols)
    pump_ids = sorted(df[pump_col].unique())
    label_vals = sorted(df[label_col].unique())
    palette = {str(v): COLORS_LABEL.get(int(v), "#999999") for v in label_vals}

    fig, axes = plt.subplots(1, n, figsize=(max(5 * n, 14), 5), squeeze=False)
    axes = axes[0]

    for ax, feat in zip(axes, feature_cols):
        if _HAS_SEABORN:
            plot_df = df[[pump_col, feat, label_col]].copy()
            plot_df[label_col] = plot_df[label_col].astype(str)
            plot_df[pump_col] = plot_df[pump_col].astype(str)
            sns.boxplot(
                data=plot_df,
                x=pump_col,
                y=feat,
                hue=label_col,
                palette=palette,
                ax=ax,
                linewidth=1.0,
                flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
                order=[str(p) for p in pump_ids],
            )
            from matplotlib.patches import Patch  # local import — visualisation uniquement
            _legend_handles = [
                Patch(color=COLORS_LABEL[0], label="Normal (0)"),
                Patch(color=COLORS_LABEL[1], label=f"{label_name} (1)"),
            ]
            ax.legend(
                handles=_legend_handles,
                title=label_name,
                fontsize=FIGURE_FONT_SIZE - 2,
                title_fontsize=FIGURE_FONT_SIZE - 2,
            )
        else:
            offsets = np.linspace(-0.2, 0.2, len(label_vals))
            for v, offset in zip(label_vals, offsets):
                data_per_pump = [
                    df[(df[pump_col] == p) & (df[label_col] == v)][feat].dropna().values
                    for p in pump_ids
                ]
                positions = np.arange(len(pump_ids)) + offset
                bp = ax.boxplot(
                    data_per_pump,
                    positions=positions,
                    widths=0.35,
                    patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.5},
                    flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
                    manage_ticks=False,
                )
                color = COLORS_LABEL.get(int(v), "#999999")
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.75)
            ax.set_xticks(range(len(pump_ids)))
            ax.set_xticklabels([str(p) for p in pump_ids], fontsize=FIGURE_FONT_SIZE - 1)

        ax.set_title(feat, fontsize=FIGURE_FONT_SIZE)
        ax.set_xlabel(pump_col, fontsize=FIGURE_FONT_SIZE - 1)
        ax.set_ylabel("")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=0)

    fig.suptitle(
        title or f"Boxplots par {pump_col} — {label_name}",
        fontsize=FIGURE_FONT_SIZE + 2,
        fontweight="bold",
    )
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_violin_by_pump_id(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    pump_col: str = "Pump_ID",
    label_name: str = "Maintenance",
    save_path: Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Violin plots des features par Pump_ID, avec hue = label.

    Montre la forme complète de la distribution (bimodalité, queues) par pompe
    et par état (normal vs faulty).

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
    label_col : str
    pump_col : str
    label_name : str
    save_path : Path | None
    title : str | None

    Returns
    -------
    plt.Figure
    """
    n = len(feature_cols)
    pump_ids = sorted(df[pump_col].unique())
    label_vals = sorted(df[label_col].unique())
    palette = {str(v): COLORS_LABEL.get(int(v), "#999999") for v in label_vals}

    fig, axes = plt.subplots(1, n, figsize=(max(5 * n, 14), 5), squeeze=False)
    axes = axes[0]

    for ax, feat in zip(axes, feature_cols):
        if _HAS_SEABORN:
            plot_df = df[[pump_col, feat, label_col]].copy()
            plot_df[label_col] = plot_df[label_col].astype(str)
            plot_df[pump_col] = plot_df[pump_col].astype(str)
            sns.violinplot(
                data=plot_df,
                x=pump_col,
                y=feat,
                hue=label_col,
                palette=palette,
                ax=ax,
                inner="quartile",
                linewidth=0.8,
                order=[str(p) for p in pump_ids],
            )
            from matplotlib.patches import Patch  # local import — visualisation uniquement
            _legend_handles = [
                Patch(color=COLORS_LABEL[0], label="Normal (0)"),
                Patch(color=COLORS_LABEL[1], label=f"{label_name} (1)"),
            ]
            ax.legend(
                handles=_legend_handles,
                title=label_name,
                fontsize=FIGURE_FONT_SIZE - 2,
                title_fontsize=FIGURE_FONT_SIZE - 2,
            )
        else:
            for idx, p in enumerate(pump_ids):
                data_by_label = [
                    df[(df[pump_col] == p) & (df[label_col] == v)][feat].dropna().values
                    for v in label_vals
                ]
                parts = ax.violinplot(
                    data_by_label,
                    positions=[idx - 0.15, idx + 0.15],
                    widths=0.25,
                    showmedians=True,
                )
                for pc, v in zip(parts["bodies"], label_vals):
                    pc.set_facecolor(COLORS_LABEL.get(int(v), "#999999"))
                    pc.set_alpha(0.7)
            ax.set_xticks(range(len(pump_ids)))
            ax.set_xticklabels([str(p) for p in pump_ids], fontsize=FIGURE_FONT_SIZE - 1)

        ax.set_title(feat, fontsize=FIGURE_FONT_SIZE)
        ax.set_xlabel(pump_col, fontsize=FIGURE_FONT_SIZE - 1)
        ax.set_ylabel("")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        title or f"Violin plots par {pump_col} — {label_name}",
        fontsize=FIGURE_FONT_SIZE + 2,
        fontweight="bold",
    )
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


# ── Dataset 1 — Pump : analyse par fenêtres Operational_Hours ───────────────


def plot_operational_hour_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    hour_col: str = "Operational_Hours",
    label_col: str = "Maintenance_Flag",
    n_windows: int = 5,
    label_name: str = "Maintenance",
    save_path: Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Boxplots + stripplot des features par fenêtres d'heures opérationnelles.

    Les heures sont discrétisées en n_windows intervalles égaux via pd.cut.
    Révèle la dérive temporelle des distributions (domain shift au fil du temps).

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
    hour_col : str
        Colonne contenant les heures opérationnelles (ex. "Operational_Hours").
    label_col : str
    n_windows : int
        Nombre de fenêtres temporelles (bins équidistants).
    label_name : str
    save_path : Path | None
    title : str | None

    Returns
    -------
    plt.Figure
    """
    df = df.copy()
    hour_min = df[hour_col].min()
    hour_max = df[hour_col].max()
    bin_edges = np.linspace(hour_min, hour_max, n_windows + 1)
    bin_labels = [
        f"[{int(bin_edges[i])}-{int(bin_edges[i+1])}]" for i in range(n_windows)
    ]
    df["_hour_window"] = pd.cut(
        df[hour_col], bins=bin_edges, labels=bin_labels, include_lowest=True
    )

    n = len(feature_cols)
    label_vals = sorted(df[label_col].unique())
    palette = {str(v): COLORS_LABEL.get(int(v), "#999999") for v in label_vals}

    fig, axes = plt.subplots(1, n, figsize=(max(5 * n, 14), 5), squeeze=False)
    axes = axes[0]

    for ax, feat in zip(axes, feature_cols):
        if _HAS_SEABORN:
            plot_df = df[["_hour_window", feat, label_col]].dropna().copy()
            plot_df[label_col] = plot_df[label_col].astype(str)
            sns.boxplot(
                data=plot_df,
                x="_hour_window",
                y=feat,
                hue=label_col,
                palette=palette,
                ax=ax,
                linewidth=1.0,
                flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
                order=bin_labels,
            )
            sns.stripplot(
                data=plot_df,
                x="_hour_window",
                y=feat,
                hue=label_col,
                palette=palette,
                ax=ax,
                alpha=0.15,
                size=2,
                dodge=True,
                order=bin_labels,
                legend=False,
            )
            from matplotlib.patches import Patch  # local import — visualisation uniquement
            _legend_handles = [
                Patch(color=COLORS_LABEL[0], label="Normal (0)"),
                Patch(color=COLORS_LABEL[1], label=f"{label_name} (1)"),
            ]
            ax.legend(
                handles=_legend_handles,
                title=label_name,
                fontsize=FIGURE_FONT_SIZE - 2,
                title_fontsize=FIGURE_FONT_SIZE - 2,
            )
        else:
            for v in label_vals:
                data_per_window = [
                    df[(df["_hour_window"] == w) & (df[label_col] == v)][feat].dropna().values
                    for w in bin_labels
                ]
                positions = np.arange(n_windows) + (0.2 if int(v) == 1 else -0.2)
                bp = ax.boxplot(
                    data_per_window,
                    positions=positions,
                    widths=0.35,
                    patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.5},
                    manage_ticks=False,
                )
                color = COLORS_LABEL.get(int(v), "#999999")
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.75)
            ax.set_xticks(range(n_windows))
            ax.set_xticklabels(bin_labels, fontsize=FIGURE_FONT_SIZE - 2)

        ax.set_title(feat, fontsize=FIGURE_FONT_SIZE)
        ax.set_xlabel(f"Fenêtres {hour_col}", fontsize=FIGURE_FONT_SIZE - 1)
        ax.set_ylabel("")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        title or f"Distribution par fenêtres d'{hour_col} ({n_windows} bins)",
        fontsize=FIGURE_FONT_SIZE + 2,
        fontweight="bold",
    )
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_boxplots_by_pump_id_hour_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    pump_col: str = "Pump_ID",
    hour_col: str = "operational_hours",
    label_col: str = "maintenance_required",
    n_windows: int = 4,
    label_name: str = "Maintenance",
    save_dir: Path | None = None,
    title_prefix: str | None = None,
) -> list[plt.Figure]:
    """Boxplots par Pump_ID × fenêtres d'heures opérationnelles, colorés par label.

    Pour chaque feature : une figure avec n_windows sous-graphes (un par fenêtre
    temporelle). Dans chaque sous-graphe, l'axe x est le Pump_ID et la couleur
    (hue) distingue les deux états du label (0=Normal, 1=Faulty/Maintenance).

    Révèle si certaines pompes dérivent différemment selon la période opérationnelle.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
    pump_col : str
        Colonne identifiant la pompe (ex. "Pump_ID").
    hour_col : str
        Colonne contenant les heures opérationnelles.
    label_col : str
    n_windows : int
        Nombre de fenêtres temporelles (bins équidistants).
    label_name : str
    save_dir : Path | None
        Répertoire de destination. Un fichier PNG par feature est créé.
    title_prefix : str | None

    Returns
    -------
    list[plt.Figure]
        Une figure par feature.
    """
    df = df.copy()

    # Discrétisation des heures en n_windows bins
    hour_min = df[hour_col].min()
    hour_max = df[hour_col].max()
    bin_edges = np.linspace(hour_min, hour_max, n_windows + 1)
    bin_labels = [
        f"[{int(bin_edges[i])}-{int(bin_edges[i + 1])}h]" for i in range(n_windows)
    ]
    df["_hour_window"] = pd.cut(
        df[hour_col], bins=bin_edges, labels=bin_labels, include_lowest=True
    )
    df[label_col] = df[label_col].astype(str)

    label_vals = sorted(df[label_col].unique())
    palette = {str(v): COLORS_LABEL.get(int(v), "#999999") for v in label_vals}
    hue_order = [str(v) for v in sorted(int(v) for v in label_vals)]

    from matplotlib.patches import Patch  # local import — visualisation uniquement

    legend_handles = [
        Patch(color=COLORS_LABEL.get(int(v), "#999999"), label=f"{'Normal' if int(v) == 0 else label_name} ({v})")
        for v in hue_order
    ]

    figs: list[plt.Figure] = []

    for feat in feature_cols:
        fig, axes = plt.subplots(
            n_windows, 1,
            figsize=(max(10, len(df[pump_col].unique()) * 0.8), 4 * n_windows),
            sharex=False,
        )
        if n_windows == 1:
            axes = [axes]

        for ax, window_label in zip(axes, bin_labels):
            plot_df = df[df["_hour_window"] == window_label][[pump_col, feat, label_col]].dropna()

            if _HAS_SEABORN and not plot_df.empty:
                pump_order = sorted(plot_df[pump_col].unique())
                sns.boxplot(
                    data=plot_df,
                    x=pump_col,
                    y=feat,
                    hue=label_col,
                    hue_order=hue_order,
                    palette=palette,
                    order=pump_order,
                    ax=ax,
                    linewidth=1.0,
                    flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
                    legend=False,
                )
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")

            ax.set_title(window_label, fontsize=FIGURE_FONT_SIZE)
            ax.set_xlabel(pump_col, fontsize=FIGURE_FONT_SIZE - 1)
            ax.set_ylabel(feat, fontsize=FIGURE_FONT_SIZE - 1)
            ax.grid(True, axis="y", alpha=0.3)
            ax.tick_params(axis="x", rotation=30)

        fig.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=FIGURE_FONT_SIZE - 1,
            title=label_name,
            title_fontsize=FIGURE_FONT_SIZE - 1,
        )
        prefix = title_prefix or f"Boxplots {feat} par {pump_col} × fenêtre horaire"
        fig.suptitle(
            f"{prefix} — {feat}",
            fontsize=FIGURE_FONT_SIZE + 2,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 0.88, 1])

        if save_dir is not None:
            save_path = Path(save_dir) / f"{feat}.png"
            _save_fig(fig, save_path)

        figs.append(fig)

    return figs


def plot_fault_rate_heatmap_pump(
    df: pd.DataFrame,
    pump_col: str = "Pump_ID",
    hour_col: str = "Operational_Hours",
    label_col: str = "Maintenance_Flag",
    n_windows: int = 5,
    save_path: Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Heatmap du taux de panne par Pump_ID × fenêtres d'Operational_Hours.

    Rows = Pump_ID, Cols = fenêtres temporelles.
    Valeur = proportion d'échantillons avec label=1.
    Permet d'identifier les pompes défaillantes et les périodes critiques.

    Parameters
    ----------
    df : pd.DataFrame
    pump_col : str
    hour_col : str
    label_col : str
    n_windows : int
    save_path : Path | None
    title : str | None

    Returns
    -------
    plt.Figure
    """
    df = df.copy()
    hour_min = df[hour_col].min()
    hour_max = df[hour_col].max()
    bin_edges = np.linspace(hour_min, hour_max, n_windows + 1)
    bin_labels = [
        f"[{int(bin_edges[i])}-{int(bin_edges[i+1])}]" for i in range(n_windows)
    ]
    df["_hour_window"] = pd.cut(
        df[hour_col], bins=bin_edges, labels=bin_labels, include_lowest=True
    )

    pivot = (
        df.groupby([pump_col, "_hour_window"], observed=True)[label_col]
        .mean()
        .unstack("_hour_window")
        .reindex(columns=bin_labels)
    )

    fig, ax = plt.subplots(figsize=(max(8, n_windows * 1.8), max(4, len(pivot) * 0.9)))

    if _HAS_SEABORN:
        sns.heatmap(
            pivot,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn_r",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Taux de panne", "shrink": 0.8},
        )
    else:
        im = ax.imshow(pivot.values, cmap="RdYlGn_r", vmin=0.0, vmax=1.0, aspect="auto")
        fig.colorbar(im, ax=ax, label="Taux de panne", shrink=0.8)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=FIGURE_FONT_SIZE - 2, color="black")
        ax.set_xticks(range(n_windows))
        ax.set_xticklabels(bin_labels, rotation=30, fontsize=FIGURE_FONT_SIZE - 2)
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels([str(p) for p in pivot.index], fontsize=FIGURE_FONT_SIZE - 1)

    ax.set_xlabel(f"Fenêtres {hour_col}", fontsize=FIGURE_FONT_SIZE)
    ax.set_ylabel(pump_col, fontsize=FIGURE_FONT_SIZE)
    ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        title or f"Heatmap taux de panne — {pump_col} × {hour_col}",
        fontsize=FIGURE_FONT_SIZE + 2,
        fontweight="bold",
    )
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


# ── Dataset 2 — Equipment : analyse par equipment et location ────────────────


def plot_boxplots_by_equipment_location(
    df: pd.DataFrame,
    feature_cols: list[str],
    equipment_col: str = "equipment",
    location_col: str = "location",
    label_col: str = "faulty",
    label_name: str = "Faulty",
    save_path: Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Boxplots croisés equipment × location, avec hue = label.

    Grille : lignes = equipment type, colonnes = feature.
    Dans chaque cellule : boxplot par location, coloré par label.
    Révèle les variations inter-site au sein de chaque type d'équipement.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
    equipment_col : str
    location_col : str
    label_col : str
    label_name : str
    save_path : Path | None
    title : str | None

    Returns
    -------
    plt.Figure
    """
    equipment_types = sorted(df[equipment_col].unique())
    n_eq = len(equipment_types)
    n_feats = len(feature_cols)
    label_vals = sorted(df[label_col].unique())
    palette = {str(v): COLORS_LABEL.get(int(v), "#999999") for v in label_vals}

    fig, axes = plt.subplots(
        n_eq, n_feats, figsize=(max(5 * n_feats, 16), 3.5 * n_eq), squeeze=False
    )

    for row_idx, eq in enumerate(equipment_types):
        df_eq = df[df[equipment_col] == eq]
        locations = sorted(df_eq[location_col].unique())

        for col_idx, feat in enumerate(feature_cols):
            ax = axes[row_idx][col_idx]

            if _HAS_SEABORN:
                plot_df = df_eq[[location_col, feat, label_col]].copy()
                plot_df[label_col] = plot_df[label_col].astype(str)
                sns.boxplot(
                    data=plot_df,
                    x=location_col,
                    y=feat,
                    hue=label_col,
                    palette=palette,
                    ax=ax,
                    linewidth=0.8,
                    flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
                    order=locations,
                )
                if row_idx == 0 and col_idx == n_feats - 1:
                    from matplotlib.patches import Patch  # local import — visualisation uniquement
                    _legend_handles = [
                        Patch(color=COLORS_LABEL[0], label="Normal (0)"),
                        Patch(color=COLORS_LABEL[1], label=f"{label_name} (1)"),
                    ]
                    ax.legend(
                        handles=_legend_handles,
                        title=label_name,
                        fontsize=FIGURE_FONT_SIZE - 2,
                        title_fontsize=FIGURE_FONT_SIZE - 2,
                    )
                else:
                    ax.get_legend().remove() if ax.get_legend() else None
            else:
                offsets = np.linspace(-0.2, 0.2, len(label_vals))
                for v, offset in zip(label_vals, offsets):
                    data_per_loc = [
                        df_eq[(df_eq[location_col] == loc) & (df_eq[label_col] == v)][feat].dropna().values
                        for loc in locations
                    ]
                    positions = np.arange(len(locations)) + offset
                    bp = ax.boxplot(
                        data_per_loc,
                        positions=positions,
                        widths=0.3,
                        patch_artist=True,
                        medianprops={"color": "black", "linewidth": 1.2},
                        manage_ticks=False,
                    )
                    color = COLORS_LABEL.get(int(v), "#999999")
                    for patch in bp["boxes"]:
                        patch.set_facecolor(color)
                        patch.set_alpha(0.75)
                ax.set_xticks(range(len(locations)))
                ax.set_xticklabels(locations, fontsize=FIGURE_FONT_SIZE - 2, rotation=30)

            if row_idx == 0:
                ax.set_title(feat, fontsize=FIGURE_FONT_SIZE)
            if col_idx == 0:
                ax.set_ylabel(str(eq), fontsize=FIGURE_FONT_SIZE, fontweight="bold")
            else:
                ax.set_ylabel("")
            ax.set_xlabel(location_col if row_idx == n_eq - 1 else "")
            ax.grid(True, axis="y", alpha=0.3)
            ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        title or f"Boxplots par {equipment_col} × {location_col}",
        fontsize=FIGURE_FONT_SIZE + 2,
        fontweight="bold",
    )
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_violin_by_location(
    df: pd.DataFrame,
    feature_cols: list[str],
    location_col: str = "location",
    label_col: str = "faulty",
    label_name: str = "Faulty",
    save_path: Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Violin plots par location, avec hue = label (normal vs faulty).

    Ligne par feature, colonnes = toutes locations. Montre les variations
    de distribution entre sites géographiques.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
    location_col : str
    label_col : str
    label_name : str
    save_path : Path | None
    title : str | None

    Returns
    -------
    plt.Figure
    """
    n = len(feature_cols)
    locations = sorted(df[location_col].unique())
    label_vals = sorted(df[label_col].unique())
    palette = {str(v): COLORS_LABEL.get(int(v), "#999999") for v in label_vals}

    fig, axes = plt.subplots(1, n, figsize=(max(5 * n, 14), 5), squeeze=False)
    axes = axes[0]

    for ax, feat in zip(axes, feature_cols):
        if _HAS_SEABORN:
            plot_df = df[[location_col, feat, label_col]].copy()
            plot_df[label_col] = plot_df[label_col].astype(str)
            sns.violinplot(
                data=plot_df,
                x=location_col,
                y=feat,
                hue=label_col,
                palette=palette,
                ax=ax,
                inner="quartile",
                linewidth=0.8,
                order=locations,
            )
            # Annotation : nombre de points par location
            for i, loc in enumerate(locations):
                n_loc = len(df[df[location_col] == loc])
                ax.text(
                    i,
                    ax.get_ylim()[0],
                    f"n={n_loc}",
                    ha="center",
                    va="bottom",
                    fontsize=FIGURE_FONT_SIZE - 3,
                    color="grey",
                )
            from matplotlib.patches import Patch  # local import — visualisation uniquement
            _legend_handles = [
                Patch(color=COLORS_LABEL[0], label="Normal (0)"),
                Patch(color=COLORS_LABEL[1], label=f"{label_name} (1)"),
            ]
            ax.legend(
                handles=_legend_handles,
                title=label_name,
                fontsize=FIGURE_FONT_SIZE - 2,
                title_fontsize=FIGURE_FONT_SIZE - 2,
            )
        else:
            for idx, loc in enumerate(locations):
                data_by_label = [
                    df[(df[location_col] == loc) & (df[label_col] == v)][feat].dropna().values
                    for v in label_vals
                ]
                parts = ax.violinplot(
                    data_by_label,
                    positions=[idx - 0.15, idx + 0.15],
                    widths=0.25,
                    showmedians=True,
                )
                for pc, v in zip(parts["bodies"], label_vals):
                    pc.set_facecolor(COLORS_LABEL.get(int(v), "#999999"))
                    pc.set_alpha(0.7)
            ax.set_xticks(range(len(locations)))
            ax.set_xticklabels(locations, fontsize=FIGURE_FONT_SIZE - 2)

        ax.set_title(feat, fontsize=FIGURE_FONT_SIZE)
        ax.set_xlabel(location_col, fontsize=FIGURE_FONT_SIZE - 1)
        ax.set_ylabel("")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        title or f"Violin plots par {location_col}",
        fontsize=FIGURE_FONT_SIZE + 2,
        fontweight="bold",
    )
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_fault_rate_heatmap_equipment(
    df: pd.DataFrame,
    equipment_col: str = "equipment",
    location_col: str = "location",
    label_col: str = "faulty",
    save_path: Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Heatmap du taux de panne par equipment × location.

    Rows = type d'équipement, Cols = site géographique.
    Valeur = proportion d'échantillons avec label=1.
    Révèle les combinaisons équipement/site à risque élevé.

    Parameters
    ----------
    df : pd.DataFrame
    equipment_col : str
    location_col : str
    label_col : str
    save_path : Path | None
    title : str | None

    Returns
    -------
    plt.Figure
    """
    pivot = (
        df.groupby([equipment_col, location_col])[label_col]
        .mean()
        .unstack(location_col)
    )
    locations = sorted(df[location_col].unique())
    pivot = pivot.reindex(columns=locations)

    n_eq = len(pivot)
    n_loc = len(pivot.columns)
    fig, ax = plt.subplots(figsize=(max(8, n_loc * 1.5), max(4, n_eq * 1.2)))

    if _HAS_SEABORN:
        # Annotation en pourcentage — compatible pandas < 2.1 (applymap) et >= 2.1 (map)
        _fmt = lambda v: f"{v:.1%}" if not np.isnan(v) else "—"
        annot = pivot.apply(lambda col: col.map(_fmt))
        sns.heatmap(
            pivot,
            ax=ax,
            annot=annot,
            fmt="",
            cmap="RdYlGn_r",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Taux de panne", "shrink": 0.8},
        )
    else:
        im = ax.imshow(pivot.values, cmap="RdYlGn_r", vmin=0.0, vmax=1.0, aspect="auto")
        fig.colorbar(im, ax=ax, label="Taux de panne", shrink=0.8)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                txt = f"{val:.1%}" if not np.isnan(val) else "—"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=FIGURE_FONT_SIZE - 2, color="black")
        ax.set_xticks(range(n_loc))
        ax.set_xticklabels(list(pivot.columns), rotation=30, fontsize=FIGURE_FONT_SIZE - 2)
        ax.set_yticks(range(n_eq))
        ax.set_yticklabels(list(pivot.index), fontsize=FIGURE_FONT_SIZE - 1)

    ax.set_xlabel(location_col, fontsize=FIGURE_FONT_SIZE)
    ax.set_ylabel(equipment_col, fontsize=FIGURE_FONT_SIZE)
    ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        title or f"Heatmap taux de panne — {equipment_col} × {location_col}",
        fontsize=FIGURE_FONT_SIZE + 2,
        fontweight="bold",
    )
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_correlation_by_equipment(
    df: pd.DataFrame,
    feature_cols: list[str],
    equipment_col: str = "equipment",
    save_path: Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Heatmaps de corrélation entre features, une par type d'équipement.

    Permet de vérifier si la structure de corrélation varie entre les domaines
    (domain shift structurel) — information utile pour le scénario CL.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
    equipment_col : str
    save_path : Path | None
    title : str | None

    Returns
    -------
    plt.Figure
    """
    equipment_types = sorted(df[equipment_col].unique())
    n_eq = len(equipment_types)
    n_feats = len(feature_cols)

    fig, axes = plt.subplots(1, n_eq, figsize=(max(4 * n_eq, 12), max(4, n_feats)), squeeze=False)
    axes = axes[0]

    for ax, eq in zip(axes, equipment_types):
        df_eq = df[df[equipment_col] == eq][feature_cols].dropna()
        corr = df_eq.corr()

        if _HAS_SEABORN:
            sns.heatmap(
                corr,
                ax=ax,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                vmin=-1,
                vmax=1,
                linewidths=0.3,
                linecolor="white",
                cbar=(ax is axes[-1]),  # colorbar uniquement sur la dernière
                annot_kws={"size": FIGURE_FONT_SIZE - 2},
            )
        else:
            im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
            if ax is axes[-1]:
                fig.colorbar(im, ax=ax, label="Corrélation", shrink=0.8)
            for i in range(n_feats):
                for j in range(n_feats):
                    ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                            fontsize=FIGURE_FONT_SIZE - 3, color="black")
            ax.set_xticks(range(n_feats))
            ax.set_xticklabels(feature_cols, rotation=30, fontsize=FIGURE_FONT_SIZE - 2)
            ax.set_yticks(range(n_feats))
            ax.set_yticklabels(feature_cols, fontsize=FIGURE_FONT_SIZE - 2)

        ax.set_title(str(eq), fontsize=FIGURE_FONT_SIZE, fontweight="bold")

    fig.suptitle(
        title or f"Matrices de corrélation par {equipment_col}",
        fontsize=FIGURE_FONT_SIZE + 2,
        fontweight="bold",
    )
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_sorted_scatter_by_label(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    label_name: str = "Label",
    alpha: float = 0.5,
    s: float = 6,
    title: str | None = None,
    save_path: "Path | None" = None,
) -> plt.Figure:
    """Sorted scatter : valeurs triées croissantes, colorées par label.

    Pour chaque feature, trie les valeurs dans l'ordre croissant (X = rang,
    Y = valeur) et colore chaque point selon son label. Révèle dans quelle
    plage de valeurs les échantillons faulty se concentrent.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
    label_col : str
    label_name : str
    alpha : float
        Transparence des points.
    s : float
        Taille des points.
    title : str | None
    save_path : Path | None

    Returns
    -------
    plt.Figure
    """
    from matplotlib.patches import Patch  # local import — visualisation uniquement

    n = len(feature_cols)
    fig, axes = plt.subplots(1, n, figsize=(max(5 * n, 16), 5))
    if n == 1:
        axes = [axes]

    label_vals = sorted(df[label_col].dropna().unique())

    for ax, feat in zip(axes, feature_cols):
        valid = df[[feat, label_col]].dropna()
        sorted_idx = valid[feat].argsort().values
        sorted_vals = valid[feat].values[sorted_idx]
        sorted_labels = valid[label_col].values[sorted_idx].astype(int)
        colors = [COLORS_LABEL.get(v, "#999999") for v in sorted_labels]

        ax.scatter(
            range(len(sorted_vals)),
            sorted_vals,
            c=colors,
            alpha=alpha,
            s=s,
            linewidths=0,
        )
        ax.set_title(feat, fontsize=FIGURE_FONT_SIZE)
        ax.set_xlabel("Rang (ordre croissant)", fontsize=FIGURE_FONT_SIZE - 2)
        ax.set_ylabel(feat if feat == feature_cols[0] else "", fontsize=FIGURE_FONT_SIZE - 1)
        ax.grid(True, alpha=0.25)

    legend_handles = [
        Patch(facecolor=COLORS_LABEL.get(int(v), "#999999"),
              label=f"{label_name}={int(v)} ({'Normal' if int(v) == 0 else 'Faulty'})")
        for v in label_vals
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=FIGURE_FONT_SIZE - 1,
        framealpha=0.9,
    )
    fig.suptitle(
        title or f"Valeurs triées par {label_name}",
        fontsize=FIGURE_FONT_SIZE + 2,
        fontweight="bold",
    )
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_pairplot_by_equipment(
    df: pd.DataFrame,
    feature_cols: list[str],
    equipment_col: str = "equipment",
    label_col: str = "faulty",
    sample_n: int = 1500,
    save_path: Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Pairplot coloré par type d'équipement, marqueurs par label.

    Diagonale : KDE par equipment. Hors-diagonale : scatter.
    Révèle les sous-espaces de features distincts entre domaines CL.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
    equipment_col : str
    label_col : str
    sample_n : int
        Nombre max d'échantillons (subsample pour rapidité).
    save_path : Path | None
    title : str | None

    Returns
    -------
    plt.Figure
    """
    from itertools import cycle

    COLORS_DOMAIN = ["#2196F3", "#FF9800", "#9C27B0", "#00BCD4", "#8BC34A"]
    MARKERS_DOMAIN = ["o", "s", "D", "^", "v"]

    plot_df = df[feature_cols + [equipment_col, label_col]].dropna()
    if len(plot_df) > sample_n:
        plot_df = plot_df.sample(n=sample_n, random_state=42)

    equipment_types = sorted(plot_df[equipment_col].unique())
    palette = dict(zip(equipment_types, cycle(COLORS_DOMAIN)))
    markers = dict(zip(equipment_types, cycle(MARKERS_DOMAIN)))

    if _HAS_SEABORN:
        pg = sns.pairplot(
            plot_df,
            vars=feature_cols,
            hue=equipment_col,
            palette=palette,
            markers=[markers[eq] for eq in equipment_types],
            diag_kind="kde",
            plot_kws={"alpha": 0.35, "s": 10, "linewidths": 0},
            diag_kws={"fill": True, "alpha": 0.3},
            corner=False,
        )
        pg.fig.suptitle(
            title or f"Pairplot par {equipment_col}",
            fontsize=FIGURE_FONT_SIZE + 2,
            fontweight="bold",
            y=1.02,
        )
        _save_fig(pg.fig, save_path)
        return pg.fig
    else:
        n = len(feature_cols)
        fig, axes = plt.subplots(n, n, figsize=(max(3 * n, 10), max(3 * n, 10)))
        for i, feat_y in enumerate(feature_cols):
            for j, feat_x in enumerate(feature_cols):
                ax = axes[i, j]
                if i == j:
                    for eq in equipment_types:
                        subset = plot_df[plot_df[equipment_col] == eq][feat_x].dropna()
                        ax.hist(subset, bins=20, alpha=0.35, color=palette[eq], density=True, label=eq)
                else:
                    for eq in equipment_types:
                        mask = plot_df[equipment_col] == eq
                        ax.scatter(
                            plot_df.loc[mask, feat_x],
                            plot_df.loc[mask, feat_y],
                            c=palette[eq],
                            marker=markers[eq],
                            alpha=0.35,
                            s=8,
                            linewidths=0,
                        )
                if i == n - 1:
                    ax.set_xlabel(feat_x, fontsize=8)
                if j == 0:
                    ax.set_ylabel(feat_y, fontsize=8)
                ax.tick_params(labelsize=7)

        handles = [
            plt.Line2D([0], [0], marker=markers[eq], color="w",
                       markerfacecolor=palette[eq], markersize=7, label=eq)
            for eq in equipment_types
        ]
        fig.legend(handles=handles, loc="upper right", fontsize=FIGURE_FONT_SIZE - 1)
        fig.suptitle(
            title or f"Pairplot par {equipment_col}",
            fontsize=FIGURE_FONT_SIZE + 2,
            fontweight="bold",
        )
        fig.tight_layout()
        _save_fig(fig, save_path)
        return fig
