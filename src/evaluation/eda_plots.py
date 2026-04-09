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
