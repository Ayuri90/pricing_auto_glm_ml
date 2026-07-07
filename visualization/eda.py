import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from visualization.config import  (
    MAIN_PALETTE,
    SEQUENTIAL_PALETTE,
    DIVERGING_PALETTE,
    PASTEL_PALETTE,
    MEAN_COLOR,
    MEDIAN_COLOR,
    OUTLIER_COLOR
)


# QUALITATIVE VARIABLES
def plot_qualitative_distributions(
    data,
    variables,
    seuil=5,
    figsize=(12, 5),
    save = ""
):
    """
    Affiche les distributions des variables qualitatives
    """

    n = len(variables)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize[0] * ncols, figsize[1] * nrows)
    )

    axes = axes.flatten()

    for ax, var in zip(axes, variables):

        order = data[var].value_counts().index

        percentages = (
            data[var]
            .value_counts(normalize=True)
            .mul(100)
            .sort_values(ascending=False)
        )

        sns.barplot(
            x=percentages.index,
            y=percentages.values,
            hue=percentages.index,
            palette=SEQUENTIAL_PALETTE,
            legend=False,
            ax=ax,
            order=order
        )

        ax.axhline(
            seuil,
            color=MEAN_COLOR,
            linestyle="--",
            linewidth=1.8,
            label=f"Seuil ({seuil}%)"
        )

        ax.set_title(
            f"Distribution de {var}",
            pad=12
        )

        ax.set_xlabel("")
        ax.set_ylabel("Pourcentage (%)")

        ax.tick_params(
            axis="x",
            rotation=30
        )

        for i, value in enumerate(percentages.values):
            ax.text(
                i,
                value + 0.4,
                f"{value:.1f}%",
                ha="center",
                fontsize=9
            )

        ax.legend(frameon=True)

    for ax in axes[n:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.savefig(f"figures/qualitative_distributions{save}.png", dpi=150, bbox_inches="tight")
    plt.show()


# CLAIM FREQUENCY BY CATEGORY
def freq_exposition(
    data,
    variables,
    exposure_col="Exposure",
    claim_col="ClaimNb",
    sort=True,
    figsize=(10, 5),
    save = ""
):
    """
    Affiche la fréquence de sinistre par modalité
    """

    n = len(variables)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize[0] * ncols, figsize[1] * nrows)
    )

    axes = axes.flatten()

    results = {}

    for ax, variable in zip(axes, variables):

        stats_df = (
            data.groupby(variable, observed=True)
            .agg(
                sinistres=(claim_col, "sum"),
                exposition=(exposure_col, "sum")
            )
            .assign(
                freq=lambda x:
                x["sinistres"] / x["exposition"]
            )
        )

        if sort:
            stats_df = stats_df.sort_values("freq")

        colors = sns.color_palette(
            SEQUENTIAL_PALETTE,
            len(stats_df)
        )

        bars = ax.bar(
            stats_df.index.astype(str),
            stats_df["freq"],
            color=colors,
            edgecolor="white",
            linewidth=1
        )

        mean_freq = stats_df["freq"].mean()

        ax.axhline(
            mean_freq,
            color=MEAN_COLOR,
            linestyle="--",
            linewidth=1.8,
            label=f"Moyenne : {mean_freq:.3f}"
        )

        ax.bar_label(
            bars,
            fmt="%.3f",
            fontsize=8,
            padding=3
        )

        ax.set_title(
            f"Fréquence de sinistre par {variable}",
            pad=12
        )

        ax.set_xlabel("")
        ax.set_ylabel("Fréquence")

        plt.setp(
            ax.get_xticklabels(),
            rotation=30,
            ha="right"
        )

        ax.legend(frameon=True)

        results[variable] = stats_df

    # Supprime les axes inutilisés
    for ax in axes[n:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.savefig(f"figures/claim_frequency{save}.png", dpi=150, bbox_inches="tight")
    plt.show()

    return results


# QUANTITATIVE DISTRIBUTIONS
def plot_quantitative_distributions(
    data,
    variables,
    bins=40
):
    """
    Affiche les distributions des variables quantitatives.
    """

    n = len(variables)

    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(16, 5 * nrows)
    )

    axes = axes.flatten()

    colors = sns.color_palette(
        MAIN_PALETTE,
        n
    )

    for ax, var, color in zip(axes, variables, colors):
        df = data[var].dropna()

        # ClaimNb
        if var == "ClaimNb":
            counts = df.value_counts().sort_index()
            ax.bar(
                counts.index,
                counts.values / len(df),
                color=color,
                edgecolor="white",
                linewidth=1
            )
            ax.set_ylabel("Fréquence relative")

        # Continuous variables
        else:
            sns.histplot(
                df,
                bins=bins,
                kde=True,
                color=color,
                alpha=0.65,
                edgecolor="white",
                linewidth=0.7,
                ax=ax
            )
            ax.set_ylabel("Effectif")

        # Mean / median
        ax.axvline(
            df.mean(),
            color=MEAN_COLOR,
            linestyle="--",
            linewidth=2,
            label=f"Moyenne : {df.mean():.2f}"
        )
        ax.axvline(
            df.median(),
            color=MEDIAN_COLOR,
            linestyle=":",
            linewidth=2,
            label=f"Médiane : {df.median():.2f}"
        )
        ax.set_title(
            var,
            pad=10
        )
        ax.set_xlabel("")
        ax.legend(frameon=True)

    for ax in axes[n:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.savefig("figures/quantitative_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()


# BOXPLOTS
def plot_boxplots(
    data,
    variables
):
    """
    Affiche les boxplots des variables quantitatives.
    """

    n = len(variables)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(16, 5 * nrows)
    )

    axes = axes.flatten()
    colors = sns.color_palette(
        PASTEL_PALETTE,
        n
    )

    for ax, var, color in zip(axes, variables, colors):
        df = data[var].dropna()
        sns.boxplot(
            y=df,
            ax=ax,
            color=color,
            width=0.45,
            linewidth=1.4,
            flierprops={
                "marker": "o",
                "markerfacecolor": OUTLIER_COLOR,
                "markeredgecolor": OUTLIER_COLOR,
                "markersize": 4,
                "alpha": 0.5
            }
        )

        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)

        iqr = q3 - q1
        n_outliers = (
            (
                (df < q1 - 1.5 * iqr) |
                (df > q3 + 1.5 * iqr)
            )
        ).sum()

        ax.set_title(
            var,
            pad=10
        )

        ax.set_xlabel(
            f"Outliers détectés : {n_outliers}",
            fontsize=9
        )

    for ax in axes[n:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.savefig("figures/boxplots.png", dpi=150, bbox_inches="tight")
    plt.show()


# CORRELATION MATRIX
def plot_correlation_matrix(
    data,
    variables,
    method="pearson"
):
    """
    Affiche la matrice de corrélation.
    """

    corr_matrix = (
        data[variables]
        .corr(method=method)
    )

    mask = np.triu(
        np.ones_like(corr_matrix, dtype=bool)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=DIVERGING_PALETTE,
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 9},
        ax=ax
    )
    ax.set_title(
        f"Matrice de corrélation ({method})",
        pad=14
    )

    plt.tight_layout()
    plt.savefig("figures/correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()


# GLOBAL EDA PIPELINE
def run_eda(
    train_freq,
    qualitative_vars,
    quantitative_vars
):
    """
    Lance l'ensemble des visualisations EDA.
    """

    # Qualitative distributions
    plot_qualitative_distributions(
        train_freq,
        qualitative_vars
    )

    # Claim frequency analysis
    freq_exposition(
        train_freq,
        qualitative_vars
    )

    # Quantitative distributions
    plot_quantitative_distributions(
        train_freq,
        quantitative_vars
    )

    # Boxplots
    plot_boxplots(
        train_freq,
        quantitative_vars
    )

    # Correlation matrix
    plot_correlation_matrix(
        train_freq,
        quantitative_vars
    )