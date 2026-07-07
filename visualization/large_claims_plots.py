"""
=========================================================
LARGE CLAIMS PLOTS
---------------------------------------------------------
Visualisations actuarielle :
- Mean Excess Function
- Split attritionnel / atypique
- Impact écrêtement
- Décomposition des primes
- Comparaison distributions
=========================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from visualization.config import (
    FIGSIZE,
    TITLE_SIZE,
    LABEL_SIZE,
    TICK_SIZE,
    LEGEND_SIZE,
    GRID_ALPHA,
    STYLE,
    PALETTE,
    COLOR_PRIMARY,
    COLOR_DANGER,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_INFO,
    COLOR_NEUTRAL
)

# STYLE
sns.set_theme(
    style=STYLE,
    palette=PALETTE
)

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.facecolor": "#F8F9FB",
    "axes.grid": True,
    "grid.alpha": GRID_ALPHA,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# MEAN EXCESS FUNCTION
def plot_mean_excess_function(
    mef_df: pd.DataFrame,
    threshold: float
):
    """
    Plot empirical Mean Excess Function.
    """

    fig, ax = plt.subplots(figsize=FIGSIZE)

    valid = mef_df["mean_excess"].notna()

    ax.plot(
        mef_df.loc[valid, "threshold"],
        mef_df.loc[valid, "mean_excess"],
        linewidth=2.5,
        color=COLOR_PRIMARY,
        label="MEF empirique"
    )

    ax.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        color=COLOR_DANGER,
        label=f"M1 = {threshold:,.0f} €"
    )

    ax.set_title(
        "Mean Excess Function",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Seuil u (€)",
        fontsize=LABEL_SIZE
    )

    ax.set_ylabel(
        "E[X - u | X > u]",
        fontsize=LABEL_SIZE
    )

    ax.tick_params(
        axis="both",
        labelsize=TICK_SIZE
    )

    ax.legend(fontsize=LEGEND_SIZE)

    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, _: f"{x/1000:.0f}k"
        )
    )

    plt.tight_layout()
    plt.savefig("figures/mean_excess_function.png", dpi=150, bbox_inches="tight")
    plt.show()


# CLAIM DISTRIBUTION SPLIT
def plot_claims_distribution_split(
    data: pd.DataFrame,
    threshold: float,
    claim_amount_col: str = "ClaimAmount",
    bins_attritional: int = 60,
    bins_large: int = 25
):
    values = data[claim_amount_col].dropna()
    mask_attr  = values <= threshold
    mask_large = values > threshold

    attritional = values[mask_attr]
    large       = values[mask_large]

    if len(attritional) == 0 or len(large) == 0:
        raise ValueError(
            f"Split vide : attr={len(attritional)}, "
            f"large={len(large)}"
        )

    # ── FIX 1 : deux sous-graphes avec leurs propres échelles ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Distribution des sinistres par composante",
                 fontsize=14, fontweight="bold")

    # ── Attritionnels — clip au P99 DE LEUR propre distribution ──
    clip_attrit = attritional.quantile(0.99)   # FIX 2 : clip local
    attrit_plot = attritional.clip(upper=clip_attrit)

    ax1.hist(attrit_plot, bins=bins_attritional,
             density=True, alpha=0.7,
             color=COLOR_PRIMARY, edgecolor="white")
    ax1.axvline(threshold, color=COLOR_NEUTRAL,
                ls="--", lw=2,
                label=f"Seuil M1 = {threshold:,.0f} €")
    ax1.set_title(f"Attritionnels ≤ {threshold:,.0f} €\n"
                  f"(n = {len(attritional):,}  |  "
                  f"moy = {attritional.mean():,.0f} €)",
                  fontweight="bold")
    ax1.set_xlabel("ClaimAmount (€)")
    ax1.set_ylabel("Densité")
    ax1.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k")
    )
    ax1.legend(fontsize=9)

    # ── Atypiques — leur propre axe X ────────────────────────────
    clip_large = large.quantile(0.99)          # FIX 2 : clip local
    large_plot = large.clip(upper=clip_large)

    ax2.hist(large_plot, bins=bins_large,
             density=True, alpha=0.7,
             color=COLOR_DANGER, edgecolor="white")
    ax2.axvline(threshold, color=COLOR_NEUTRAL,
                ls="--", lw=2,
                label=f"Seuil M1 = {threshold:,.0f} €")
    ax2.set_title(f"Atypiques > {threshold:,.0f} €\n"
                  f"(n = {len(large):,}  |  "
                  f"moy = {large.mean():,.0f} €)",
                  fontweight="bold")
    ax2.set_xlabel("ClaimAmount (€)")
    ax2.set_ylabel("Densité")
    ax2.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k")
    )
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("figures/claims_distribution_split.png", dpi=150, bbox_inches="tight")
    plt.show()


# CAPPED VS UNCAPPED
def plot_capped_vs_uncapped_severity(
    data: pd.DataFrame,
    uncapped_col: str = "severity_mean",
    capped_col: str = "severity_capped",
    quantile_clip: float = 0.98,
    bins: int = 50
):
    """
    Compare capped vs uncapped severity.
    """

    fig, ax = plt.subplots(figsize=FIGSIZE)

    uncapped = data[uncapped_col]
    capped = data[capped_col]

    clip_uncapped = uncapped.quantile(
        quantile_clip
    )

    clip_capped = capped.quantile(
        quantile_clip
    )

    ax.hist(
        uncapped.clip(upper=clip_uncapped),
        bins=bins,
        density=True,
        alpha=0.5,
        color=COLOR_WARNING,
        edgecolor="white",
        label=f"Avant écrêtement\n(moy={uncapped.mean():.0f}€)"
    )

    ax.hist(
        capped.clip(upper=clip_capped),
        bins=bins,
        density=True,
        alpha=0.5,
        color=COLOR_SUCCESS,
        edgecolor="white",
        label=f"Après écrêtement\n(moy={capped.mean():.0f}€)"
    )

    ax.axvline(
        uncapped.mean(),
        color=COLOR_WARNING,
        linestyle="--",
        linewidth=2
    )

    ax.axvline(
        capped.mean(),
        color=COLOR_SUCCESS,
        linestyle="--",
        linewidth=2
    )

    ax.set_title(
        "Impact de l'écrêtement",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Sévérité moyenne (€)",
        fontsize=LABEL_SIZE
    )

    ax.set_ylabel(
        "Densité",
        fontsize=LABEL_SIZE
    )

    ax.tick_params(
        axis="both",
        labelsize=TICK_SIZE
    )

    ax.legend(fontsize=LEGEND_SIZE)

    plt.tight_layout()
    plt.savefig("figures/capped_vs_uncapped_severity.png", dpi=150, bbox_inches="tight")
    plt.show()



# PREMIUM DECOMPOSITION


def plot_premium_decomposition(
    data: pd.DataFrame,
    group_col: str,
    attritional_col: str = "prime_attritional",
    large_col: str = "prime_large"
):
    """
    Plot premium decomposition by segment.
    """

    grouped = (
        data.groupby(group_col, observed=True)
        .agg(
            attritional=(attritional_col, "mean"),
            large=(large_col, "mean")
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)

    x = np.arange(len(grouped))
    width = 0.6

    ax.bar(
        x,
        grouped["attritional"],
        width=width,
        color=COLOR_PRIMARY,
        alpha=0.9,
        edgecolor="white",
        label="Attritionnelle"
    )

    ax.bar(
        x,
        grouped["large"],
        width=width,
        bottom=grouped["attritional"],
        color=COLOR_DANGER,
        alpha=0.9,
        edgecolor="white",
        label="Atypique"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        grouped[group_col].astype(str),
        rotation=25,
        fontsize=TICK_SIZE
    )

    ax.set_title(
        f"Décomposition de prime par {group_col}",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Prime pure (€)",
        fontsize=LABEL_SIZE
    )

    ax.legend(fontsize=LEGEND_SIZE)

    plt.tight_layout()
    plt.savefig("figures/premium_decomposition.png", dpi=150, bbox_inches="tight")
    plt.show()



# PREMIUM DISTRIBUTION COMPARISON
def plot_two_component_premium_distribution(
    data: pd.DataFrame,
    baseline_col: str = "pure_premium",
    two_component_col: str = "pure_premium_2comp",
    quantile_clip: float = 0.99,
    bins: int = 60
):
    """
    Compare premium distributions.
    """

    fig, ax = plt.subplots(figsize=FIGSIZE)

    clip_value = data[two_component_col].quantile(
        quantile_clip
    )

    baseline = data[baseline_col].clip(
        upper=clip_value
    )

    two_component = data[two_component_col].clip(
        upper=clip_value
    )

    ax.hist(
        baseline,
        bins=bins,
        density=True,
        alpha=0.5,
        color=COLOR_NEUTRAL,
        edgecolor="white",
        label=(
            "Modèle classique\n"
            f"(moy={baseline.mean():.1f}€)"
        )
    )

    ax.hist(
        two_component,
        bins=bins,
        density=True,
        alpha=0.5,
        color=COLOR_INFO,
        edgecolor="white",
        label=(
            "Deux composantes\n"
            f"(moy={two_component.mean():.1f}€)"
        )
    )

    ax.axvline(
        baseline.mean(),
        color=COLOR_NEUTRAL,
        linestyle="--",
        linewidth=2
    )

    ax.axvline(
        two_component.mean(),
        color=COLOR_INFO,
        linestyle="--",
        linewidth=2
    )

    ax.set_title(
        "Distribution des primes",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Prime pure (€)",
        fontsize=LABEL_SIZE
    )

    ax.set_ylabel(
        "Densité",
        fontsize=LABEL_SIZE
    )

    ax.tick_params(
        axis="both",
        labelsize=TICK_SIZE
    )

    ax.legend(fontsize=LEGEND_SIZE)

    plt.tight_layout()
    plt.savefig("figures/two_component_premium_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()



# FULL LARGE CLAIMS REPORT
def run_large_claims_report(
    train_sev: pd.DataFrame,
    test_freq: pd.DataFrame,
    threshold_info: dict,
    group_col: str = "DriverAge_class",
    claim_amount: str = "ClaimAmount_total"
):
    """
    Full actuarial visualization report.
    """

    threshold = threshold_info["threshold"]

    
    # MEF
    plot_mean_excess_function(
        mef_df=threshold_info["mef"],
        threshold=threshold
    )

    
    # Distribution split
    plot_claims_distribution_split(
        data=train_sev,
        threshold=threshold,
        claim_amount_col=claim_amount
    )

    
    # Capped vs uncapped
    if (
        "severity_mean" in train_sev.columns
        and
        "severity_capped" in train_sev.columns
    ):

        plot_capped_vs_uncapped_severity(
            train_sev
        )

    
    # Premium decomposition
    plot_premium_decomposition(
        data=test_freq,
        group_col=group_col
    )

    
    # Distribution comparison
    plot_two_component_premium_distribution(
        data=test_freq
    )