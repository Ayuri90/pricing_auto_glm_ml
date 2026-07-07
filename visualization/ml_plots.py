"""
=========================================================
ML PLOTS
---------------------------------------------------------
Visualisations d'évaluation et d'interprétabilité :
- Lift / calibration par décile
- Courbes de Lorenz (comparaison de modèles)
- Importance des variables (gain)
- Partial Dependence Plots (calcul manuel, compatible
  variables catégorielles)
- Résumé SHAP (TreeSHAP natif LightGBM via
  predict(pred_contrib=True), sans dépendance externe)
=========================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from visualization.config import (
    FIGSIZE,
    TITLE_SIZE,
    LABEL_SIZE,
    TICK_SIZE,
    LEGEND_SIZE,
    COLOR_PRIMARY,
    COLOR_DANGER,
    COLOR_SUCCESS,
    COLOR_INFO,
    COLOR_NEUTRAL,
)

MODEL_COLORS = [
    COLOR_PRIMARY,
    COLOR_DANGER,
    COLOR_SUCCESS,
    COLOR_INFO,
    COLOR_NEUTRAL,
]


# =========================================================
# LIFT / CALIBRATION PAR DECILE
# =========================================================

def plot_lift(lift_df, model_name="model", save=""):
    """
    Barres prime prédite vs coût observé par décile
    de prime prédite (sortie de ml_evaluate.lift_table).
    """

    fig, ax = plt.subplots(figsize=FIGSIZE)

    x = np.arange(len(lift_df))
    width = 0.38

    ax.bar(
        x - width / 2,
        lift_df["pred_mean"],
        width=width,
        color=COLOR_PRIMARY,
        edgecolor="white",
        label="Prime prédite moyenne"
    )

    ax.bar(
        x + width / 2,
        lift_df["obs_mean"],
        width=width,
        color=COLOR_DANGER,
        alpha=0.85,
        edgecolor="white",
        label="Coût observé moyen"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(lift_df["bin"].astype(str))

    ax.set_title(
        f"Lift / calibration par décile — {model_name}",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )
    ax.set_xlabel(
        "Décile de prime prédite",
        fontsize=LABEL_SIZE
    )
    ax.set_ylabel("€", fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)

    plt.tight_layout()
    plt.savefig(
        f"figures/lift_{model_name}{save}.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.show()


# =========================================================
# LORENZ
# =========================================================

def plot_lorenz_curves(curves, save=""):
    """
    Compare les courbes de Lorenz de plusieurs modèles.

    curves : dict {nom: (cum_policies, cum_claims, gini)}
    """

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(
        [0, 1], [0, 1],
        linestyle="--",
        linewidth=1.5,
        color=COLOR_NEUTRAL,
        label="Tarif aléatoire (Gini = 0)"
    )

    for (name, (x, y, g)), color in zip(
        curves.items(), MODEL_COLORS
    ):
        ax.plot(
            x, y,
            linewidth=2.2,
            color=color,
            label=f"{name} (Gini = {g:.3f})"
        )

    ax.set_title(
        "Courbes de Lorenz",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )
    ax.set_xlabel(
        "Part cumulée de polices\n(triées par prime prédite croissante)",
        fontsize=LABEL_SIZE
    )
    ax.set_ylabel(
        "Part cumulée des sinistres",
        fontsize=LABEL_SIZE
    )
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc="upper left")

    plt.tight_layout()
    plt.savefig(
        f"figures/lorenz_curves{save}.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.show()


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def plot_feature_importance(
    lgb_model,
    model_name="model",
    save=""
):
    """
    Importance des variables par gain total
    (lgb_model : LGBMRegressor entraîné).
    """

    booster = lgb_model.booster_

    imp = pd.Series(
        booster.feature_importance(importance_type="gain"),
        index=booster.feature_name()
    ).sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        imp.index,
        imp.values,
        color=COLOR_PRIMARY,
        edgecolor="white"
    )

    ax.set_title(
        f"Importance des variables (gain) — {model_name}",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )
    ax.set_xlabel("Gain total", fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    plt.tight_layout()
    plt.savefig(
        f"figures/importance_{model_name}{save}.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.show()


# =========================================================
# PARTIAL DEPENDENCE (calcul manuel)
# =========================================================

def _pdp_values(predict_fn, X, feature, n_grid=20, sample=3000):
    """
    PDP manuel : on fait varier une variable sur une
    grille en gardant les autres observées, et on
    moyenne les prédictions. Compatible catégorielles.
    """

    Xs = X.sample(
        n=min(sample, len(X)), random_state=42
    ).copy()

    if isinstance(Xs[feature].dtype, pd.CategoricalDtype):
        grid = list(Xs[feature].cat.categories)
    else:
        quantiles = np.linspace(0.02, 0.98, n_grid)
        grid = np.unique(
            Xs[feature].quantile(quantiles).values
        )

    means = []
    for value in grid:
        Xg = Xs.copy()
        if isinstance(Xs[feature].dtype, pd.CategoricalDtype):
            Xg[feature] = pd.Categorical(
                [value] * len(Xg),
                categories=Xs[feature].cat.categories
            )
        else:
            Xg[feature] = value
        means.append(predict_fn(Xg).mean())

    return grid, np.array(means)


def plot_pdp(
    predict_fn,
    X,
    features,
    model_name="model",
    ylabel="Prédiction moyenne",
    save=""
):
    """
    Partial Dependence Plots pour une liste de variables.

    predict_fn : fonction X -> prédictions
    (ex. le .predict du wrapper).
    """

    n = len(features)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4.5 * nrows),
        squeeze=False
    )

    axes = axes.flatten()

    for ax, feature in zip(axes, features):

        grid, means = _pdp_values(predict_fn, X, feature)

        if isinstance(X[feature].dtype, pd.CategoricalDtype):
            ax.bar(
                [str(g) for g in grid],
                means,
                color=COLOR_PRIMARY,
                edgecolor="white"
            )
            plt.setp(
                ax.get_xticklabels(),
                rotation=30,
                ha="right"
            )
        else:
            ax.plot(
                grid, means,
                linewidth=2.2,
                color=COLOR_PRIMARY
            )

        ax.set_title(feature, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)

    for ax in axes[n:]:
        fig.delaxes(ax)

    fig.suptitle(
        f"Partial Dependence — {model_name}",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(
        f"figures/pdp_{model_name}{save}.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.show()


# =========================================================
# SHAP (TreeSHAP natif LightGBM)
# =========================================================

def compute_shap_values(lgb_model, X, sample=3000):
    """
    Valeurs SHAP via LightGBM pred_contrib=True
    (TreeSHAP exact, sans dépendance au package shap).

    Retourne (X_sample, DataFrame des contributions).
    """

    Xs = X.sample(
        n=min(sample, len(X)), random_state=42
    )

    contrib = lgb_model.predict(Xs, pred_contrib=True)

    # Dernière colonne = valeur de base (espérance)
    shap_df = pd.DataFrame(
        contrib[:, :-1],
        columns=Xs.columns,
        index=Xs.index
    )

    return Xs, shap_df


def plot_shap_summary(
    lgb_model,
    X,
    model_name="model",
    top_dependence=2,
    save=""
):
    """
    Résumé SHAP :
    - barres |SHAP| moyen par variable (importance),
    - nuages SHAP vs valeur pour les variables numériques
      les plus influentes (sens et forme de l'effet).

    NB : les SHAP d'un objectif Poisson/Gamma/Tweedie de
    LightGBM sont sur l'échelle du prédicteur linéaire
    (log) : un SHAP de +0.1 multiplie la prédiction
    par exp(0.1) ≈ 1.105.
    """

    Xs, shap_df = compute_shap_values(lgb_model, X)

    mean_abs = (
        shap_df.abs().mean().sort_values()
    )

    numeric_feats = [
        f for f in mean_abs.index[::-1]
        if not isinstance(Xs[f].dtype, pd.CategoricalDtype)
    ][:top_dependence]

    ncols = 1 + len(numeric_feats)

    fig, axes = plt.subplots(
        1, ncols,
        figsize=(6 * ncols, 5.5),
        squeeze=False
    )
    axes = axes.flatten()

    # Importance SHAP
    axes[0].barh(
        mean_abs.index,
        mean_abs.values,
        color=COLOR_PRIMARY,
        edgecolor="white"
    )
    axes[0].set_title(
        "|SHAP| moyen (échelle log)",
        fontweight="bold"
    )

    # Dependence plots
    for ax, feat in zip(axes[1:], numeric_feats):
        ax.scatter(
            Xs[feat],
            shap_df[feat],
            s=6,
            alpha=0.3,
            color=COLOR_DANGER
        )
        ax.axhline(0, color=COLOR_NEUTRAL, lw=1, ls="--")
        ax.set_title(
            f"SHAP — {feat}",
            fontweight="bold"
        )
        ax.set_xlabel(feat, fontsize=LABEL_SIZE)
        ax.set_ylabel(
            "Contribution SHAP (log)",
            fontsize=10
        )

    fig.suptitle(
        f"Analyse SHAP — {model_name}",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(
        f"figures/shap_{model_name}{save}.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.show()
