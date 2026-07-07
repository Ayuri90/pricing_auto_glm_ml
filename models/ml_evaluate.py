"""
=========================================================
EVALUATION ACTUARIELLE DES MODELES
---------------------------------------------------------
- Déviances Poisson / Gamma (les bonnes métriques pour
  fréquence et sévérité, contrairement au MAE/RMSE)
- Courbe de Lorenz et indice de Gini
- Table de lift / calibration par décile
- Facteur de recalage global (S/P = 1 sur le train)
=========================================================
"""

import numpy as np
import pandas as pd


# =========================================================
# DEVIANCES
# =========================================================

def poisson_deviance(y_true, y_pred, sample_weight=None):
    """
    Déviance de Poisson moyenne.

    Gère y_true = 0 (le terme y·log(y/mu) vaut 0).
    Métrique adaptée aux comptages de sinistres, là où
    le MAE/RMSE récompense un modèle qui prédit 0 partout.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(
        np.asarray(y_pred, dtype=float), 1e-12
    )

    term = np.where(
        y_true > 0,
        y_true * np.log(np.where(y_true > 0, y_true, 1) / y_pred),
        0.0
    )

    dev = 2 * (term - (y_true - y_pred))

    return float(np.average(dev, weights=sample_weight))


def gamma_deviance(y_true, y_pred, sample_weight=None):
    """
    Déviance Gamma moyenne (y_true strictement positif).
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(
        np.asarray(y_pred, dtype=float), 1e-12
    )

    dev = 2 * (
        np.log(y_pred / y_true)
        + y_true / y_pred
        - 1
    )

    return float(np.average(dev, weights=sample_weight))


# =========================================================
# LORENZ / GINI
# =========================================================

def lorenz_curve(y_true, y_pred):
    """
    Courbe de Lorenz ordonnée : part cumulée des sinistres
    en fonction de la part cumulée de POLICES, les polices
    étant triées par prime prédite croissante.

    NB : ne pas utiliser l'exposition cumulée en abscisse
    avec un tri par prime : la prime étant proportionnelle
    à l'exposition, les deux conventions se mélangent et
    la courbe peut s'inverser (d'autant que dans freMTPL
    les polices à faible exposition portent une
    sinistralité disproportionnée — l'exposition est
    partiellement endogène au sinistre).
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    order = np.argsort(y_pred)

    cum_claims = np.cumsum(y_true[order]) / y_true.sum()
    cum_policies = (
        np.arange(1, len(y_true) + 1) / len(y_true)
    )

    return cum_policies, cum_claims


def gini(y_true, y_pred):
    """
    Indice de Gini (pouvoir de segmentation du tarif).
    Positif dès que le tarif ordonne mieux qu'au hasard.
    """

    x, y = lorenz_curve(y_true, y_pred)

    # np.trapz retiré de NumPy 2.x (renommé trapezoid)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    area = trapezoid(y, x)

    return float(1 - 2 * area)


# =========================================================
# LIFT / CALIBRATION PAR DECILE
# =========================================================

def lift_table(
    y_true,
    y_pred,
    n_bins=10
):
    """
    Table de lift : les polices sont triées par prime
    prédite et découpées en n_bins groupes d'effectifs
    égaux ; on compare la prime prédite moyenne et le
    coût observé moyen par groupe.

    Un modèle bien calibré a pred ≈ obs dans chaque
    groupe ; un modèle segmentant a un fort écart entre
    le premier et le dernier groupe.
    """

    df = pd.DataFrame({
        "true": np.asarray(y_true, dtype=float),
        "pred": np.asarray(y_pred, dtype=float)
    })

    df["bin"] = pd.qcut(
        df["pred"].rank(method="first"),
        q=n_bins,
        labels=range(1, n_bins + 1)
    )

    table = (
        df.groupby("bin", observed=True)
        .agg(
            n=("true", "size"),
            pred_mean=("pred", "mean"),
            obs_mean=("true", "mean")
        )
        .reset_index()
    )

    table["ratio_obs_pred"] = (
        table["obs_mean"] / table["pred_mean"]
    )

    return table


# =========================================================
# RECALAGE GLOBAL
# =========================================================

def compute_rebalancing_factor(
    observed_total,
    predicted_total
):
    """
    Facteur multiplicatif tel que la prime totale
    recalée égale la charge observée (S/P = 1 sur le
    périmètre de calcul, typiquement le train).

    Contrairement au GLM Poisson (qui garantit l'équilibre
    total sur le train par construction), un GBM n'est pas
    calibré au niveau agrégé : ce recalage est
    indispensable avant toute comparaison de S/P.
    """

    return observed_total / predicted_total


# =========================================================
# RAPPORT COMPARATIF
# =========================================================

def model_report(
    name,
    claim_nb,
    freq_pred,
    premium_pred,
    claim_amount
):
    """
    Ligne de rapport standardisée pour un modèle :
    déviance Poisson (fréquence), Gini (segmentation),
    S/P (calibration globale).

    freq_pred peut être None (modèle Tweedie direct,
    sans composante fréquence) : la déviance Poisson
    vaut alors NaN.
    """

    return {
        "model": name,
        "poisson_deviance": (
            round(poisson_deviance(claim_nb, freq_pred), 5)
            if freq_pred is not None
            else np.nan
        ),
        "gini": round(
            gini(claim_amount, premium_pred), 4
        ),
        "sp_ratio": round(
            float(
                np.asarray(claim_amount).sum()
                / np.asarray(premium_pred).sum()
            ),
            4
        )
    }
