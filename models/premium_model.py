import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error
)
from models.frequency_model import predict_frequency
from models.severity_model import predict_severity


# =========================================================
# PURE PREMIUM
# =========================================================

def compute_pure_premium(
    data,
    frequency_model,
    severity_model,
    exposure_col="Exposure"
):
    """
    Calcule la prime pure :
    
    Prime pure = fréquence × sévérité
    """

    data = data.copy()

    # =====================================
    # Frequency predictions
    # =====================================

    data["freq_pred"] = predict_frequency(
        frequency_model,
        data,
        exposure_col=exposure_col
    )

    # =====================================
    # Severity predictions
    # =====================================

    data["sev_pred"] = predict_severity(
        severity_model,
        data
    )

    # =====================================
    # Pure premium
    # =====================================

    data["pure_premium"] = (
        data["freq_pred"]
        * data["sev_pred"]
    )

    return data


# =========================================================
# PURE PREMIUM SUMMARY
# =========================================================

def summarize_pure_premium(
    data,
    premium_col="pure_premium"
):
    """
    Résumé statistique de la prime pure.
    """

    summary = (
        data[premium_col]
        .describe()
        .round(2)
    )

    return summary


# =========================================================
# PURE PREMIUM BY GROUP
# =========================================================

def premium_by_group(
    data,
    group_var,
    premium_col="pure_premium"
):
    """
    Prime pure moyenne par groupe.
    """

    summary = (
        data.groupby(group_var)[premium_col]
        .mean()
        .round(2)
        .sort_values()
    )

    return summary


# =========================================================
# OBSERVED COST
# =========================================================

def compute_observed_cost(
    data,
    claim_amount_col="ClaimAmount_total"
):
    """
    Calcule le coût observé.

    ClaimAmount_total est déjà le coût total de la
    police (somme de ses sinistres) : on le somme
    directement, sans le multiplier par ClaimNb.
    """

    observed_cost = data[claim_amount_col].sum()

    return observed_cost


# =========================================================
# PREDICTED PREMIUM
# =========================================================

def compute_predicted_premium(
    data,
    premium_col="pure_premium"
):
    """
    Somme des primes prédites.
    """

    predicted_premium = (
        data[premium_col]
        .sum()
    )

    return predicted_premium


# =========================================================
# S/P RATIO
# =========================================================

def compute_sp_ratio(
    observed_cost,
    predicted_premium
):
    """
    Calcule le ratio Sinistres / Primes.
    """

    return observed_cost / predicted_premium


# =========================================================
# GLOBAL PREMIUM EVALUATION
# =========================================================

def evaluate_premium_model(
    data,
    premium_col="pure_premium",
    observed_col="ClaimAmount_total"
):
    """
    Évalue les performances
    du modèle de prime pure.
    """

    mae = mean_absolute_error(
        data[observed_col],
        data[premium_col]
    )

    rmse = root_mean_squared_error(
        data[observed_col],
        data[premium_col]
    )

    metrics = {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2)
    }

    return metrics


# =========================================================
# COMPLETE PIPELINE
# =========================================================

def build_premium_predictions(
    data,
    frequency_model,
    severity_model,
    exposure_col="Exposure"
):
    """
    Pipeline complet :

    - prédiction fréquence,
    - prédiction sévérité,
    - calcul prime pure,
    - calcul S/P.
    """

    # =====================================
    # Pure premium
    # =====================================

    data = compute_pure_premium(
        data,
        frequency_model,
        severity_model,
        exposure_col
    )

    # =====================================
    # Metrics
    # =====================================

    observed_cost = compute_observed_cost(
        data
    )

    predicted_premium = (
        compute_predicted_premium(data)
    )

    sp_ratio = compute_sp_ratio(
        observed_cost,
        predicted_premium
    )

    premium_metrics = (
        evaluate_premium_model(data)
    )

    results = {
        "observed_cost": round(observed_cost, 2),
        "predicted_premium": round(predicted_premium, 2),
        "sp_ratio": round(sp_ratio, 4),
        "premium_metrics": premium_metrics
    }

    return data, results


def compute_two_component_premium(
    data,
    frequency_predictions,
    severity_attritional_predictions,
    atypical_loading,
    output_col="pure_premium_2comp"
):
    """
    Calcule la prime pure à deux composantes.

    Prime totale =
        fréquence × sévérité attritionnelle
        +
        fréquence × chargement atypique

    Les prédictions de fréquence issues du GLM Poisson
    intègrent déjà l'exposition via l'offset
    log(Exposure) : on ne multiplie donc PAS une
    seconde fois par l'exposition.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset contenant les observations.

    frequency_predictions : array-like
        Nombre de sinistres attendu par police
        (exposition déjà incluse via l'offset).

    severity_attritional_predictions : array-like
        Sévérité attritionnelle prédite
        par le modèle Gamma écrêté.

    atypical_loading : float
        Chargement atypique unitaire :
        P(atypique) × E[excédent | atypique]

    output_col : str
        Nom de la colonne finale.

    Returns
    -------
    pd.DataFrame
        Dataset enrichi :
            - freq_pred
            - sev_attrit_pred
            - prime_attrit
            - prime_atyp
            - prime pure totale
    """

    df = data.copy()

    # =====================================================
    # Fréquence prédite
    # =====================================================

    df["freq_pred"] = np.asarray(frequency_predictions)

    # =====================================================
    # Sévérité attritionnelle prédite
    # =====================================================

    df["sev_attrit_pred"] = np.asarray(
        severity_attritional_predictions
    )

    # =====================================================
    # Prime attritionnelle
    # =====================================================

    df["prime_attrit"] = (
        df["freq_pred"]
        * df["sev_attrit_pred"]
    )

    # =====================================================
    # Prime atypique
    # =====================================================

    df["prime_atyp"] = (
        df["freq_pred"]
        * atypical_loading
    )

    # =====================================================
    # Prime pure totale
    # =====================================================

    df[output_col] = (
        df["prime_attrit"]
        + df["prime_atyp"]
    )

    # =====================================================
    # Relativité atypique
    # =====================================================

    df["atypical_share"] = (
        df["prime_atyp"]
        / df[output_col]
    )

    return df
