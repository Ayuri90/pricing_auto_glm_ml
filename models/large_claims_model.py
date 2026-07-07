"""
=========================================================
LARGE CLAIMS MODEL
---------------------------------------------------------
Gestion actuarielle des sinistres atypiques :
- Détection du seuil M1
- Split attritionnels / atypiques
- Écrêtement
- GLM Gamma attritionnel
- Chargement atypique
- Prime pure en deux composantes
=========================================================
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import glm


def compute_mean_excess(
    values: pd.Series,
    thresholds: np.ndarray,
    min_count: int = 20
) -> pd.DataFrame:
    """
    Compute empirical Mean Excess Function (MEF).

    E[X - u | X > u]

    Parameters
    ----------
    values : pd.Series
        Claim amounts.
    thresholds : np.ndarray
        Threshold grid.
    min_count : int
        Minimum number of observations above threshold.

    Returns
    -------
    pd.DataFrame
    """

    rows = []

    for u in thresholds:

        above = values[values > u]

        if len(above) >= min_count:

            mean_excess = (above - u).mean()

        else:

            mean_excess = np.nan

        rows.append({
            "threshold": u,
            "mean_excess": mean_excess,
            "n_above": len(above)
        })

    return pd.DataFrame(rows)


# THRESHOLD DETECTION
def detect_large_claims_threshold(
    data: pd.DataFrame,
    claim_amount_col: str = "ClaimAmount",
    method: str = "quantile",
    quantile: float = 0.95,
    fixed_threshold: Optional[float] = None
) -> Dict:
    """
    Detect large claim threshold M1.

    Parameters
    ----------
    method :
        "quantile"
        "fixed"

    Returns
    -------
    dict
    """

    claim_amounts = data[claim_amount_col].dropna()

    if method == "quantile":

        threshold = claim_amounts.quantile(quantile)

    elif method == "fixed":

        if fixed_threshold is None:
            raise ValueError(
                "fixed_threshold must be provided "
                "when method='fixed'"
            )

        threshold = fixed_threshold

    else:

        raise ValueError(
            "method must be 'quantile' or 'fixed'"
        )

    thresholds_grid = np.percentile(
        claim_amounts,
        np.arange(50, 99.5, 0.5)
    )

    mef_df = compute_mean_excess(
        claim_amounts,
        thresholds_grid
    )

    stats_df = (
        claim_amounts
        .describe(
            percentiles=[
                .75,
                .90,
                .95,
                .97,
                .99,
                .995
            ]
        )
        .to_frame(name="value")
    )

    n_large = (claim_amounts > threshold).sum()

    return {
        "threshold": threshold,
        "n_large_claims": n_large,
        "pct_large_claims":
            n_large / len(claim_amounts),
        "stats": stats_df,
        "mef": mef_df
    }


# SPLIT ATTRITIONAL / LARGE
def split_attritional_large_claims(
    data: pd.DataFrame,
    threshold: float,
    claim_amount_col: str = "ClaimAmount"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split claims into:
    - attritional
    - large claims
    """

    attritional = data[
        data[claim_amount_col] <= threshold
    ].copy()

    large = data[
        data[claim_amount_col] > threshold
    ].copy()

    return attritional, large


# CAPPING
def cap_claim_amounts(
    data: pd.DataFrame,
    threshold: float,
    claim_amount_col: str = "ClaimAmount",
    claim_nb_col: str = "ClaimNb"
) -> pd.DataFrame:
    """
    Cap claim amounts at threshold M1.

    Adds:
    - ClaimAmount_capped
    - severity_capped
    """

    df = data.copy()

    df["ClaimAmount_capped"] = (
        df[claim_amount_col]
        .clip(upper=threshold)
    )

    df["severity_capped"] = (
        df["ClaimAmount_capped"]
        / df[claim_nb_col]
    )

    return df


# CAPPING SUMMARY (réutilisable GLM et ML)
def prepare_severity_capping(
    train_sev: pd.DataFrame,
    quantile: float = 0.995,
    claim_amount_col: str = "ClaimAmount_total",
    claim_nb_col: str = "ClaimNb"
) -> Dict:
    """
    Prépare l'écrêtement de la sévérité pour n'importe
    quel modèle (GLM ou ML) :

    - seuil M1 au quantile donné du montant total par police,
    - dataset écrêté (colonnes ClaimAmount_capped et
      severity_capped),
    - chargement atypique unitaire
      p_large × mean_excess = excédent total / nb sinistres,
      qui reconstruit exactement l'excédent écrêté sur le train.
    """

    threshold = train_sev[claim_amount_col].quantile(quantile)

    large = train_sev[
        train_sev[claim_amount_col] > threshold
    ]

    train_capped = cap_claim_amounts(
        train_sev,
        threshold=threshold,
        claim_amount_col=claim_amount_col,
        claim_nb_col=claim_nb_col
    )

    if len(large) > 0:
        mean_excess = (
            large[claim_amount_col] - threshold
        ).mean()
    else:
        mean_excess = 0.0

    p_large = len(large) / train_sev[claim_nb_col].sum()

    return {
        "train_capped": train_capped,
        "threshold": threshold,
        "n_large": len(large),
        "p_large": p_large,
        "mean_excess": mean_excess,
        "loading_per_claim": p_large * mean_excess
    }


# ATTRITIONAL SEVERITY MODEL
@dataclass
class AttritionalSeverityModel:
    """
    GLM Gamma severity model on capped claims.
    """

    formula: str
    weight_col: str = "ClaimNb"

    def __post_init__(self):

        self.model_ = None
        self.results_ = None



    def fit(
        self,
        data: pd.DataFrame
    ):
        """
        Fit Gamma GLM.

        Pondération par le nombre de sinistres :
        la cible est une sévérité moyenne par police.
        """

        self.model_ = glm(
            formula=self.formula,
            data=data,
            family=sm.families.Gamma(
                link=sm.families.links.Log()
            ),
            var_weights=data[self.weight_col]
        )

        self.results_ = self.model_.fit()

        return self

    

    def predict(
        self,
        data: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict capped severity.
        """

        if self.results_ is None:

            raise ValueError(
                "Model must be fitted first."
            )

        return self.results_.predict(data)


    def summary(self):

        if self.results_ is None:

            raise ValueError(
                "Model must be fitted first."
            )

        return self.results_.summary()

    

    @property
    def params_(self):

        if self.results_ is None:

            raise ValueError(
                "Model must be fitted first."
            )

        return self.results_.params


# LARGE CLAIM LOADING
def compute_large_claim_loading(
    large_claims: pd.DataFrame,
    threshold: float,
    claim_amount_col: str = "ClaimAmount"
) -> Dict:
    """
    Compute actuarial loading for large claims.

    Returns
    -------
    dict
        Clés identiques quelle que soit la branche :
        mean_large_claim, mean_excess, n_large.
    """

    if len(large_claims) == 0:

        return {
            "mean_large_claim": 0.0,
            "mean_excess": 0.0,
            "n_large": 0
        }

    mean_large_claim = (
        large_claims[claim_amount_col]
        .mean()
    )

    mean_excess = (
        large_claims[claim_amount_col]
        - threshold
    ).mean()

    return {
        "mean_large_claim": mean_large_claim,
        "mean_excess": mean_excess,
        "n_large": len(large_claims)
    }


# TWO-COMPONENT PREMIUM
def compute_two_component_premium(
    data: pd.DataFrame,
    freq_pred_col: str,
    sev_pred_col: str,
    p_large: float,
    loading_per_claim: float
) -> pd.DataFrame:
    """
    Compute:
    - attritional premium
    - large claim loading
    - total pure premium
    """

    df = data.copy()

    # Attritional premium
    df["prime_attritional"] = (
        df[freq_pred_col]
        * df[sev_pred_col]
    )

    # Large claim premium
    df["prime_large"] = (
        df[freq_pred_col]
        * p_large
        * loading_per_claim
    )

    # Total premium
    df["pure_premium_2comp"] = (
        df["prime_attritional"]
        + df["prime_large"]
    )

    return df


# FULL PIPELINE
def run_large_claims_pipeline(
    train_sev: pd.DataFrame,
    test_freq: pd.DataFrame,
    formula: str,
    threshold_method: str = "quantile",
    quantile: float = 0.95,
    claim_amount_col: str = "ClaimAmount",
    claim_nb_col: str = "ClaimNb",
    freq_pred_col: str = "freq_pred"
):
    """
    Complete actuarial large claims pipeline.

    Le GLM attritionnel est TOUJOURS ajusté sur la
    sévérité écrêtée (severity_capped) : la partie
    gauche de la formule fournie est remplacée, seule
    la partie droite (les variables) est conservée.
    Sans cela, le modèle "attritionnel" serait ajusté
    sur la sévérité brute et le chargement atypique
    double-compterait les gros sinistres.
    """

    # Threshold
    threshold_info = detect_large_claims_threshold(
        train_sev,
        claim_amount_col=claim_amount_col,
        method=threshold_method,
        quantile=quantile
    )

    M1 = threshold_info["threshold"]

    # Split
    attritional, large = (
        split_attritional_large_claims(
            train_sev,
            threshold=M1,
            claim_amount_col=claim_amount_col
        )
    )

    # Capping
    train_capped = cap_claim_amounts(
        train_sev,
        threshold=M1,
        claim_amount_col=claim_amount_col,
        claim_nb_col=claim_nb_col
    )

    # Fit attritional model sur la sévérité écrêtée :
    # la cible de la formule est forcée à severity_capped
    rhs = formula.split("~", 1)[1]
    formula_capped = "severity_capped ~" + rhs

    model = AttritionalSeverityModel(
        formula=formula_capped,
        weight_col=claim_nb_col
    )

    model.fit(train_capped)

    # Predict capped severity
    test_freq = test_freq.copy()

    test_freq["sev_attritional_pred"] = (
        model.predict(test_freq)
    )

    # Large claim loading
    loading_info = compute_large_claim_loading(
        large,
        threshold=M1,
        claim_amount_col=claim_amount_col
    )

    # Probabilité qu'un sinistre déclenche le dépassement
    # de seuil de sa police : nombre de polices atypiques
    # rapporté au nombre total de sinistres. Ainsi
    # p_large × mean_excess = excédent total / nb sinistres,
    # et le chargement reconstruit exactement l'excédent
    # écrêté sur le train (le seuil étant appliqué au
    # montant total par police, une seule "occurrence
    # d'excédent" par police atypique).
    p_large = (
        len(large)
        / train_sev[claim_nb_col].sum()
    )

    loading_info["p_large"] = p_large

    loading_per_claim = (
        loading_info["mean_excess"]
    )

    # Premium
    test_freq = compute_two_component_premium(
        data=test_freq,
        freq_pred_col=freq_pred_col,
        sev_pred_col="sev_attritional_pred",
        p_large=p_large,
        loading_per_claim=loading_per_claim
    )

    return {
        "threshold_info": threshold_info,
        "attritional_claims": attritional,
        "large_claims": large,
        "train_capped": train_capped,
        "severity_model": model,
        "loading_info": loading_info,
        "test_predictions": test_freq
    }

def compute_tariff_with_large_claim_loading(
    data,
    base_premium_col="pure_premium",
    frequency_col="freq_pred",
    loading_per_claim=0.0,
    output_col="pure_premium_loaded"
):
    """
    Ajoute un chargement gros sinistres
    à une prime pure existante.

    Formule
    --------
    Prime finale =
        Prime pure standard
        +
        fréquence x chargement atypique

    La fréquence prédite par le GLM Poisson intègre
    déjà l'exposition (offset log(Exposure)) : pas de
    multiplication supplémentaire par l'exposition.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset contenant les prédictions.

    base_premium_col : str
        Colonne de prime pure standard.

    frequency_col : str
        Colonne fréquence prédite
        (nombre de sinistres attendu par police).

    loading_per_claim : float
        Chargement atypique unitaire :
        P(atypique) × E[excédent | atypique]

    output_col : str
        Colonne de sortie.

    Returns
    -------
    pd.DataFrame
        Dataset enrichi :
            - large_claim_loading
            - pure_premium_loaded
            - loading_ratio
    """

    df = data.copy()

    # =====================================================
    # Chargement gros sinistres
    # =====================================================

    df["large_claim_loading"] = (
        df[frequency_col]
        * loading_per_claim
    )

    # =====================================================
    # Prime finale chargée
    # =====================================================

    df[output_col] = (
        df[base_premium_col]
        + df["large_claim_loading"]
    )

    # =====================================================
    # Part du chargement
    # =====================================================

    df["loading_ratio"] = (
        df["large_claim_loading"]
        / df[output_col]
    )

    return df