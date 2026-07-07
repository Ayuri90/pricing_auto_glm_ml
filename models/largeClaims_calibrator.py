import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class LargeClaimConfig:
    quantile_threshold: float = 0.995
    exposure_col: str = "Exposure"
    claim_col: str = "ClaimAmount"


class LargeClaimCalibrator:
    """
    Calibration actuarielle des sinistres larges :
    - séparation attritionnels / atypiques
    - chargement large claims
    - optimisation du seuil et du loading
    """

    def __init__(self, config: LargeClaimConfig = LargeClaimConfig()):
        self.config = config

        # résultats internes
        self.threshold_ = None
        self.loading_ = None
        self.p_atyp_ = None
        self.excess_mean_ = None


    # SEUIL
    def compute_threshold(self, data):
        self.threshold_ = data[self.config.claim_col].quantile(
            self.config.quantile_threshold
        )
        return self.threshold_


    # SPLIT
    def split_claims(self, data):
        if self.threshold_ is None:
            self.compute_threshold(data)

        attrit = data[data[self.config.claim_col] <= self.threshold_].copy()
        atyp   = data[data[self.config.claim_col] > self.threshold_].copy()

        return attrit, atyp


    # CHARGEMENT ATYPIQUE
    def fit_loading(self, data, claim_nb_col="ClaimNb"):
        _, atyp = self.split_claims(data)

        # Nombre de polices atypiques rapporté au nombre
        # total de sinistres : p_atyp × excess_mean
        # reconstruit exactement l'excédent total écrêté
        # (une occurrence d'excédent par police atypique,
        # le seuil s'appliquant au montant total par police).
        self.p_atyp_ = (
            len(atyp) / data[claim_nb_col].sum()
        )

        if len(atyp) > 0:
            self.excess_mean_ = (atyp[self.config.claim_col] - self.threshold_).mean()
        else:
            self.excess_mean_ = 0.0

        self.loading_ = self.p_atyp_ * self.excess_mean_

        return self.loading_


    # PRIME TWO COMPONENTS
    def compute_premium(self, freq_df, sev_model):
        """
        freq_df : dataframe fréquence (avec freq_pred)
        sev_model : modèle attritionnel (GLM Gamma)
        """

        df = freq_df.copy()

        df["sev_attrit_pred"] = sev_model.predict(df)

        df["prime_attrit"] = df["freq_pred"] * df["sev_attrit_pred"]

        df["prime_atyp"] = df["freq_pred"] * self.loading_

        df["prime_total"] = df["prime_attrit"] + df["prime_atyp"]

        return df


    # S/P RATIO
    def compute_s_p_ratio(self, df, claims_total):
        return claims_total / df["prime_total"].sum()


    # OPTIMISATION DU SEUIL
    def optimize_threshold(self, data, freq_df, sev_model, grid=None):
        """
        Recherche du meilleur quantile en minimisant |S/P - 1|
        """

        if grid is None:
            grid = np.arange(0.90, 0.999, 0.005)

        best = {"q": None, "error": np.inf}

        claims_total = data[self.config.claim_col].sum()

        for q in grid:

            self.config.quantile_threshold = q
            self.compute_threshold(data)
            self.fit_loading(data)

            df = self.compute_premium(freq_df, sev_model)

            sp = self.compute_s_p_ratio(df, claims_total)

            error = abs(1 - sp)

            if error < best["error"]:
                best = {"q": q, "error": error, "sp": sp}

        # reset best
        self.config.quantile_threshold = best["q"]
        self.compute_threshold(data)
        self.fit_loading(data)

        return best