import lightgbm as lgb


DEFAULT_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}


class LGBTweedieModel:
    """
    Modèle Tweedie direct de prime pure.

    Alternative à la décomposition fréquence × sévérité :
    un seul modèle sur la charge annuelle par police
    (ClaimAmount_total / Exposure), pondéré par
    l'exposition. La loi Tweedie avec variance power
    dans (1, 2) est un Poisson composé Gamma : masse en
    zéro (pas de sinistre) + montants continus positifs,
    exactement la structure d'une charge de sinistres.

    Avantages :
    - apprend sur tout le portefeuille (y compris les
      polices sans sinistre) et non sur les seules
      polices sinistrées comme le modèle de sévérité ;
    - pas de risque d'incohérence à l'assemblage de deux
      modèles séparés.

    predict() renvoie une prime pure ANNUELLE :
    multiplier par l'exposition de la police.
    """

    def __init__(
        self,
        tweedie_variance_power=1.5,
        monotone_constraints=None,
        **params
    ):

        self.monotone_constraints = monotone_constraints or {}
        self.params = {
            **DEFAULT_PARAMS,
            "tweedie_variance_power": tweedie_variance_power,
            **params,
        }
        self.model = None

    def _build_constraints(self, X):
        return [
            int(self.monotone_constraints.get(col, 0))
            for col in X.columns
        ]

    def fit(
        self,
        X_train,
        claim_amount,
        exposure,
        X_val=None,
        claim_amount_val=None,
        exposure_val=None,
        early_stopping_rounds=50
    ):

        params = dict(self.params)

        if self.monotone_constraints:
            params["monotone_constraints"] = (
                self._build_constraints(X_train)
            )

        self.model = lgb.LGBMRegressor(**params)

        # Charge annuelle observée par police
        y = claim_amount / exposure

        fit_kwargs = {"sample_weight": exposure}

        if X_val is not None:
            y_val = claim_amount_val / exposure_val
            fit_kwargs.update(
                eval_set=[(X_val, y_val)],
                eval_sample_weight=[exposure_val],
                eval_metric="tweedie",
                callbacks=[
                    lgb.early_stopping(
                        early_stopping_rounds, verbose=False
                    ),
                    lgb.log_evaluation(0),
                ],
            )

        self.model.fit(
            X_train,
            y,
            **fit_kwargs
        )

        return self

    def predict(self, X):

        return self.model.predict(X)
