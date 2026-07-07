import lightgbm as lgb


DEFAULT_PARAMS = {
    "objective": "poisson",
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


class LGBFrequencyModel:
    """
    Modèle de fréquence LightGBM (objectif Poisson).

    Équivalent du GLM Poisson avec offset log(Exposure) :
    la cible est le TAUX de sinistres (ClaimNb / Exposure)
    et chaque observation est pondérée par son exposition.
    predict() renvoie donc une fréquence annuelle (taux par
    unité d'exposition) : multiplier par l'exposition pour
    obtenir un nombre de sinistres attendu.

    - n_estimators élevé + early stopping sur un jeu de
      validation (déviance Poisson) : le nombre d'arbres
      effectif est choisi par les données, pas fixé a priori.
    - monotone_constraints : dict {feature: +1/-1} pour
      imposer un effet monotone actuariellement justifié
      (ex. fréquence croissante avec la densité). Les
      features non listées sont libres (0).
    """

    def __init__(self, monotone_constraints=None, **params):

        self.monotone_constraints = monotone_constraints or {}
        self.params = {**DEFAULT_PARAMS, **params}
        self.model = None

    def _build_constraints(self, X):
        return [
            int(self.monotone_constraints.get(col, 0))
            for col in X.columns
        ]

    def fit(
        self,
        X_train,
        y_train,
        exposure,
        X_val=None,
        y_val=None,
        exposure_val=None,
        early_stopping_rounds=50
    ):

        params = dict(self.params)

        if self.monotone_constraints:
            params["monotone_constraints"] = (
                self._build_constraints(X_train)
            )

        self.model = lgb.LGBMRegressor(**params)

        rate = y_train / exposure

        fit_kwargs = {"sample_weight": exposure}

        if X_val is not None:
            rate_val = y_val / exposure_val
            fit_kwargs.update(
                eval_set=[(X_val, rate_val)],
                eval_sample_weight=[exposure_val],
                eval_metric="poisson",
                callbacks=[
                    lgb.early_stopping(
                        early_stopping_rounds, verbose=False
                    ),
                    lgb.log_evaluation(0),
                ],
            )

        self.model.fit(
            X_train,
            rate,
            **fit_kwargs
        )

        return self

    def predict(self, X):

        return self.model.predict(X)
