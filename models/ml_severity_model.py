import lightgbm as lgb


DEFAULT_PARAMS = {
    "objective": "gamma",
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}


class LGBSeverityModel:
    """
    Modèle de sévérité LightGBM (objectif Gamma).

    - Pondération par le nombre de sinistres (même logique
      que var_weights du GLM Gamma) : la cible est une
      sévérité moyenne par police.
    - Early stopping sur jeu de validation (déviance Gamma).
    - À entraîner de préférence sur la sévérité ÉCRÊTÉE
      (severity_capped) : un GBM ne peut pas apprendre une
      queue de distribution à partir de quelques dizaines
      de sinistres extrêmes ; l'excédent est tarifé à part
      via un chargement (cf. large_claims_model).
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
        claim_nb=None,
        X_val=None,
        y_val=None,
        claim_nb_val=None,
        early_stopping_rounds=50
    ):

        params = dict(self.params)

        if self.monotone_constraints:
            params["monotone_constraints"] = (
                self._build_constraints(X_train)
            )

        self.model = lgb.LGBMRegressor(**params)

        fit_kwargs = {"sample_weight": claim_nb}

        if X_val is not None:
            fit_kwargs.update(
                eval_set=[(X_val, y_val)],
                eval_sample_weight=(
                    [claim_nb_val]
                    if claim_nb_val is not None
                    else None
                ),
                eval_metric="gamma",
                callbacks=[
                    lgb.early_stopping(
                        early_stopping_rounds, verbose=False
                    ),
                    lgb.log_evaluation(0),
                ],
            )

        self.model.fit(
            X_train,
            y_train,
            **fit_kwargs
        )

        return self

    def predict(self, X):

        return self.model.predict(X)
