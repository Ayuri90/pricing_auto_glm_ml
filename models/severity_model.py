import pandas as pd
import statsmodels.api as sm
from itertools import combinations
from statsmodels.formula.api import glm
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error
)

# DEFAULT VARIABLES
DEFAULT_SEVERITY_VARIABLES = [
    "C(Power)",
    "C(Brand)",
    "C(Region)",
    "C(DriverAge_class)",
    "C(CarAge_class)",
    "C(Gas)",
    "C(Density_class)"
]


def build_severity_formula(variables):
    """
    Construit la formule statsmodels.
    """
    return (
        "severity_mean ~ "
        + " + ".join(variables)
    )


# TRAIN GAMMA MODEL
def train_gamma_model(
    train_data,
    formula,
    weight_col="ClaimNb"
):
    """
    Entraîne un modèle Gamma GLM.

    La cible étant une sévérité moyenne par police,
    chaque observation est pondérée par son nombre de
    sinistres (var_weights) : une moyenne calculée sur
    4 sinistres est plus fiable que sur 1 seul.
    """

    family = sm.families.Gamma(
        link=sm.genmod.families.links.Log()
    )

    model = glm(
        formula=formula,
        data=train_data,
        family=family,
        var_weights=train_data[weight_col]
    ).fit()

    return model


# FULL SEVERITY MODEL
def train_severity_model(
    train_data,
    variables=None
):
    """
    Entraîne le modèle de sévérité final.
    """

    if variables is None:
        variables = DEFAULT_SEVERITY_VARIABLES

    formula = build_severity_formula(
        variables
    )

    model = train_gamma_model(
        train_data,
        formula
    )

    return model


# MODEL SELECTION
def search_best_severity_model(
    train_data,
    candidate_variables=None,
    min_variables=2
):
    """
    Recherche les meilleurs modèles
    selon l'AIC.
    """

    if candidate_variables is None:
        candidate_variables = (
            DEFAULT_SEVERITY_VARIABLES
        )

    results = []

    # Test all combinations
    for n_vars in range(
        min_variables,
        len(candidate_variables) + 1
    ):

        for variables in combinations(
            candidate_variables,
            n_vars
        ):

            formula = build_severity_formula(
                list(variables)
            )

            model = train_gamma_model(
                train_data,
                formula
            )

            results.append({
                "variables": variables,
                "formula": formula,
                "AIC": round(model.aic, 2),
                "BIC": round(model.bic, 2),
                "LogLikelihood": round(model.llf, 2)
            })

    results_df = pd.DataFrame(results)

    results_df = (
        results_df
        .sort_values("AIC")
        .reset_index(drop=True)
    )

    return results_df


# GET BEST MODEL
def train_best_severity_model(
    train_data,
    candidate_variables=None
):
    """
    Recherche puis entraîne
    le meilleur modèle.
    """

    results_df = search_best_severity_model(
        train_data,
        candidate_variables
    )

    best_formula = results_df.iloc[0]["formula"]

    best_model = train_gamma_model(
        train_data,
        best_formula
    )

    return best_model, results_df


# PREDICTIONS
def predict_severity(
    model,
    data
):
    """
    Génère les prédictions de sévérité.
    """

    predictions = model.predict(
        exog=data
    )

    return predictions


# EVALUATION
def evaluate_severity_model(
    model,
    test_data,
    target_col="severity_mean"
):
    """
    Évalue le modèle de sévérité.
    """

    predictions = predict_severity(
        model,
        test_data
    )

    mae = mean_absolute_error(
        test_data[target_col],
        predictions
    )

    rmse = root_mean_squared_error(
        test_data[target_col],
        predictions
    )

    metrics = {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4)
    }

    return metrics