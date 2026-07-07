import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import combinations
from statsmodels.formula.api import glm
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error
)

# DEFAULT VARIABLES
DEFAULT_FREQUENCY_VARIABLES = [
    "C(Power)",
    "C(Brand)",
    "C(Region)",
    "C(DriverAge_class)",
    "C(CarAge_class)",
    "C(Gas)",
    "C(Density_class)"
]

# BUILD FORMULA
def build_frequency_formula(variables):
    """
    Construit la formule statsmodels
    pour le modèle de fréquence.
    """

    return (
        "ClaimNb ~ "
        + " + ".join(variables)
    )


# TRAIN POISSON MODEL
def train_poisson_model(
    train_data,
    formula,
    exposure_col="Exposure"
):
    """
    Entraîne un modèle GLM Poisson.
    """

    family = sm.families.Poisson()

    model = glm(
        formula=formula,
        data=train_data,
        family=family,
        offset=np.log(train_data[exposure_col])
    ).fit()

    return model


# FULL FREQUENCY MODEL
def train_frequency_model(
    train_data,
    variables=None
):
    """
    Entraîne le modèle de fréquence final.
    """

    if variables is None:
        variables = DEFAULT_FREQUENCY_VARIABLES

    formula = build_frequency_formula(
        variables
    )

    model = train_poisson_model(
        train_data,
        formula
    )
    return model


# MODEL SELECTION
def search_best_frequency_model(
    train_data,
    candidate_variables=None,
    min_variables=2
):
    """
    Recherche les meilleurs modèles
    de fréquence selon l'AIC.
    """

    if candidate_variables is None:

        candidate_variables = (
            DEFAULT_FREQUENCY_VARIABLES
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

            formula = build_frequency_formula(
                list(variables)
            )

            model = train_poisson_model(
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

# BEST MODEL
def train_best_frequency_model(
    train_data,
    candidate_variables=None
):
    """
    Recherche puis entraîne
    le meilleur modèle de fréquence.
    """

    results_df = search_best_frequency_model(
        train_data,
        candidate_variables
    )

    best_formula = results_df.iloc[0]["formula"]

    best_model = train_poisson_model(
        train_data,
        best_formula
    )

    return best_model, results_df

# PREDICTIONS
def predict_frequency(
    model,
    data,
    exposure_col="Exposure"
):
    """
    Génère les prédictions de fréquence.
    """

    predictions = model.predict(
        exog=data,
        offset=np.log(data[exposure_col])
    )

    return predictions

# EVALUATION
def evaluate_frequency_model(
    model,
    test_data,
    target_col="ClaimNb"
):
    """
    Évalue le modèle de fréquence.
    """

    predictions = predict_frequency(
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

# PEARSON RESIDUALS
def compute_pearson_residuals(
    data,
    observed_col="ClaimNb",
    predicted_col="freq_pred"
):
    """
    Calcule les résidus de Pearson.
    """

    residuals = (
        data[observed_col]
        - data[predicted_col]
    ) / np.sqrt(data[predicted_col])

    return residuals

# RESIDUAL ANALYSIS
def residuals_by_group(
    data,
    group_var,
    residual_col="pearson_residuals"
):
    """
    Moyenne des résidus par groupe.
    """

    summary = (
        data.groupby(group_var)[residual_col]
        .mean()
        .round(4)
        .sort_values()
    )

    return summary