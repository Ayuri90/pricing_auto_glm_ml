from data.data_loader import load_data
from data.preprocessing import prepare_datasets
from features.engineering import FeatureEngineer
from visualization.eda import run_eda
from models.frequency_model import train_frequency_model
from models.severity_model import train_severity_model
from models.premium_model import build_premium_predictions
from models.frequency_model import evaluate_frequency_model
from models.severity_model import evaluate_severity_model


# Load data
data, data_sev = load_data()

qualitatives_var = data.drop(
    columns=['PolicyID']
    ).select_dtypes(include=['object', 'category']).columns

quantitatives_var = data.drop(
    columns=['PolicyID']
    ).select_dtypes(include=['float64', 'int64']).columns


# Preprocessing
train_freq, test_freq, train_sev, test_sev = (
    prepare_datasets(data, data_sev)
)


# Feature engineering
fe = FeatureEngineer()

train_freq = fe.fit_transform(train_freq)

test_freq = fe.transform(test_freq)

train_sev = fe.transform(train_sev)

test_sev = fe.transform(test_sev)


# Eda
run_eda(
    train_freq,
    qualitatives_var,
    quantitatives_var
)


# Frequency model
freq_model = train_frequency_model(
    train_freq
)

freq_metrics = evaluate_frequency_model(
    freq_model,
    test_freq
)


# Severity model
sev_model = train_severity_model(
    train_sev
)

sev_metrics = evaluate_severity_model(
    sev_model,
    test_sev
)


# Pure premium model
premium_data, premium_results = (
    build_premium_predictions(
        test_freq,
        freq_model,
        sev_model
    )
)

print(freq_metrics)
print(sev_metrics)
print(premium_results)


from pricing.tarif_coefficients import GLMCoefficients
from pricing.pricing_engine import PricingEngine
from pricing.relativities import compute_relativities

coeffs = GLMCoefficients(freq_model, sev_model)
engine = PricingEngine(coeffs)

VARIABLES = {
    "Power":           {"ref": "d",
                        "modalites": ["d", "f", "h", "i_j_k",
                                      "l_o_g", "m_e", "n"]},
    "Brand":           {"ref": "Japanese (except Nissan) or Korean",
                        "modalites": ["Japanese (except Nissan) or Korean",
                                      "MCB_Fiat",
                                      "Opel, General Motors or Ford",
                                      "RNC_other",
                                      "Volkswagen, Audi, Skoda or Seat"]},
    "Region":          {"ref": "Centre_Bretagne_Normandie",
                        "modalites": ["Centre_Bretagne_Normandie",
                                      "Ile-de-France",
                                      "NPC_Limousin",
                                      "PL_PC_Aquitaine"]},
    "DriverAge_class": {"ref": "DriverAge_class_1",
                        "modalites": ["DriverAge_class_1",
                                      "DriverAge_class_2",
                                      "DriverAge_class_3",
                                      "DriverAge_class_4",
                                      "DriverAge_class_5"]},
    "CarAge_class":    {"ref": "CarAge_class_1",
                        "modalites": ["CarAge_class_1",
                                      "CarAge_class_2"]},
    "Gas":             {"ref": "Diesel",
                        "modalites": ["Diesel", "Regular"]},
    "Density_class":   {"ref": "Density_class_1",
                        "modalites": ["Density_class_1",
                                      "Density_class_2",
                                      "Density_class_3",
                                      "Density_class_4",
                                      "Density_class_5"]},
}

rel_table = compute_relativities(coeffs, VARIABLES)

prime_ref = engine.reference_premium()

profile = {
    "Power": "d",
    "Brand": "MCB_Fiat",
    "Region": "Centre_Bretagne_Normandie",
    "DriverAge_class": "DriverAge_class_3",
    "CarAge_class": "CarAge_class_1",
    "Gas": "Diesel",
    "Density_class": "Density_class_2"
}

print(f"Prime de référence : {prime_ref:.2f}")
print(f"Prime pure : {engine.pure_premium(profile)}")
print(f"Tableau de relativities : {rel_table}")

fixed_profile = {
    "Power": "d",
    "Brand": "Japanese (except Nissan) or Korean",
    "Region": "Centre_Bretagne_Normandie",
    "DriverAge_class": "DriverAge_class_3",
    "CarAge_class": "CarAge_class_1",
    "Gas": "Diesel",
    "Density_class": "Density_class_3"
}

powers = VARIABLES["Power"]["modalites"]
brands = VARIABLES["Brand"]["modalites"]

from pricing.tarif_grid import build_grid
grid = build_grid(engine, "Power", "Brand", fixed_profile, powers, brands)

print(grid)