from sklearn.model_selection import train_test_split
import pandas as pd

from config import TEST_SIZE, RANDOM_STATE


def split_data(data, data_sev):
    """
    Split train/test au niveau police.

    Le split sévérité est dérivé du split fréquence via PolicyID :
    une même police ne peut pas être dans le train fréquence
    et le test sévérité (ou inversement), ce qui évite toute
    fuite d'information entre les deux modèles.
    """

    train_freq, test_freq = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_ids = set(train_freq["PolicyID"])

    train_sev = data_sev[data_sev["PolicyID"].isin(train_ids)].copy()
    test_sev = data_sev[~data_sev["PolicyID"].isin(train_ids)].copy()

    return train_freq, test_freq, train_sev, test_sev

def regroupement(df):
    mapping_power = {
        "f": "f", "g": "l_o_g", "e": "m_e", "d": "d", "h": "h",
        "i": "i_j_k", "j": "i_j_k",
        "k": "i_j_k", "l": "l_o_g", "m": "m_e",
        "n": "n", "o": "l_o_g",
    }
    df["Power"] = df["Power"].map(mapping_power)
    df["Power"] = pd.Categorical(df["Power"],
        categories=["d", "f", "h", "i_j_k", "m_e", "l_o_g", "n"],
        ordered=False)

    ## Brand
    ### Fiat_(Mercedes, Chrysler or BMW), other_(Renault, Nissan or Citroen)
    fiat_MCB = ["Fiat", "Mercedes, Chrysler or BMW"]
    df["Brand"] = df["Brand"].apply(lambda x: "MCB_Fiat" if x in fiat_MCB else x)
    other_brands = ["other", "Renault, Nissan or Citroen"]
    df["Brand"] = df["Brand"].apply(lambda x: "RNC_other" if x in other_brands else x)

    ## Region
    centre_bretagne_N = ["Centre", "Bretagne", "Haute-Normandie", "Basse-Normandie"]
    df["Region"] = df["Region"].apply(lambda x: "Centre_Bretagne_Normandie" if x in centre_bretagne_N else x)
    PL_PC_Aquitaine = ["Poitou-Charentes", "Aquitaine", "Pays-de-la-Loire"]
    df["Region"] = df["Region"].apply(lambda x: "PL_PC_Aquitaine" if x in PL_PC_Aquitaine else x)
    NPC_Limousin = ["Nord-Pas-de-Calais", "Limousin"]
    df["Region"] = df["Region"].apply(lambda x: "NPC_Limousin" if x in NPC_Limousin else x)

    # Catégories explicites : la première modalité est la référence
    # utilisée par patsy dans les GLM (ne dépend plus de l'ordre
    # alphabétique implicite).
    df["Brand"] = pd.Categorical(
        df["Brand"],
        categories=[
            "Japanese (except Nissan) or Korean",
            "MCB_Fiat",
            "Opel, General Motors or Ford",
            "RNC_other",
            "Volkswagen, Audi, Skoda or Seat",
        ],
        ordered=False
    )

    df["Region"] = pd.Categorical(
        df["Region"],
        categories=[
            "Centre_Bretagne_Normandie",
            "Ile-de-France",
            "NPC_Limousin",
            "PL_PC_Aquitaine",
        ],
        ordered=False
    )

    df["Gas"] = pd.Categorical(
        df["Gas"],
        categories=["Diesel", "Regular"],
        ordered=False
    )

    return df

def prepare_datasets(data, data_sev):

    train_freq, test_freq, train_sev, test_sev = split_data(
        data,
        data_sev
    )

    train_freq = regroupement(train_freq)
    test_freq = regroupement(test_freq)

    train_sev = regroupement(train_sev)
    test_sev = regroupement(test_sev)

    return train_freq, test_freq, train_sev, test_sev
