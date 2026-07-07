import pandas as pd

from config import MAX_EXPOSURE


def load_data():

    freq = pd.read_csv("data/raw/mydata.csv")
    sev = pd.read_csv("data/raw/freMTPLsev.csv")

    # Écrêtement des expositions anormales (> 1 an)
    freq["Exposure"] = freq["Exposure"].clip(upper=MAX_EXPOSURE)

    agg_sev = (
        sev.groupby("PolicyID")["ClaimAmount"]
        .sum()
        .reset_index()
        .rename(columns={"ClaimAmount": "ClaimAmount_total"})
    )

    data = freq.merge(agg_sev, on="PolicyID", how="left")

    data["ClaimAmount_total"] = (
        data["ClaimAmount_total"].fillna(0)
    )

    data_sev = data[data["ClaimAmount_total"] > 0].copy()

    data_sev["severity_mean"] = (
        data_sev["ClaimAmount_total"] /
        data_sev["ClaimNb"]
    )

    return data, data_sev
