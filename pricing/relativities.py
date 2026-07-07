import numpy as np
import pandas as pd

def compute_relativities(coeffs, variables):
    rows = []

    for var, info in variables.items():
        ref = info["ref"]

        for mod in info["modalites"]:
            cf = coeffs.freq(var, mod)
            cs = coeffs.sev(var, mod)

            rel_f = np.exp(cf)
            rel_s = np.exp(cs)

            rows.append({
                "Variable": var,
                "Modalité": mod,
                "Référence": mod == ref,
                "Rel_Freq": rel_f,
                "Rel_Sev": rel_s,
                "Rel_Combined": rel_f * rel_s
            })

    return pd.DataFrame(rows)