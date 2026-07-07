import pandas as pd
import numpy as np
from itertools import product

def build_grid(engine, var_x, var_y, fixed_profile, x_values, y_values):

    grid = pd.DataFrame(index=x_values, columns=y_values)

    for x, y in product(x_values, y_values):

        profile = fixed_profile.copy()
        profile[var_x] = x
        profile[var_y] = y

        res = engine.pure_premium(profile)

        grid.loc[x, y] = res["premium"]

    return grid.astype(float).round(2)