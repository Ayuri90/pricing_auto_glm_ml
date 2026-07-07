import numpy as np
import pandas as pd

class MLFeatureEngineer:

    def fit(self, df):
        return self

    def transform(self, df):

        df = df.copy()

        df["log_Density"] = np.log1p(
            df["Density"]
        )

        return df

    def fit_transform(self, df):

        return self.fit(df).transform(df)