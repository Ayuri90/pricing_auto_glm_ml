import numpy as np
import pandas as pd

from config import N_BINS


class FeatureEngineer:

    def __init__(self):

        self.feature_params = {
            "CarAge": {
                "n_bins": N_BINS["CarAge"],
                "transform": lambda x: x
            },

            "DriverAge": {
                "n_bins": N_BINS["DriverAge"],
                "transform": lambda x: x
            },

            "Density": {
                "n_bins": N_BINS["Density"],
                "transform": np.log
            }
        }

        self.bins_dict = {}
        self.labels_dict = {}

    def _discretize_train(
        self,
        series,
        n_bins,
        prefix
    ):

        _, bins = pd.qcut(
            series,
            q=n_bins,
            retbins=True,
            duplicates="drop"
        )

        labels = [
            f"{prefix}_{i+1}"
            for i in range(len(bins) - 1)
        ]

        discretized = pd.cut(
            series,
            bins=bins,
            labels=labels,
            include_lowest=True
        )

        return discretized, bins, labels

    def fit(self, df):

        for variable, params in self.feature_params.items():

            transformed = params["transform"](
                df[variable]
            )

            class_name = f"{variable}_class"

            (
                _,
                bins,
                labels
            ) = self._discretize_train(
                transformed,
                params["n_bins"],
                class_name
            )

            self.bins_dict[variable] = bins
            self.labels_dict[variable] = labels

        return self

    def transform(self, df):

        df = df.copy()

        for variable, params in self.feature_params.items():

            transformed = params["transform"](
                df[variable]
            )

            class_name = f"{variable}_class"

            # Bornes extrêmes ouvertes : une valeur de test hors de
            # l'intervalle observé sur le train est affectée à la
            # classe extrême au lieu de devenir NaN silencieusement.
            bins = self.bins_dict[variable].copy()
            bins[0] = -np.inf
            bins[-1] = np.inf

            df[class_name] = pd.cut(
                transformed,
                bins=bins,
                labels=self.labels_dict[variable],
                include_lowest=True
            )

        return df

    def fit_transform(self, df):

        self.fit(df)

        return self.transform(df)
    
    def get_discretization_summary(self):

        rows = []

        for variable in self.bins_dict:

            bins = self.bins_dict[variable]
            labels = self.labels_dict[variable]

            for i in range(len(labels)):

                rows.append({
                    "variable": variable,
                    "classe": labels[i],
                    "borne_inf": bins[i],
                    "borne_sup": bins[i + 1]
                })

        return pd.DataFrame(rows)
    
    def get_tariff_classes(self):

        output = {}

        for variable in self.bins_dict:

            bins = self.bins_dict[variable]
            labels = self.labels_dict[variable]

            output[variable] = {
                label: (
                    bins[i],
                    bins[i + 1]
                )
                for i, label in enumerate(labels)
            }

        return output