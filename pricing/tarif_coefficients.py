import re
import warnings


class GLMCoefficients:
    def __init__(self, freq_model, sev_model):
        self.freq_model = freq_model
        self.sev_model = sev_model

        self.coef_freq = freq_model.params
        self.coef_sev = sev_model.params

        self.intercept_freq = self.coef_freq["Intercept"]
        self.intercept_sev = self.coef_sev["Intercept"]

        # Variables réellement présentes dans chaque modèle,
        # extraites des noms de coefficients patsy C(var)[T.mod]
        self.vars_freq = self._extract_variables(self.coef_freq)
        self.vars_sev = self._extract_variables(self.coef_sev)

    @staticmethod
    def _extract_variables(coef_dict):
        pattern = re.compile(r"C\((\w+)\)\[T\.")
        variables = set()
        for key in coef_dict.index:
            match = pattern.match(key)
            if match:
                variables.add(match.group(1))
        return variables

    def get(self, coef_dict, model_vars, model_name, variable, modality):
        # Un coefficient absent pour une variable présente dans le
        # modèle correspond à la modalité de référence : 0 est correct.
        # En revanche, une variable totalement absente du modèle
        # (éliminée par sélection AIC par ex.) doit être signalée,
        # sinon la relativité affichée (1.0) laisse croire à un
        # effet nul testé alors que la variable n'a pas été retenue.
        if variable not in model_vars:
            warnings.warn(
                f"La variable '{variable}' est absente du modèle "
                f"{model_name} : coefficient 0 renvoyé "
                f"(relativité de 1.0 par construction, "
                f"pas un effet estimé).",
                UserWarning,
                stacklevel=3
            )
            return 0.0

        key = f"C({variable})[T.{modality}]"
        return coef_dict.get(key, 0.0)

    def freq(self, var, mod):
        return self.get(
            self.coef_freq, self.vars_freq, "fréquence", var, mod
        )

    def sev(self, var, mod):
        return self.get(
            self.coef_sev, self.vars_sev, "sévérité", var, mod
        )
