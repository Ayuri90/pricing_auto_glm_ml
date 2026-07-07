class LGBPremiumModel:
    """
    Prime pure = fréquence annuelle × exposition × sévérité.

    Le modèle de fréquence ML prédit un taux par unité
    d'exposition : l'exposition doit être fournie pour
    revenir à un nombre de sinistres attendu par police.
    """

    @staticmethod
    def compute_premium(
        freq_model,
        sev_model,
        X,
        exposure
    ):

        freq_rate = freq_model.predict(X)

        sev_pred = sev_model.predict(X)

        premium = freq_rate * exposure * sev_pred

        return premium
