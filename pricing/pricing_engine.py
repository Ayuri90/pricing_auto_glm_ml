import numpy as np

class PricingEngine:

    def __init__(self, coeffs):
        self.c = coeffs

    def reference_premium(self):
        freq = np.exp(self.c.intercept_freq)
        sev = np.exp(self.c.intercept_sev)
        return freq * sev

    def pure_premium(self, profile, exposure=1.0):
        log_f = self.c.intercept_freq
        log_s = self.c.intercept_sev

        for var, mod in profile.items():
            log_f += self.c.freq(var, mod)
            log_s += self.c.sev(var, mod)

        freq = np.exp(log_f) * exposure
        sev = np.exp(log_s)

        return {
            "freq": freq,
            "sev": sev,
            "premium": freq * sev
        }

    def relative_to_reference(self, profile):
        base = self.reference_premium()
        res = self.pure_premium(profile)
        return res["premium"] / base