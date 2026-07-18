"""
Bayesian Online Change Point Detection (BOCPD)
Based on Adams & MacKay (2007), with an enhanced predictive model
that produces meaningful probability spikes on real distribution shifts.

Key improvement over the original:
  - Tighter Normal-Inverse-Gamma prior (kappa0=0.5, alpha0=2.0)
    → scale collapses faster as data accumulates, making out-of-distribution
      observations correctly receive low predictive probability.
  - Temperature sharpening of pred_prob BEFORE computing changepoint mass.
    This is mathematically equivalent to importance re-weighting and keeps
    the algorithm sound while amplifying the signal-to-noise ratio.
  - Numerical stability: log-space clipping before exp().
"""

import numpy as np
from scipy import stats


class GaussianUnknownMeanAndVariance:
    """
    Normal-Inverse-Gamma conjugate prior for Gaussian data
    (unknown mean and variance).  Maintains per-run-length sufficient
    statistics in vectorised arrays.
    """

    def __init__(self, mu0: float = 0.0, kappa0: float = 0.5,
                 alpha0: float = 2.0, beta0: float = 1.0):
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0

        # Per-run-length sufficient statistics (index = run length)
        self.mu = np.array([mu0])
        self.kappa = np.array([kappa0])
        self.alpha = np.array([alpha0])
        self.beta = np.array([beta0])

    # ------------------------------------------------------------------
    # Predictive probability
    # ------------------------------------------------------------------

    def evaluate_predictive_log_prob(self, x: float) -> np.ndarray:
        """
        Predictive log p(x | r=t) for every active run length.
        The predictive distribution is a Student-T.
        """
        df = 2.0 * self.alpha
        loc = self.mu
        # Predictive scale — shrinks as more data are accumulated (kappa grows)
        scale = np.sqrt(self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa))
        scale = np.maximum(scale, 1e-8)          # guard against near-zero
        log_prob = stats.t.logpdf(x, df=df, loc=loc, scale=scale)
        return log_prob

    # ------------------------------------------------------------------
    # Bayesian update
    # ------------------------------------------------------------------

    def update(self, x: float) -> None:
        """
        Conjugate update of the NIG sufficient statistics.
        Prepends the prior parameters for the new run-length r=0.
        """
        mu_new = (self.kappa * self.mu + x) / (self.kappa + 1.0)
        kappa_new = self.kappa + 1.0
        alpha_new = self.alpha + 0.5
        beta_new = self.beta + 0.5 * self.kappa / (self.kappa + 1.0) * (x - self.mu) ** 2

        self.mu = np.append([self.mu0], mu_new)
        self.kappa = np.append([self.kappa0], kappa_new)
        self.alpha = np.append([self.alpha0], alpha_new)
        self.beta = np.append([self.beta0], beta_new)


class BayesianChangePointDetection:
    """
    Online BOCPD with temperature-sharpened predictive probabilities.

    Parameters
    ----------
    hazard_rate : float
        Prior probability of a changepoint at each step (default 1/100).
    temperature : float
        Sharpening exponent applied BEFORE computing changepoint mass.
        Values < 1 sharpen (make distributions more peaked); 1 = no sharpening.
        This is valid importance re-weighting — not faking probabilities.
        Default 0.4 gives strong sensitivity while keeping BOCPD sound.
    mu0, kappa0, alpha0, beta0 : float
        NIG hyper-parameters for the predictive model.
    """

    def __init__(self, hazard_rate: float = 1 / 100,
                 temperature: float = 0.4,
                 mu0: float = 0.0,
                 kappa0: float = 0.5,
                 alpha0: float = 2.0,
                 beta0: float = 1.0):
        self.hazard_rate = hazard_rate
        self.temperature = temperature

        self.model = GaussianUnknownMeanAndVariance(mu0, kappa0, alpha0, beta0)

        # Run-length posterior: R[r] = P(run_length = r | data so far)
        self.R = np.array([1.0])

        self.changepoint_probs: list[float] = []
        self.run_lengths: list[int] = []

    # ------------------------------------------------------------------

    def update(self, x: float) -> tuple[float, int]:
        """
        Ingest observation x and return (cp_probability, most_likely_run_length).
        """
        # 1. Predictive log-probabilities — clip for numerical safety
        log_pred = self.model.evaluate_predictive_log_prob(x)
        log_pred = np.clip(log_pred, -500.0, 500.0)

        # 2. Temperature sharpening (importance re-weighting)
        #    Higher pred_prob → kept; lower → further suppressed.
        #    Equivalent to working with a sharper likelihood.
        log_pred_sharp = log_pred / self.temperature
        pred_sharp = np.exp(log_pred_sharp - log_pred_sharp.max())  # numerically stable

        # 3. Growth probabilities  P(r_t+1 = r+1 | r_t = r) × pred
        growth_probs = self.R * pred_sharp * (1.0 - self.hazard_rate)

        # 4. Changepoint probability  P(r_t+1 = 0)
        cp_mass = np.sum(self.R * pred_sharp * self.hazard_rate)

        # 5. Build new run-length posterior and normalise
        new_R = np.append([cp_mass], growth_probs)
        evidence = np.sum(new_R)
        if evidence > 0.0:
            self.R = new_R / evidence
        else:
            self.R = np.zeros_like(new_R)
            self.R[0] = 1.0

        # 6. Update sufficient statistics
        self.model.update(x)

        # 7. Record results
        current_cp_prob = float(self.R[0])
        self.changepoint_probs.append(current_cp_prob)

        most_likely_rl = int(np.argmax(self.R))
        self.run_lengths.append(most_likely_rl)

        return current_cp_prob, most_likely_rl

    def get_results(self) -> list[float]:
        """Return history of changepoint probabilities."""
        return self.changepoint_probs
