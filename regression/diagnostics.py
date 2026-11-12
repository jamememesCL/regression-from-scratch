from statistics import NormalDist
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def residuals(y, y_pred):
    return np.asarray(y) - np.asarray(y_pred)


def qq_points(res):
    #Theoretical normal quantiles vs. sorted standardized residuals.
    res = np.asarray(res)
    n = len(res)
    z = (res - res.mean()) / res.std(ddof=1)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical = np.array([NormalDist().inv_cdf(p) for p in probs])
    return theoretical, np.sort(z)


def plot_diagnostics(y, y_pred, title="Residual diagnostics", save_path=None):
    res = residuals(y, y_pred)
    theoretical, sample = qq_points(res)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1) Residuals vs. fitted: look for patterns or a funnel shape
    axes[0].scatter(y_pred, res, alpha=0.6)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set(xlabel="Fitted values", ylabel="Residuals", title="Residuals vs. fitted")

    # 2) Histogram: should look roughly bell-shaped
    axes[1].hist(res, bins=25, edgecolor="black")
    axes[1].set(xlabel="Residual", ylabel="Frequency", title="Residual distribution")

    # 3) Q-Q plot: points should follow the diagonal
    axes[2].scatter(theoretical, sample, alpha=0.6)
    lims = [theoretical.min(), theoretical.max()]
    axes[2].plot(lims, lims, color="red", linestyle="--")
    axes[2].set(xlabel="Theoretical quantiles", ylabel="Sample quantiles", title="Normal Q-Q plot")

    fig.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()