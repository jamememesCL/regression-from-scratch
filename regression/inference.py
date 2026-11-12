import numpy as np
import pandas as pd


def coef_summary(model, X, y, names=None):
    #Standard errors, t-stats and approx. 95% CIs for OLS coefficients.
    #Var(beta) = sigma^2 (X'X)^-1, with sigma^2 = RSS / (n - p)."""
    Xb = model._add_intercept(X)
    y = np.asarray(y, dtype=float)
    n, p = Xb.shape

    res = y - Xb @ model.beta
    sigma2 = res @ res / (n - p)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(Xb.T @ Xb)))
    t = model.beta / se

    index = ["intercept"] + list(names) if names is not None else None
    return pd.DataFrame({
        "coef": model.beta,
        "std_err": se,
        "t": t,
        "ci_low": model.beta - 1.96 * se,   # normal approx; fine for n > 100
        "ci_high": model.beta + 1.96 * se,
    }, index=index).round(4)

def robust_se(model, X, y):
    #Heteroskedasticity-robust (HC3) standard errors.
    Xb = model._add_intercept(X)
    y = np.asarray(y, dtype=float)
    res = y - Xb @ model.beta
    XtX_inv = np.linalg.inv(Xb.T @ Xb)
    h = np.einsum("ij,jk,ik->i", Xb, XtX_inv, Xb)  # leverage
    meat = (Xb * (res / (1 - h))[:, None]).T @ (Xb * (res / (1 - h))[:, None])
    return np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))