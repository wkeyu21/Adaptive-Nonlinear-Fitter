import numpy as np
from scipy.optimize import curve_fit, least_squares
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from config import CONFIG

def build_model(terms):

    def model(x, *betas):
        x1, x2, x3, x4, x5 = x.T
        y_pred = np.zeros_like(x1)
        for i in range(len(betas)):
            y_pred += betas[i] * terms[i](x1, x2, x3, x4, x5)
        return y_pred

    return model

def fit_and_prune_terms(X_norm, y_norm, terms, term_names, term_exprs):

    n_terms = len(terms)
    model_func = build_model(terms)
    p0 = np.array([np.mean(y_norm)] + [1.0] * (n_terms - 1))
    bounds = (
        [CONFIG["fit_bounds"][0]] * n_terms,
        [CONFIG["fit_bounds"][1]] * n_terms
    )

    try:
        popt, _ = curve_fit(
            model_func, X_norm, y_norm,
            p0=p0, bounds=bounds,
            maxfev=CONFIG["max_fev"],
            ftol=1e-6, xtol=1e-6,
            method='trf'
        )
    except:
        def loss(betas):
            y_pred = model_func(X_norm, *betas)
            residual = y_norm - y_pred
            l2_penalty = CONFIG["l2_lambda"] * np.sum(betas[1:] ** 2)
            return np.sum(residual ** 2) + l2_penalty

        res = least_squares(loss, x0=p0, bounds=bounds, max_nfev=CONFIG["max_fev"])
        popt = res.x

    X_features = np.zeros((len(X_norm), n_terms))
    x1, x2, x3, x4, x5 = X_norm.T
    for i in range(n_terms):
        X_features[:, i] = terms[i](x1, x2, x3, x4, x5)

    ols_model = OLS(y_norm, add_constant(X_features[:, 1:])).fit()
    p_values = np.hstack([ols_model.pvalues[0], ols_model.pvalues[1:]])

    valid_mask = np.zeros(n_terms, dtype=bool)
    valid_mask[0] = True
    for i in range(1, n_terms):
        coeff_abs = np.abs(popt[i])
        p_val = p_values[i]
        if coeff_abs > CONFIG["coeff_threshold"] and p_val < CONFIG["p_value_threshold"]:
            valid_mask[i] = True

    pruned_terms = [terms[i] for i in range(n_terms) if valid_mask[i]]
    pruned_names = [term_names[i] for i in range(n_terms) if valid_mask[i]]
    pruned_exprs = [term_exprs[i] for i in range(n_terms) if valid_mask[i]]

    if len(pruned_terms) < CONFIG["min_terms"]:
        coeff_rank = np.argsort(-np.abs(popt[1:])) + 1  # 按系数绝对值降序
        for i in coeff_rank:
            if len(pruned_terms) >= CONFIG["min_terms"]:
                break
            if not valid_mask[i]:
                pruned_terms.append(terms[i])
                pruned_names.append(term_names[i])
                pruned_exprs.append(term_exprs[i])
                valid_mask[i] = True

    pruned_model = build_model(pruned_terms)
    pruned_p0 = popt[valid_mask]
    pruned_bounds = (
        [CONFIG["fit_bounds"][0]] * len(pruned_terms),
        [CONFIG["fit_bounds"][1]] * len(pruned_terms)
    )
    pruned_popt, _ = curve_fit(
        pruned_model, X_norm, y_norm,
        p0=pruned_p0, bounds=pruned_bounds,
        maxfev=CONFIG["max_fev"], ftol=1e-6
    )

    print(
        f"   Term pruning: {len(terms)} initial terms → {len(pruned_terms)} pruned terms ({len(terms) - len(pruned_terms)} invalid terms removed)")
    return pruned_terms, pruned_names, pruned_exprs, pruned_popt, pruned_model

def generate_fitting_expr(betas, term_exprs, X_mean, X_std, y_mean, y_std):

    norm_terms = []
    for i in range(len(betas)):
        coeff = betas[i]
        if abs(coeff) < 1e-6:
            continue
        term_str = f"({coeff:.6f}) × {term_exprs[i]}"
        norm_terms.append(term_str)
    norm_y_prime = " + ".join(norm_terms) if norm_terms else "0"
    norm_expr = f"y' = {norm_y_prime}"

    var_replace = {}
    for i in range(5):
        var_prime = f"x'{i + 1}"
        mu, sigma = X_mean[i], X_std[i]
        var_replace[var_prime] = f"(x{i + 1} - {mu:.6f})/{sigma:.6f}"

    raw_y_prime = norm_y_prime
    for var_p, raw_expr in var_replace.items():
        raw_y_prime = raw_y_prime.replace(var_p, raw_expr)
    raw_expr = f"y = ({raw_y_prime}) × {y_std:.6f} + {y_mean:.6f}"

    return norm_expr, raw_expr