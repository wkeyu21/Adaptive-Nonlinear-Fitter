import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# CONFIG

CONFIG = {
    "max_terms": 10,  # Maximum number of fitting terms
    "min_terms": 0,  # Minimum number of reserved terms
    "search_rounds": 30,  # Number of rounds for function combination search
    "compound_func_prob": 0.2,  # Probability of single-factor compound function
    "enable_triple_terms": False,  # Disable triple-factor terms
    "enable_cross_factor_nonlinear": True,  # Enable nonlinear interaction between factors
    "cross_factor_ratio": 0.7,  # Ratio of nonlinear interaction terms in double-factor slots
    "enable_power_interaction": True,  # Enable x1^x2 power-exponential interaction terms
    "p_value_threshold": 0.05,  # Significance test threshold
    "coeff_threshold": 1e-6,  # Coefficient threshold
    "l2_lambda": 1e-5,  # Slight L2 regularization
    "fit_bounds": (-50, 50),  # Parameter range
    "max_fev": 30000  # Maximum number of fitting iterations
}

# Nonlinear function pool

SINGLE_FUNC_POOL = {
    'pow_0.5': (lambda x: np.sqrt(np.abs(x) + 1e-6), 'sqrt(|{x}| + 1e-6)'),
    'pow_2': (lambda x: x ** 2, '({x})^2'),
    'pow_3': (lambda x: x ** 3, '({x})^3'),
    'pow_neg1': (lambda x: 1 / (np.abs(x) + 1e-6), '1/(|{x}| + 1e-6)'),
    'log1p': (lambda x: np.log1p(np.abs(x)), 'ln(|{x}| + 1)'),
    'exp_neg': (lambda x: np.exp(-np.abs(x)), 'exp(-|{x}|)'),
    'exp_neg2': (lambda x: np.exp(-x ** 2), 'exp(-({x})^2)'),
    'tanh': (lambda x: np.tanh(x), 'tanh({x})'),
    'arctan': (lambda x: np.arctan(x), 'arctan({x})'),
    'linear': (lambda x: x, '{x}')
}

SINGLE_FUNC_NAMES = list(SINGLE_FUNC_POOL.keys())
SINGLE_FUNC_LIST = [v[0] for v in SINGLE_FUNC_POOL.values()]
SINGLE_FUNC_EXPRS = [v[1] for v in SINGLE_FUNC_POOL.values()]

# Interaction function pool between factors

CROSS_FUNC_POOL = {
    'x1_exp_x2': (
        lambda x1, x2: x1 * np.exp(np.clip(x2, -20, 20)),
        '{x1} × exp({x2})'
    ),
    'x2_exp_x1': (
        lambda x1, x2: x2 * np.exp(np.clip(x1, -20, 20)),
        '{x2} × exp({x1})'
    ),
    'x1_pow_x2': (
        lambda x1, x2: (np.abs(x1) + 1e-6) ** x2,
        '(|{x1}| + 1e-6)^({x2})'
    ),
    'x2_pow_x1': (
        lambda x1, x2: (np.abs(x2) + 1e-6) ** x1,
        '(|{x2}| + 1e-6)^({x1})'
    ),
    'exp_x1_sub_x2': (
        lambda x1, x2: np.exp(np.clip(x1 - x2, -20, 20)),
        'exp({x1} - {x2})'
    ),
    'x1_log_x2': (
        lambda x1, x2: x1 * np.log1p(np.abs(x2)),
        '{x1} × ln(|{x2}| + 1)'
    ),
    'x2_log_x1': (
        lambda x1, x2: x2 * np.log1p(np.abs(x1)),
        '{x2} × ln(|{x1}| + 1)'
    ),
    'x1_div_x2': (
        lambda x1, x2: x1 / (np.abs(x2) + 1e-6),
        '{x1} / (|{x2}| + 1e-6)'
    ),
    'x2_div_x1': (
        lambda x1, x2: x2 / (np.abs(x1) + 1e-6),
        '{x2} / (|{x1}| + 1e-6)'
    ),
    'log_x1x2': (
        lambda x1, x2: np.log1p(np.abs(x1 * x2)),
        'ln(|{x1}×{x2}| + 1)'
    ),
    'x1_mul_x2': (
        lambda x1, x2: x1 * x2,
        '{x1} × {x2}'
    )
}

if not CONFIG["enable_power_interaction"]:
    CROSS_FUNC_POOL = {k: v for k, v in CROSS_FUNC_POOL.items() if 'pow' not in k}

CROSS_FUNC_NAMES = list(CROSS_FUNC_POOL.keys())
CROSS_FUNC_LIST = [v[0] for v in CROSS_FUNC_POOL.values()]
CROSS_FUNC_EXPRS = [v[1] for v in CROSS_FUNC_POOL.values()]


# Function combination

def generate_terms_with_cross_nonlinear():
    def get_single_func():
        if np.random.random() < CONFIG["compound_func_prob"]:
            idx1, idx2 = np.random.choice(len(SINGLE_FUNC_LIST), 2, replace=False)
            f, f_expr = SINGLE_FUNC_LIST[idx1], SINGLE_FUNC_EXPRS[idx1]
            g, g_expr = SINGLE_FUNC_LIST[idx2], SINGLE_FUNC_EXPRS[idx2]

            def composite(x, f=f, g=g):
                return f(g(x))

            comp_expr = f_expr.format(x=f"({g_expr.format(x='{x}')})")
            return composite, f"comp({SINGLE_FUNC_NAMES[idx1]}({SINGLE_FUNC_NAMES[idx2]}))", comp_expr
        else:
            idx = np.random.choice(len(SINGLE_FUNC_LIST))
            return SINGLE_FUNC_LIST[idx], SINGLE_FUNC_NAMES[idx], SINGLE_FUNC_EXPRS[idx]

    x_single_funcs = []
    x_single_names = []
    x_single_exprs = []
    for i in range(5):
        func, name, expr = get_single_func()
        x_single_funcs.append(func)
        x_single_names.append(name)
        x_single_exprs.append(expr)

    terms = []
    term_names = []
    term_exprs = []

    terms.append(lambda x1, x2, x3, x4, x5: np.ones_like(x1))
    term_names.append("Constant Term")
    term_exprs.append("1")
    remaining_slots = CONFIG["max_terms"] - 1

    single_count = max(min(int(remaining_slots * 0.4), 3), 1)
    single_idx = np.random.choice(5, single_count, replace=False)
    for idx in single_idx:
        func, name, expr = x_single_funcs[idx], x_single_names[idx], x_single_exprs[idx]
        var_name = f"x'{idx + 1}"
        terms.append(lambda x1, x2, x3, x4, x5, idx=idx, func=func: func([x1, x2, x3, x4, x5][idx]))
        term_names.append(f"x{idx + 1}({name})")
        term_exprs.append(expr.format(x=var_name))
    remaining_slots -= single_count

    if remaining_slots > 0:
        pair_count = remaining_slots
        for _ in range(pair_count):
            i, j = np.random.choice(5, 2, replace=False)
            var_i_name = f"x'{i + 1}"
            var_j_name = f"x'{j + 1}"

            if CONFIG["enable_cross_factor_nonlinear"] and np.random.random() < CONFIG["cross_factor_ratio"]:
                cross_idx = np.random.choice(len(CROSS_FUNC_LIST))
                cross_func = CROSS_FUNC_LIST[cross_idx]
                cross_name = CROSS_FUNC_NAMES[cross_idx]
                cross_expr = CROSS_FUNC_EXPRS[cross_idx]

                terms.append(
                    lambda x1, x2, x3, x4, x5, i=i, j=j, cross_func=cross_func:
                    cross_func([x1, x2, x3, x4, x5][i], [x1, x2, x3, x4, x5][j])
                )
                term_names.append(f"x{i + 1}&x{j + 1}({cross_name})")
                term_exprs.append(cross_expr.format(x1=var_i_name, x2=var_j_name))
            else:
                f1, f2 = x_single_funcs[i], x_single_funcs[j]
                n1, n2 = x_single_names[i], x_single_names[j]
                e1, e2 = x_single_exprs[i], x_single_exprs[j]
                terms.append(
                    lambda x1, x2, x3, x4, x5, i=i, j=j, f1=f1, f2=f2:
                    f1([x1, x2, x3, x4, x5][i]) * f2([x1, x2, x3, x4, x5][j])
                )
                term_names.append(f"x{i + 1}({n1})×x{j + 1}({n2})")
                term_exprs.append(f"[{e1.format(x=var_i_name)}]×[{e2.format(x=var_j_name)}]")
        remaining_slots -= pair_count

    if CONFIG["enable_triple_terms"] and remaining_slots > 0:
        i, j, k = np.random.choice(5, 3, replace=False)
        f1, f2, f3 = x_single_funcs[i], x_single_funcs[j], x_single_funcs[k]
        n1, n2, n3 = x_single_names[i], x_single_names[j], x_single_names[k]
        e1, e2, e3 = x_single_exprs[i], x_single_exprs[j], x_single_exprs[k]
        v1, v2, v3 = f"x'{i + 1}", f"x'{j + 1}", f"x'{k + 1}"
        terms.append(lambda x1, x2, x3, x4, x5, i=i, j=j, k=k, f1=f1, f2=f2, f3=f3:
                     f1([x1, x2, x3, x4, x5][i]) * f2([x1, x2, x3, x4, x5][j]) + f3([x1, x2, x3, x4, x5][k]))
        term_names.append(f"[x{i + 1}×x{j + 1}]+x{k + 1}")
        term_exprs.append(f"([{e1.format(x=v1)}]×[{e2.format(x=v2)}])+[{e3.format(x=v3)}]")

    terms = terms[:CONFIG["max_terms"]]
    term_names = term_names[:CONFIG["max_terms"]]
    term_exprs = term_exprs[:CONFIG["max_terms"]]
    while len(terms) < CONFIG["min_terms"]:
        idx = np.random.choice(5)
        terms.append(lambda x1, x2, x3, x4, x5, idx=idx: [x1, x2, x3, x4, x5][idx])
        term_names.append(f"x{idx + 1}(linear)")
        term_exprs.append(f"x'{idx + 1}")

    return terms, term_names, term_exprs, x_single_names


# Model fitting and automatic pruning

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
        from scipy.optimize import least_squares
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
        coeff_rank = np.argsort(-np.abs(popt[1:])) + 1
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


# Fitting expression generation

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


# Data loading

def load_raw_data(df):
    X = df.iloc[:, 0:5].values
    y = df.iloc[:, 5].values
    print(f"Data loading completed:")

    # Standardization

    X_mean, X_std = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
    y_mean, y_std = np.nanmean(y), np.nanstd(y)

    X_norm = np.nan_to_num((X - X_mean) / (X_std + 1e-8), nan=0.0)
    y_norm = np.nan_to_num((y - y_mean) / (y_std + 1e-8), nan=0.0)

    return X, y, X_norm, y_norm, X_mean, X_std, y_mean, y_std


# Main execution function

def run_raw_data_fit(file_path, excel_save_path=None):
    try:

        df_raw = pd.read_excel(file_path)
        X, y, X_norm, y_norm, X_mean, X_std, y_mean, y_std = load_raw_data(df_raw)

        # Multi-round search for optimal model
        best_adj_r2 = -np.inf
        best_result = None

        for round_idx in range(CONFIG["search_rounds"]):
            terms, term_names, term_exprs, x_single_names = generate_terms_with_cross_nonlinear()
            pruned_terms, pruned_names, pruned_exprs, popt, model_func = fit_and_prune_terms(
                X_norm, y_norm, terms, term_names, term_exprs
            )
            y_fit_norm = model_func(X_norm, *popt)
            y_fit = y_fit_norm * y_std + y_mean

            valid_y_mask = ~np.isnan(y)
            r2 = r2_score(y[valid_y_mask], y_fit[valid_y_mask])
            adj_r2 = 1 - (1 - r2) * (np.sum(valid_y_mask) - 1) / (np.sum(valid_y_mask) - len(popt) - 1)
            mse = mean_squared_error(y[valid_y_mask], y_fit[valid_y_mask])

            if adj_r2 > best_adj_r2:
                best_adj_r2 = adj_r2
                best_result = {
                    "terms": pruned_terms, "term_names": pruned_names, "term_exprs": pruned_exprs,
                    "x_single_names": x_single_names, "betas": popt, "model_func": model_func,
                    "y_fit": y_fit, "r2": r2,
                }

        best = best_result
        y_fit = best["y_fit"]
        valid_y_mask = ~np.isnan(y)
        mae = mean_absolute_error(y[valid_y_mask], y_fit[valid_y_mask])
        rmse = np.sqrt(best["mse"])
        residual_std = np.nanstd(y - y_fit)

        norm_expr, raw_expr = generate_fitting_expr(
            best["betas"], best["term_exprs"], X_mean, X_std, y_mean, y_std
        )
        print("\n Final fitting expression:")

        # Excel result organization
        param_df = pd.DataFrame({
            "Term Index": range(1, len(best["betas"]) + 1),
            "Term Name": best["term_names"],
            "Fitting Coefficient β": best["betas"].round(6),
            "Term Expression Template": best["term_exprs"],
            "Coefficient Contribution Ratio": (np.abs(best["betas"]) / np.sum(np.abs(best["betas"]))).round(4)
        })

        eval_df = pd.DataFrame({
            "Evaluation Metric": ["R-squared","Root Mean Squared Error (RMSE)",
                                  "Total Rows of Raw Data", "Final Number of Fitting Terms"],
            "Metric Value": [round(best["r2"], 6), round(best["adj_r2"], 6), round(best["mse"], 6), round(mae, 6),
                             round(rmse, 6), round(residual_std, 6), len(X), len(best["betas"])],
            "Metric Description": ["Closer to 1 is better", "Adjusted for number of terms", "Smaller is better",
                                   "Smaller is better", "Smaller is better", "Smaller is better",
                                   "Includes all raw data", "Includes constant term"]
        })

        x_func_df = pd.DataFrame({
            "Independent Variable": [f"x{i + 1}" for i in range(5)],
            "Assigned Single-Factor Function": best["x_single_names"],
            "Function Description": ["Compound nonlinear function" if "comp" in f else f for f in
                                     best["x_single_names"]]
        })

        predict_df = pd.DataFrame(X, columns=[f"x{i + 1} Raw Value" for i in range(5)]).round(6)
        predict_df["Sample Index"] = range(1, len(y) + 1)
        predict_df["y Actual Value"] = y.round(6)
        predict_df["y Predicted Value"] = y_fit.round(6)
        predict_df["Residual"] = (y - y_fit).round(6)
        predict_df["Absolute Residual"] = np.abs(y - y_fit).round(6)
        predict_df = predict_df[["Sample Index"] + [col for col in predict_df.columns if col != "Sample Index"]]

        expr_df = pd.DataFrame({
            "Expression Type": ["Normalized Scale Expression", "Raw Scale Expression"],
            "Complete Expression": [norm_expr, raw_expr],
            "Description": ["For understanding model structure", "Directly used for new data prediction"]
        })

        norm_param_df = pd.DataFrame({
            "Parameter Type": ["Independent Variable Mean"] * 5 + ["Independent Variable Standard Deviation"] * 5 + [
                "Dependent Variable Mean", "Dependent Variable Standard Deviation"],
            "Corresponding Variable": [f"x{i + 1}" for i in range(5)] + [f"x{i + 1}" for i in range(5)] + ["y", "y"],
            "Parameter Value": np.hstack([X_mean, X_std, y_mean, y_std]).round(6),
            "Purpose": ["Denormalization/Standardization calculation"] * 12
        })

        if excel_save_path is None:
            excel_save_path = r'D:\your_path\Fitting_Results.xlsx'
        with pd.ExcelWriter(excel_save_path, engine='openpyxl') as writer:
            param_df.to_excel(writer, sheet_name="Fitting Coefficient Details", index=False)
            eval_df.to_excel(writer, sheet_name="Model Evaluation Metrics", index=False)
            x_func_df.to_excel(writer, sheet_name="Independent Variable Function Assignment", index=False)
            predict_df.to_excel(writer, sheet_name="Actual vs Predicted Values", index=False)
            expr_df.to_excel(writer, sheet_name="Fitting Expressions", index=False)
            norm_param_df.to_excel(writer, sheet_name="Standardization Parameters", index=False)
        print(f"\n Results exported to Excel: {excel_save_path}")

        return {
            "Final Number of Terms": len(best["terms"]),
            "Optimal R-squared": best["r2"],
            "Adjusted R-squared": best["adj_r2"],
            "Fitting Coefficients": best["betas"],
            "Fitting Expression": raw_expr,
            "Excel Path": excel_save_path
        }

    except Exception as e:
        print(f"Execution error: {str(e)}")
        return None


if __name__ == '__main__':
    EXCEL_PATH = r'D:\your_path\demo.xlsx'  # file path
    result = run_raw_data_fit(file_path=EXCEL_PATH)