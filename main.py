import pandas as pd
import numpy as np
from config import CONFIG
from term_generator import generate_terms_with_cross_nonlinear
from model_fitting import fit_and_prune_terms, generate_fitting_expr
from data_processing import load_raw_data
from result_export import export_results_to_excel, calculate_metrics

def run_raw_data_fit(file_path, excel_save_path=None):

    try:

        df_raw = pd.read_excel(file_path)
        X, y, X_norm, y_norm, X_mean, X_std, y_mean, y_std = load_raw_data(df_raw)


        best_adj_r2 = -np.inf
        best_result = None

        for round_idx in range(CONFIG["search_rounds"]):
            print(f"\nSearch round {round_idx + 1}/{CONFIG['search_rounds']}")

            terms, term_names, term_exprs, x_single_names = generate_terms_with_cross_nonlinear()

            pruned_terms, pruned_names, pruned_exprs, popt, model_func = fit_and_prune_terms(
                X_norm, y_norm, terms, term_names, term_exprs
            )

            y_fit_norm = model_func(X_norm, *popt)
            y_fit = y_fit_norm * y_std + y_mean
            metrics = calculate_metrics(y, y_fit, len(pruned_terms))

            if metrics["adj_r2"] > best_adj_r2:
                best_adj_r2 = metrics["adj_r2"]
                best_result = {
                    "terms": pruned_terms,
                    "term_names": pruned_names,
                    "term_exprs": pruned_exprs,
                    "x_single_names": x_single_names,
                    "betas": popt,
                    "model_func": model_func,
                    "y_fit": y_fit,
                    "r2": metrics["r2"],
                    "adj_r2": metrics["adj_r2"],
                    "mse": metrics["mse"]
                }


        best = best_result
        y_fit = best["y_fit"]
        norm_expr, raw_expr = generate_fitting_expr(
            best["betas"], best["term_exprs"], X_mean, X_std, y_mean, y_std
        )
        print(f"Optimal Adjusted R-squared: {best['adj_r2']:.6f}")
        print(f"Final fitting expression:\n{raw_expr}")


        if excel_save_path is None:
            excel_save_path = r'D:\your_path\Fitting_Results.xlsx'  # results path
        export_results_to_excel(
            best_result, X, y, y_fit, X_mean, X_std, y_mean, y_std,
            norm_expr, raw_expr, excel_save_path
        )


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