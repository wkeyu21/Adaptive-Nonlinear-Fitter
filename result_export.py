import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def calculate_metrics(y_true, y_pred, n_terms):

    valid_y_mask = ~np.isnan(y_true)
    r2 = r2_score(y_true[valid_y_mask], y_pred[valid_y_mask])
    n_samples = np.sum(valid_y_mask)
    adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_terms - 1)
    mse = mean_squared_error(y_true[valid_y_mask], y_pred[valid_y_mask])
    mae = mean_absolute_error(y_true[valid_y_mask], y_pred[valid_y_mask])
    rmse = np.sqrt(mse)
    residual_std = np.nanstd(y_true - y_pred)

    return {
        "r2": r2,
        "adj_r2": adj_r2,
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "residual_std": residual_std
    }


def export_results_to_excel(
        best_result, X, y, y_fit, X_mean, X_std, y_mean, y_std,
        norm_expr, raw_expr, excel_save_path
):

    param_df = pd.DataFrame({
        "Term Index": range(1, len(best_result["betas"]) + 1),
        "Term Name": best_result["term_names"],
        "Fitting Coefficient β": best_result["betas"].round(6),
        "Term Expression Template": best_result["term_exprs"],
        "Coefficient Contribution Ratio": (np.abs(best_result["betas"]) / np.sum(np.abs(best_result["betas"]))).round(4)
    })

    metrics = calculate_metrics(y, y_fit, len(best_result["betas"]))
    eval_df = pd.DataFrame({
        "Evaluation Metric": [
            "R-squared", "Adjusted R-squared", "Mean Squared Error (MSE)",
            "Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)",
            "Residual Standard Deviation", "Total Rows of Raw Data", "Final Number of Fitting Terms"
        ],
        "Metric Value": [
            round(metrics["r2"], 6), round(metrics["adj_r2"], 6), round(metrics["mse"], 6),
            round(metrics["mae"], 6), round(metrics["rmse"], 6), round(metrics["residual_std"], 6),
            len(X), len(best_result["betas"])
        ],
        "Metric Description": [
            "Closer to 1 is better", "Adjusted for number of terms", "Smaller is better",
            "Smaller is better", "Smaller is better", "Smaller is better",
            "Includes all raw data", "Includes constant term"
        ]
    })

    x_func_df = pd.DataFrame({
        "Independent Variable": [f"x{i + 1}" for i in range(5)],
        "Assigned Single-Factor Function": best_result["x_single_names"],
        "Function Description": ["Compound nonlinear function" if "comp" in f else f for f in
                                 best_result["x_single_names"]]
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

    # export Excel
    with pd.ExcelWriter(excel_save_path, engine='openpyxl') as writer:
        param_df.to_excel(writer, sheet_name="Fitting Coefficient Details", index=False)
        eval_df.to_excel(writer, sheet_name="Model Evaluation Metrics", index=False)
        x_func_df.to_excel(writer, sheet_name="Independent Variable Function Assignment", index=False)
        predict_df.to_excel(writer, sheet_name="Actual vs Predicted Values", index=False)
        expr_df.to_excel(writer, sheet_name="Fitting Expressions", index=False)
        norm_param_df.to_excel(writer, sheet_name="Standardization Parameters", index=False)

    print(f"\n Results exported to Excel: {excel_save_path}")
    return metrics