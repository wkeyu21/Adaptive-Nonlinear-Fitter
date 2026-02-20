
import numpy as np
import pandas as pd

def load_raw_data(df):

    X = df.iloc[:, 0:5].values
    y = df.iloc[:, 5].values
    print(f"Data loading completed: {len(X)} samples, 5 independent variables, 1 dependent variable")

    X_mean, X_std = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
    y_mean, y_std = np.nanmean(y), np.nanstd(y)

    X_norm = np.nan_to_num((X - X_mean) / (X_std + 1e-8), nan=0.0)
    y_norm = np.nan_to_num((y - y_mean) / (y_std + 1e-8), nan=0.0)

    return X, y, X_norm, y_norm, X_mean, X_std, y_mean, y_std