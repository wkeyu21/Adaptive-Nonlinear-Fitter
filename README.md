# Adaptive Nonlinear Function Combination Fitter for 5-Dimensional Data
## Automatic nonlinear fitting tool for 5-dimensional data (5 independent + 1 dependent variables).

# Overview
A Python tool for automatic nonlinear regression fitting of 5-dimensional independent variable data. It generates nonlinear function combinations dynamically, prunes insignificant terms via statistical tests, searches for the optimal model through multiple rounds, and exports structured results to Excel.

# Features
1. Auto-generate nonlinear functions (power, exp, log, interaction terms)
2. Term pruning via p-value/coefficient threshold
3. Multi-round optimal model search (adjusted R²)
4. Auto standardization/denormalization

# Requirements
Python 3.10+

# Prepare Excel
cols 1-5 = x1~x5, col 6 = y

# Data Format
### File: .xlsx
### Supports missing values (auto-handled)
### Output
### Fitting coefficients/metrics (R², MSE, MAE, RMSE)
### Normalized/original fitting expressions
### Actual vs predicted values

# License
Apache License 2.0

