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