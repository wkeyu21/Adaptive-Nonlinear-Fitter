# function_pools.py
import numpy as np
from config import CONFIG

# Nonlinear function pool (single factor)
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