import numpy as np
from config import CONFIG
from function_pools import (
    SINGLE_FUNC_LIST, SINGLE_FUNC_NAMES, SINGLE_FUNC_EXPRS,
    CROSS_FUNC_LIST, CROSS_FUNC_NAMES, CROSS_FUNC_EXPRS
)

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