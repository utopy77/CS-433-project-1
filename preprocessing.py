import numpy as np


def preprocess(x_train, x_eval, feature_names):
    """Scale numeric features with missing flags, then encode categories."""
    columns = {name: i for i, name in enumerate(feature_names)}
    train_columns = [np.ones(x_train.shape[0])]
    eval_columns = [np.ones(x_eval.shape[0])]

    for name in ("_AGE80", "_BMI5", "PHYSHLTH", "MENTHLTH"):
        train_values = x_train[:, columns[name]].astype(float)
        eval_values = x_eval[:, columns[name]].astype(float)
        if name in ("PHYSHLTH", "MENTHLTH"):
            train_values[train_values == 88] = 0
            eval_values[eval_values == 88] = 0
            train_values[np.isin(train_values, [77, 99])] = np.nan
            eval_values[np.isin(eval_values, [77, 99])] = np.nan
        train_missing = np.isnan(train_values)
        eval_missing = np.isnan(eval_values)
        observed = train_values[~train_missing]
        median = np.median(observed) if observed.size else 0.0
        train_values = np.where(train_missing, median, train_values)
        eval_values = np.where(eval_missing, median, eval_values)
        mean = train_values.mean()
        std = train_values.std() if train_values.max() != train_values.min() else 1.0
        train_columns.append((train_values - mean) / std)
        eval_columns.append((eval_values - mean) / std)
        train_columns.append(train_missing.astype(float))
        eval_columns.append(eval_missing.astype(float))

    categories = [
        ("SEX", 2),
        ("GENHLTH", 5),
        ("BPHIGH4", 4),
        ("DIABETE3", 4),
        ("_SMOKER3", 4),
        ("CVDSTRK3", 2),
        ("TOLDHI2", 2),
        ("EXERANY2", 2),
        ("CHCKIDNY", 2),
        ("CHCCOPD1", 2),
        ("ADDEPEV2", 2),
        ("HAVARTH3", 2),
        ("QLACTLM2", 2),
        ("USEEQUIP", 2),
        ("DIFFWALK", 2),
        ("EMPLOY1", 8),
        ("INCOME2", 8),
        ("EDUCA", 6),
        ("MARITAL", 6),
    ]
    for name, last_code in categories:
        train_values = x_train[:, columns[name]]
        eval_values = x_eval[:, columns[name]]
        codes = np.arange(1, last_code + 1)
        for code in codes:
            train_columns.append((train_values == code).astype(float))
            eval_columns.append((eval_values == code).astype(float))
        train_columns.append((~np.isin(train_values, codes)).astype(float))
        eval_columns.append((~np.isin(eval_values, codes)).astype(float))

    return np.column_stack(train_columns), np.column_stack(eval_columns)
