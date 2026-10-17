import numpy as np


def preprocess(x_train, x_eval, feature_names):
    """Fill and scale numeric features from training data, then encode categories."""
    columns = {name: i for i, name in enumerate(feature_names)}
    train_columns = [np.ones(x_train.shape[0])]
    eval_columns = [np.ones(x_eval.shape[0])]

    for name in ("_AGE80", "_BMI5"):
        train_values = x_train[:, columns[name]].astype(float)
        eval_values = x_eval[:, columns[name]].astype(float)
        observed = train_values[~np.isnan(train_values)]
        median = np.median(observed) if observed.size else 0.0
        train_values = np.where(np.isnan(train_values), median, train_values)
        eval_values = np.where(np.isnan(eval_values), median, eval_values)
        mean = train_values.mean()
        std = train_values.std() if train_values.max() != train_values.min() else 1.0
        train_columns.append((train_values - mean) / std)
        eval_columns.append((eval_values - mean) / std)

    categories = [
        ("SEX", 2),
        ("GENHLTH", 5),
        ("BPHIGH4", 4),
        ("DIABETE3", 4),
        ("_SMOKER3", 4),
        ("CVDSTRK3", 2),
        ("TOLDHI2", 2),
        ("EXERANY2", 2),
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
