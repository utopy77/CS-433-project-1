import numpy as np


def make_stratified_folds(y, k=5, seed=120705):
    """Return (training row numbers, validation row numbers) for each fold.

    'Stratified' means each group has roughly the same share of +1 and -1.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)

    shuffled_rows = np.concatenate([
        rng.permutation(np.flatnonzero(y == label)) for label in (-1, 1)
    ])

    all_rows = np.arange(y.size)
    folds = []
    for fold in range(k):
        validation_rows = shuffled_rows[fold::k]
        is_training = np.ones(y.size, dtype=bool)
        is_training[validation_rows] = False
        folds.append((all_rows[is_training], validation_rows))
    return folds


def classification_metrics(y_true, y_pred):
    """Return f1, precision, recall and accuracy in a dictionary."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == -1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == -1)))

    return {
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "accuracy": float(np.mean(y_true == y_pred)) if y_true.size else 0.0
    }
