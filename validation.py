import numpy as np
SEED = 42

def confusion_counts(y_true, y_pred):
    """Return TP, FP, FN, TN for binary labels in {0, 1}."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    return tp, fp, fn, tn


def accuracy(y_true, y_pred):
    """Fraction of correctly classified samples."""
    return np.mean(y_true == y_pred)


def f1_score(y_true, y_pred):
    """F1 score of the positive class: 2TP / (2TP + FP + FN)."""
    tp, fp, fn, _ = confusion_counts(y_true, y_pred)
    return 2 * tp / (2 * tp + fp + fn)

def auc_score(y_true, scores):
    """AUC via the Mann-Whitney U statistic, with average ranks for tied scores."""
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2   # average rank of each tied group
    ranks = avg_rank[inverse]
    n_pos = np.sum(y_true == 1)
    n_neg = len(y_true) - n_pos
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def stratified_kfold_indices(y, k, seed):
    """Split sample indices into k disjoint folds, preserving the class proportions in each fold."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])   # shuffle the indices of class c
        for i, part in enumerate(np.array_split(idx, k)):
            folds[i].append(part)
    return [np.concatenate(f) for f in folds]

def best_threshold(y_true, scores, grid=np.linspace(0, 1, 201)):
    """Return the threshold in grid that maximizes F1, and the corresponding F1."""
    f1s = np.array([f1_score(y_true, (scores >= t).astype(int)) for t in grid])
    return grid[np.argmax(f1s)], f1s.max()


def cross_validate(x, y, folds, train_and_predict, threshold=None, seed=SEED):
    """k-fold CV. The decision threshold is fixed if `threshold` is given, otherwise it is tuned on an
    inner split of each training part. Returns per-fold arrays of F1, accuracy, AUC and thresholds."""
    f1s, accs, aucs, ths = [], [], [], []
    for i in range(len(folds)):
        va = folds[i]
        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != i])

        t = threshold
        if t is None:
            # Inner split of the training part: one fifth is used to select the threshold
            inner = stratified_kfold_indices(y[tr], 5, seed)
            in_va, in_tr = tr[inner[0]], tr[np.concatenate(inner[1:])]
            t, _ = best_threshold(y[in_va], train_and_predict(x[in_tr], y[in_tr], x[in_va]))

        # Fit on the whole training part and evaluate on the outer validation fold
        scores = train_and_predict(x[tr], y[tr], x[va])
        y_pred = (scores >= t).astype(int)
        f1s.append(f1_score(y[va], y_pred))
        accs.append(accuracy(y[va], y_pred))
        aucs.append(auc_score(y[va], scores))
        ths.append(t)
    return np.array(f1s), np.array(accs), np.array(aucs), np.array(ths)


def print_cv(name, f1s, accs, aucs, ths):
    """Print mean and standard deviation of the CV metrics and of the selected thresholds."""
    print(f"{name:<22} F1: {f1s.mean():.4f} ± {f1s.std():.4f} | acc: {accs.mean():.4f} ± {accs.std():.4f} | "
          f"AUC: {aucs.mean():.4f} ± {aucs.std():.4f} | threshold: {ths.mean():.3f} ± {ths.std():.3f}")

def corrected_se(d, k):
    """Standard error of a mean CV difference, corrected for overlapping training sets (Nadeau-Bengio)."""
    return np.sqrt((1 / k + 1 / (k - 1)) * d.var(ddof=1))