from implementations import *

def build_k_indices(y, k_fold, seed):
    """Return shuffled validation indices for each fold."""
    if k_fold < 2 or k_fold > y.shape[0]:
        raise ValueError("k_fold must be between 2 and the number of samples")

    rng = np.random.default_rng(seed)
    return list(np.array_split(rng.permutation(y.shape[0]), k_fold))


def compute_f1(y_true, y_pred):
    """Compute F1 score for binary labels in {0, 1}."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    if tp == 0:
        return 0.0
    return 2.0 * tp / (2.0 * tp + fp + fn)


def cross_validation_step(y, tx, k_indices, k, lambda_, max_iters, gamma):
    """Train on k-1 folds and return validation probabilities and ground truth."""
    val_idx = k_indices[k]
    train_idx = np.delete(k_indices, k, axis=0).ravel()

    tx_tr, y_tr = tx[train_idx], y[train_idx]
    tx_val, y_val = tx[val_idx], y[val_idx]

    initial_w = np.zeros(tx.shape[1])
    w, _ = reg_logistic_regression(y_tr, tx_tr, lambda_, initial_w, max_iters, gamma)

    # Predict probabilities on the validation fold
    y_val_prob = sigmoid(tx_val.dot(w))
    return y_val, y_val_prob


def cross_validation(
    y, tx, k_fold, lambdas, thresholds=None, max_iters=1000, gamma=0.01, seed=12
):
    """Select (lambda, threshold) maximizing mean validation F1-score.

    ``y`` must contain binary labels encoded as 0 and 1.
    """
    if thresholds is None:
        # Sweep thresholds from 0.10 to 0.50 (step 0.01)
        thresholds = np.linspace(0.10, 0.50, 41)

    k_indices = build_k_indices(y, k_fold, seed)
    
    best_lambda = None
    best_threshold = 0.5
    best_f1 = -1.0

    for lambda_ in lambdas:
        print(f"Using lambda: {lambda_}")
        
        # Train K models once per lambda and store validation probabilities
        fold_results = [
            cross_validation_step(y, tx, k_indices, k, lambda_, max_iters, gamma)
            for k in range(k_fold)
        ]

        # Fast sweep over all candidate thresholds using cached probabilities
        for thresh in thresholds:
            fold_f1s = [
                compute_f1(y_val, (y_val_prob >= thresh).astype(int))
                for y_val, y_val_prob in fold_results
            ]
            mean_f1 = float(np.mean(fold_f1s))

            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_lambda = lambda_
                best_threshold = float(thresh)

    print(f"Best lambda: {best_lambda} | Best threshold: {best_threshold:.2f} | Best CV F1: {best_f1:.4f}")
    return best_lambda, best_threshold, best_f1