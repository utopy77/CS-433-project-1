import numpy as np

# ---------------------------------------------------------------------------
# Helpers: mean squared error
# ---------------------------------------------------------------------------


def compute_mse(y, tx, w):
    """Mean squared error (with the 0.5 factor).

    L(w) = 1 / (2N) * sum_n e_n^2,  with e = y - tx @ w.
    Returns a scalar.
    """
    e = y - tx.dot(w)  # residuals, shape (N,)
    return 0.5 * np.mean(e**2)  # = (1 / (2N)) * sum(e^2)


def compute_mse_gradient(y, tx, w):
    """Gradient of ``compute_mse`` with respect to w.

    grad L(w) = -(1 / N) * tx^T e,  with e = y - tx @ w.
    Returns a 1D array of shape (D,).
    """
    e = y - tx.dot(w)  # residuals, shape (N,)
    return -tx.T.dot(e) / y.shape[0]  # shape (D,)


# ---------------------------------------------------------------------------
# Helpers: logistic regression
# ---------------------------------------------------------------------------


def sigmoid(t):
    """Numerically stable logistic sigmoid, applied element-wise.

    sigma(t) = 1 / (1 + exp(-t)).
    For t >= 0 we use 1 / (1 + exp(-t)) and for t < 0 the mathematically
    equivalent exp(t) / (1 + exp(t)); each branch only ever evaluates
    exp() on a non-positive argument, so no overflow can occur.
    """
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)

    pos = t >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-t[pos]))

    exp_t = np.exp(t[~pos])  # safe: t < 0  =>  exp(t) in (0, 1)
    out[~pos] = exp_t / (1.0 + exp_t)

    return out


def compute_logistic_loss(y, tx, w):
    """Average negative log-likelihood for logistic regression, y in {0, 1}.

    L(w) = (1 / N) * sum_n [ log(1 + exp(x_n^T w)) - y_n * x_n^T w ].

    ``np.logaddexp(0, z)`` computes log(1 + exp(z)) in a numerically stable
    way (it never overflows, even for large positive z).
    Returns a scalar.
    """
    z = tx.dot(w)  # scores, shape (N,)
    return np.mean(np.logaddexp(0.0, z) - y * z)


def compute_logistic_gradient(y, tx, w):
    """Gradient of ``compute_logistic_loss`` with respect to w.

    grad L(w) = (1 / N) * tx^T (sigma(tx @ w) - y).
    Returns a 1D array of shape (D,).
    """
    pred = sigmoid(tx.dot(w))  # probabilities, shape (N,)
    return tx.T.dot(pred - y) / y.shape[0]  # shape (D,)


# ---------------------------------------------------------------------------
# 1) Linear regression using gradient descent
# ---------------------------------------------------------------------------


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using full gradient descent on the MSE cost.

    Update rule: w <- w - gamma * grad L(w).

    Parameters
    ----------
    y         : ndarray, shape (N,)
    tx        : ndarray, shape (N, D)
    initial_w : ndarray, shape (D,)  - starting weight vector
    max_iters : int                  - number of gradient steps
    gamma     : float                - step size

    Returns
    -------
    (w, loss) : last weight vector and the MSE evaluated at that w.
                If ``max_iters == 0`` the initial vector and its loss are
                returned.
    """
    w = np.asarray(initial_w, dtype=float).copy()  # never modify the input

    for _ in range(max_iters):
        grad = compute_mse_gradient(y, tx, w)  # gradient at the current w
        w = w - gamma * grad  # gradient-descent step

    loss = compute_mse(y, tx, w)  # loss of the LAST w
    return w, loss


# ---------------------------------------------------------------------------
# 2) Linear regression using stochastic gradient descent
# ---------------------------------------------------------------------------


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using SGD on the MSE cost, mini-batch size 1.

    At every iteration one single data point n is drawn uniformly at random
    and the weights are updated with the gradient computed on that point
    only:  w <- w - gamma * (-(y_n - x_n^T w) * x_n).

    Returns
    -------
    (w, loss) : last weight vector and the MSE over the FULL data set,
                evaluated at that w.
    """
    w = np.asarray(initial_w, dtype=float).copy()
    n_samples = y.shape[0]

    for _ in range(max_iters):
        # Draw a single sample (standard mini-batch size of 1).
        n = np.random.randint(n_samples)
        y_n = y[n : n + 1]  # keep shapes (1,) / (1, D)
        tx_n = tx[n : n + 1]

        grad = compute_mse_gradient(y_n, tx_n, w)  # stochastic gradient
        w = w - gamma * grad  # SGD step

    loss = compute_mse(y, tx, w)  # full-data loss at last w
    return w, loss


# ---------------------------------------------------------------------------
# 3) Least squares regression using normal equations
# ---------------------------------------------------------------------------


def least_squares(y, tx):
    """Least squares solution obtained from the normal equations.

    The minimizer of the MSE satisfies  (tx^T tx) w = tx^T y.
    The system is solved with ``np.linalg.solve`` (``np.linalg.lstsq`` is
    not allowed); if the Gram matrix happens to be singular we fall back on
    the pseudo-inverse, which returns the minimum-norm solution.

    Returns
    -------
    (w, loss) : optimal weights and the corresponding MSE.
    """
    a = tx.T.dot(tx)  # Gram matrix, (D, D)
    b = tx.T.dot(y)  # right-hand side, (D,)

    try:
        w = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:  # singular Gram matrix
        w = np.linalg.pinv(a).dot(b)

    loss = compute_mse(y, tx, w)
    return w, loss


# ---------------------------------------------------------------------------
# 4) Ridge regression using normal equations
# ---------------------------------------------------------------------------


def ridge_regression(y, tx, lambda_):
    """Ridge regression solved through the (regularized) normal equations.

    The penalized cost is
        L(w) = 1 / (2N) * ||y - tx w||^2 + lambda_ * ||w||^2,
    whose stationarity condition reads
        (tx^T tx + 2 N lambda_ I) w = tx^T y.
    The factor ``2 * N * lambda_`` is exactly what makes the penalty
    consistent with the 0.5 / N normalisation of the MSE used here.

    Returns
    -------
    (w, loss) : ridge weights and the MSE WITHOUT the penalty term.
    """
    n_samples, n_features = tx.shape

    lambda_prime = 2.0 * n_samples * lambda_  # rescaled regularizer
    a = tx.T.dot(tx) + lambda_prime * np.eye(n_features)
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)  # a is positive definite
    # as soon as lambda_ > 0
    loss = compute_mse(y, tx, w)  # penalty NOT included
    return w, loss


# ---------------------------------------------------------------------------
# 5) Logistic regression using gradient descent
# ---------------------------------------------------------------------------


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression (y in {0, 1}) trained by gradient descent.

    Cost: average negative log-likelihood (see ``compute_logistic_loss``).
    Update rule: w <- w - gamma * (1 / N) * tx^T (sigma(tx w) - y).

    Returns
    -------
    (w, loss) : last weight vector and the logistic loss at that w.
    """
    w = np.asarray(initial_w, dtype=float).copy()

    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w)  # gradient at current w
        w = w - gamma * grad  # gradient-descent step

    loss = compute_logistic_loss(y, tx, w)  # loss of the LAST w
    return w, loss


# ---------------------------------------------------------------------------
# 6) Regularized logistic regression using gradient descent
# ---------------------------------------------------------------------------


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """L2-regularized logistic regression (y in {0, 1}) by gradient descent.

    Penalized cost:
        J(w) = L(w) + lambda_ * ||w||^2,
    with L the average negative log-likelihood. Its gradient is
        grad J(w) = grad L(w) + 2 * lambda_ * w,
    which is the direction used for the update.

    Returns
    -------
    (w, loss) : last weight vector and the logistic loss L(w) WITHOUT the
                penalty term.
    """
    w = np.asarray(initial_w, dtype=float).copy()

    for _ in range(max_iters):
        # Gradient of the PENALIZED objective.
        grad = compute_logistic_gradient(y, tx, w) + 2.0 * lambda_ * w
        w = w - gamma * grad  # gradient-descent step

    loss = compute_logistic_loss(y, tx, w)  # penalty NOT included
    return w, loss

def build_k_indices(y, k_fold, seed):
    """Return shuffled validation indices for each fold."""
    if k_fold < 2 or k_fold > y.shape[0]:
        raise ValueError("k_fold must be between 2 and the number of samples")

    rng = np.random.default_rng(seed)
    return list(np.array_split(rng.permutation(y.shape[0]), k_fold))


def cross_validation_step(
    y, tx, k_indices, k, lambda_, max_iters=1000, gamma=0.01
):
    """Train regularized logistic regression and evaluate one fold."""
    test_indices = k_indices[k]
    train_indices = np.concatenate(
        [indices for fold, indices in enumerate(k_indices) if fold != k]
    )

    y_tr, y_te = y[train_indices], y[test_indices]
    tx_tr, tx_te = tx[train_indices], tx[test_indices]
    initial_w = np.zeros(tx.shape[1])

    w, _ = reg_logistic_regression(
        y_tr, tx_tr, lambda_, initial_w, max_iters, gamma
    )
    loss_tr = float(compute_logistic_loss(y_tr, tx_tr, w))
    loss_te = float(compute_logistic_loss(y_te, tx_te, w))

    return loss_tr, loss_te


def cross_validation(
    y, tx, k_fold, lambdas, max_iters=1000, gamma=0.01, seed=12
):
    """Select ``lambda`` using k-fold validation for logistic regression.

    Returns the lambda with the lowest mean validation logistic loss and that
    loss. ``y`` must contain binary labels encoded as 0 and 1.
    """
    k_indices = build_k_indices(y, k_fold, seed)
    validation_losses = []

    for lambda_ in lambdas:
        print("Using lambda:", lambda_)
        fold_losses = [
            cross_validation_step(
                y, tx, k_indices, k, lambda_, max_iters, gamma
            )[1]
            for k in range(k_fold)
        ]
        validation_losses.append(np.mean(fold_losses))

    best_index = int(np.argmin(validation_losses))
    return lambdas[best_index], float(validation_losses[best_index])
