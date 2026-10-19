"""Reproduce our AIcrowd submission for Project 1 (prediction of MICHD from BRFSS 2015 data).

Pipeline: recoding and preprocessing of preprocessing.py (configuration chosen in cross-validation),
logistic regression by gradient descent, decision threshold chosen to maximize F1 with the same
procedure used in cross-validation. Usage: python run.py (data in dataset/).
"""
import numpy as np

import preprocessing as pp
from helpers import load_csv_data, create_csv_submission
from implementations import logistic_regression
from validation import SEED, stratified_kfold_indices, best_threshold

DATA_PATH = "dataset/"
OUTPUT_PATH = "submission.csv"
MAX_ITERS = 1000


def sigmoid(t):
    """Logistic function."""
    return 1 / (1 + np.exp(-t))


def fit_logistic(x, y, nominal):
    """Fit the preprocessing on x, then logistic regression by gradient descent with step size 1/L,
    where L = lambda_max(X^T X / n) / 4 is the Lipschitz constant of the gradient. Returns (prep, w)."""
    prep = pp.fit_preprocessing(x, pp.TAU, nominal)
    tx = pp.transform(x, prep)
    L = np.linalg.eigvalsh(tx.T @ tx / len(tx)).max() / 4
    w, _ = logistic_regression(y, tx, np.zeros(tx.shape[1]), MAX_ITERS, 1 / L)
    return prep, w


def predict_proba(x, prep, w):
    """Estimated probabilities P(y = 1 | x)."""
    return sigmoid(pp.transform(x, prep) @ w)


def main():
    x_train, x_test, y_train, _, test_ids = load_csv_data(DATA_PATH)
    col_names = pp.read_column_names(DATA_PATH + "x_train.csv")
    y = ((y_train + 1) / 2).astype(int)               # labels {-1, 1} -> {0, 1}

    xr_train, xr_test, kept_names = pp.recode_data(x_train, x_test, col_names)
    nominal = pp.column_indices(kept_names, pp.NOMINAL)

    # Decision threshold: same procedure as in cross-validation (stratified inner split, 4/5 to fit, 1/5 to choose)
    inner = stratified_kfold_indices(y, 5, SEED)
    in_va, in_tr = inner[0], np.concatenate(inner[1:])
    prep, w = fit_logistic(xr_train[in_tr], y[in_tr], nominal)
    p_va = predict_proba(xr_train[in_va], prep, w)
    threshold, f1_va = best_threshold(y[in_va], p_va)
    print(f"Threshold: {threshold:.3f} | F1 on the inner split: {f1_va:.4f} | "
          f"predicted positive rate: {np.mean(p_va >= threshold):.4f}")

    # Final model on the whole training set
    prep, w = fit_logistic(xr_train, y, nominal)
    p_test = predict_proba(xr_test, prep, w)
    y_pred = np.where(p_test >= threshold, 1, -1)       # labels {0, 1} -> {-1, 1}
    print(f"Test -> features: {len(w)} | predicted positive rate: {np.mean(y_pred == 1):.4f}")

    create_csv_submission(test_ids, y_pred, OUTPUT_PATH)
    print(f"Submission written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
