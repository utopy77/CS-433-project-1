import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from helpers import create_csv_submission, load_csv_data
from implementations import (
    compute_logistic_gradient,
    reg_logistic_regression,
    sigmoid,
)
from preprocessing import preprocess
from validation import classification_metrics, make_stratified_folds


def fit_model(
    y01, tx, lambda_, gamma=0.1, block_size=100, max_iters=20000, gradient_tol=0.0001
):
    """Train from zero weights, checking the penalized gradient after each block."""
    w = np.zeros(tx.shape[1])
    iterations, max_gradient = 0, float("inf")
    for iterations in range(block_size, max_iters + 1, block_size):
        w, _ = reg_logistic_regression(y01, tx, lambda_, w, block_size, gamma)
        gradient = compute_logistic_gradient(y01, tx, w) + 2 * lambda_ * w
        max_gradient = float(np.max(np.abs(gradient)))
        if max_gradient < gradient_tol:
            break
        if iterations % 1000 == 0:
            print(f"  {iterations} iterations, max gradient={max_gradient:.6g}", flush=True)
    return w, iterations, max_gradient


def main():
    """Compare five-fold results, log every setting and write a submission."""
    root = Path(__file__).resolve().parent
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    seed = 120705
    lambdas = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    thresholds = [i / 100 for i in range(1, 100)]
    settings = {
        "gamma": 0.1,
        "block_size": 100,
        "max_iters": 20000,
        "gradient_tol": 0.0001,
    }
    features = [
        "_AGE80", "_BMI5", "SEX", "GENHLTH", "BPHIGH4", "DIABETE3",
        "_SMOKER3", "CVDSTRK3", "TOLDHI2", "EXERANY2",
        "PHYSHLTH", "MENTHLTH", "CHCKIDNY", "CHCCOPD1", "ADDEPEV2",
        "HAVARTH3", "QLACTLM2", "USEEQUIP", "DIFFWALK", "EMPLOY1",
        "INCOME2", "EDUCA", "MARITAL", "HAREHAB1", "CVDASPRN",
    ]

    print("Loading data once...", flush=True)
    with (root / "dataset" / "x_train.csv").open(newline="") as source:
        feature_names = next(csv.reader(source))[1:]
    x_train, x_test, y, train_ids, test_ids = load_csv_data(str(root / "dataset"))
    columns = [feature_names.index(name) for name in features]
    x_train, x_test = x_train[:, columns], x_test[:, columns]
    folds = make_stratified_folds(y, k=5, seed=seed)
    scores = np.empty((len(lambdas), len(thresholds), len(folds), 4))
    iterations_used = np.zeros((len(lambdas), len(folds)), dtype=int)
    oof_probabilities = np.empty((len(lambdas), y.size))
    fold_assignments = np.empty(y.size, dtype=int)

    for fold, (train_rows, valid_rows) in enumerate(folds):
        tx_train, tx_valid = preprocess(x_train[train_rows], x_train[valid_rows], features)
        fold_assignments[valid_rows] = fold
        y01 = (y[train_rows] + 1) / 2
        for li, lambda_ in enumerate(lambdas):
            print(f"Fold {fold + 1}/5, lambda={lambda_}: training...", flush=True)
            w, iterations, max_gradient = fit_model(y01, tx_train, lambda_, **settings)
            iterations_used[li, fold] = iterations
            status = " (reached iteration cap)" if iterations == settings["max_iters"] else ""
            print(
                f"  Finished at {iterations} iterations{status}; max gradient={max_gradient:.6g}",
                flush=True,
            )
            probabilities = sigmoid(tx_valid.dot(w))
            oof_probabilities[li, valid_rows] = probabilities
            for ti, threshold in enumerate(thresholds):
                predictions = np.where(probabilities >= threshold, 1, -1)
                metrics = classification_metrics(y[valid_rows], predictions)
                scores[li, ti, fold] = [
                    metrics["f1"], metrics["precision"], metrics["recall"], metrics["accuracy"]
                ]

    mean_f1 = scores[:, :, :, 0].mean(axis=2)
    best_li, best_ti = np.unravel_index(np.argmax(mean_f1), mean_f1.shape)
    best_lambda = lambdas[best_li]
    best_threshold = float(thresholds[best_ti])
    selected = scores[best_li, best_ti]
    records = []
    for li, lambda_ in enumerate(lambdas):
        for ti, threshold in enumerate(thresholds):
            fold_scores = scores[li, ti]
            records.append({
                "timestamp": timestamp,
                "feature_names": "|".join(features),
                "seed": seed,
                "folds": len(folds),
                "initial_weights": "zeros",
                **settings,
                "lambda_": lambda_,
                "threshold": float(threshold),
                **{f"fold_{i + 1}_f1": float(value) for i, value in enumerate(fold_scores[:, 0])},
                "mean_f1": float(fold_scores[:, 0].mean()),
                "std_f1": float(fold_scores[:, 0].std()),
                "mean_precision": float(fold_scores[:, 1].mean()),
                "mean_recall": float(fold_scores[:, 2].mean()),
                "mean_accuracy": float(fold_scores[:, 3].mean()),
                "iterations_used": "|".join(map(str, iterations_used[li])),
                "capped_folds": "|".join(
                    str(i + 1) for i, count in enumerate(iterations_used[li])
                    if count == settings["max_iters"]
                ),
                "selected": (li, ti) == (best_li, best_ti),
            })
    log_path = root / "logs" / "experiments.csv"
    log_path.parent.mkdir(exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    oof_path = log_path.parent / "oof_predictions.npz"
    np.savez_compressed(
        oof_path,
        probabilities=oof_probabilities[best_li],
        train_ids=train_ids,
        labels=y,
        fold_assignments=fold_assignments,
        lambda_=best_lambda,
        threshold=best_threshold,
        seed=seed,
        feature_names=np.asarray(features),
        positive_label=1,
    )

    print(f"Selected lambda={best_lambda}, threshold={best_threshold:.2f}, seed={seed}", flush=True)
    print(
        f"Validation F1={selected[:, 0].mean():.6f} +/- {selected[:, 0].std():.6f}, "
        f"precision={selected[:, 1].mean():.6f}, recall={selected[:, 2].mean():.6f}, "
        f"accuracy={selected[:, 3].mean():.6f}",
        flush=True,
    )
    print("Fold F1: " + ", ".join(f"{value:.6f}" for value in selected[:, 0]), flush=True)
    print("Retraining on all training rows...", flush=True)
    tx_train, tx_test = preprocess(x_train, x_test, features)
    w, iterations, max_gradient = fit_model((y + 1) / 2, tx_train, best_lambda, **settings)
    status = " (reached iteration cap)" if iterations == settings["max_iters"] else ""
    print(f"Final fit: {iterations} iterations{status}; max gradient={max_gradient:.6g}", flush=True)
    predictions = np.where(sigmoid(tx_test.dot(w)) >= best_threshold, 1, -1)
    submission_path = root / "submission.csv"
    create_csv_submission(test_ids, predictions, str(submission_path))
    print(f"Experiment log: {log_path}", flush=True)
    print(f"Out-of-fold predictions: {oof_path}", flush=True)
    print(f"Submission: {submission_path}", flush=True)


if __name__ == "__main__":
    main()
