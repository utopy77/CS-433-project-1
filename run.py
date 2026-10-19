import numpy as np
from helpers import *
from implementations import *
from validation import *
from preprocessing import preprocess

tx_tr, tx_te, y_train, train_ids, test_ids = preprocess()
gamma = 0.1
max_iters = 1000
best_lambda = 0.001
best_threshold = 0.19


y_train_01 = (y_train + 1) / 2  # Converts {-1, 1} to {0, 1} for regularized logistic regression
best_lambda,best_threshold, best_f1 = cross_validation(
    y_train_01,
    tx_tr,
    k_fold=5,
    lambdas=np.array([0.1]),
    max_iters=1000,
    gamma=gamma,
)

print("Best lambda:", best_lambda)
print("Best threshold:", best_threshold)
print("Best F1 score:", best_f1)

w, _ = reg_logistic_regression(y_train_01, tx_tr, best_lambda, np.zeros(tx_tr.shape[1]), 1000, gamma)

y_pred = np.where(sigmoid(tx_te.dot(w)) >= best_threshold, 1, -1)

create_csv_submission(
    test_ids,
    y_pred,
    "submission.csv",
)