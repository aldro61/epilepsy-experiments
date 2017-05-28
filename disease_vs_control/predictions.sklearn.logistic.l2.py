"""
Logistic regression with L2-norm regularization predictions for each train/test split

"""
import cPickle as c
import h5py as h
import numpy as np
import os

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score


n_cpu = 1

dataset = h.File("data/numpy_dataset.h5", "r")
idx_by_patient_id = {id: i for i, id in enumerate(dataset["example_ids"])}

if not os.path.exists(os.path.join("predictions", "l2.logistic")):
    os.mkdir(os.path.join("predictions", "l2.logistic"))

for split in os.listdir("splits"):
    print "...{0!s}".format(split)

    if os.path.exists(os.path.join("predictions", "l2.logistic", split, "parameters.pkl")):
        continue

    train_idx = [idx_by_patient_id[l.strip()] for l in open(os.path.join("splits", split, "train_ids.tsv"), "r")]
    test_idx = [idx_by_patient_id[l.strip()] for l in open(os.path.join("splits", split, "test_ids.tsv"), "r")]

    X = dataset["features"][...]
    y = dataset["labels"][...]

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    del X, y

    estimator = LogisticRegressionCV(Cs=10, cv=10, penalty="l2", n_jobs=n_cpu)
    estimator.fit(X_train, y_train)
    train_predictions = estimator.predict(X_train)
    train_predictions_proba = estimator.predict_proba(X_train)[:, 1]
    test_predictions = estimator.predict(X_test)
    test_predictions_proba = estimator.predict_proba(X_test)[:, 1]

    os.mkdir(os.path.join("predictions", "l2.logistic", split))
    open(os.path.join("predictions", "l2.logistic", split, "train_predictions_binary.tsv"), "w").write("\n".join(train_predictions.astype(np.str)))
    open(os.path.join("predictions", "l2.logistic", split, "test_predictions_binary.tsv"), "w").write("\n".join(test_predictions.astype(np.str)))
    open(os.path.join("predictions", "l2.logistic", split, "train_predictions_proba.tsv"), "w").write("\n".join(train_predictions_proba.astype(np.str)))
    open(os.path.join("predictions", "l2.logistic", split, "test_predictions_proba.tsv"), "w").write("\n".join(test_predictions_proba.astype(np.str)))
    c.dump(estimator.get_params(), open(os.path.join("predictions", "l2.logistic", split, "parameters.pkl"), "w"))
    c.dump(estimator, open(os.path.join("predictions", "l2.logistic", split, "model.pkl"), "w"))

    print "......accuracy:", accuracy_score(y_true=y_test, y_pred=test_predictions)
    print "......auc:", roc_auc_score(y_true=y_test, y_score=test_predictions_proba)
