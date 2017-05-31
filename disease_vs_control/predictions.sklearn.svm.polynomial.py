"""
Polynomial kernel SVM predictions for each train/test split

"""
import cPickle as c
import h5py as h
import numpy as np
import os

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV


n_cpu = 1

dataset = h.File("data/numpy_dataset.h5", "r")
idx_by_patient_id = {id: i for i, id in enumerate(dataset["example_ids"])}

if not os.path.exists(os.path.join("predictions", "poly_svm")):
    os.mkdir(os.path.join("predictions", "poly_svm"))

for split in os.listdir("splits"):
    print "...{0!s}".format(split)

    if os.path.exists(os.path.join("predictions", "poly_svm", split, "parameters.pkl")):
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

    params = dict(kernel=["poly"],
                  C=np.logspace(-4, 4, 10),
                  degree=[1, 2, 3, 7, 9],
                  coef0=[1],
                  class_weight=["balanced"],
                  probability=[True])

    estimator = GridSearchCV(estimator=SVC(), param_grid=params, n_jobs=n_cpu, cv=10)
    estimator.fit(X_train, y_train)
    train_predictions = estimator.predict(X_train)
    train_predictions_proba = estimator.predict_proba(X_train)[:, 1]
    test_predictions = estimator.predict(X_test)
    test_predictions_proba = estimator.predict_proba(X_test)[:, 1]

    os.mkdir(os.path.join("predictions", "poly_svm", split))
    open(os.path.join("predictions", "poly_svm", split, "train_predictions_binary.tsv"), "w").write("\n".join(train_predictions.astype(np.str)))
    open(os.path.join("predictions", "poly_svm", split, "test_predictions_binary.tsv"), "w").write("\n".join(test_predictions.astype(np.str)))
    open(os.path.join("predictions", "poly_svm", split, "train_predictions_proba.tsv"), "w").write("\n".join(train_predictions_proba.astype(np.str)))
    open(os.path.join("predictions", "poly_svm", split, "test_predictions_proba.tsv"), "w").write("\n".join(test_predictions_proba.astype(np.str)))
    c.dump(estimator.best_params_, open(os.path.join("predictions", "poly_svm", split, "parameters.pkl"), "w"))
    c.dump(estimator.best_estimator_, open(os.path.join("predictions", "poly_svm", split, "model.pkl"), "w"))

    print "......accuracy:", accuracy_score(y_true=y_test, y_pred=test_predictions)
    print "......auc:", roc_auc_score(y_true=y_test, y_score=test_predictions_proba)
