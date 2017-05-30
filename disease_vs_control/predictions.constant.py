"""
Constant model predictions for each train/test split

This model predicts the label of the most abundant class in the training set.

"""
import cPickle as c
import h5py as h
import numpy as np
import os

from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, roc_auc_score


class ConstantModel(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        if len(np.unique(y)) > 2:
            raise Exception("I only handle binary classification!")
        self.predicted_value = Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self.predicted_value] * X.shape[0])


dataset = h.File("data/numpy_dataset.h5", "r")
idx_by_patient_id = {id: i for i, id in enumerate(dataset["example_ids"])}

if not os.path.exists(os.path.join("predictions", "constant")):
    os.mkdir(os.path.join("predictions", "constant"))

for split in os.listdir("splits"):
    print "...{0!s}".format(split)

    if os.path.exists(os.path.join("predictions", "constant", split, "parameters.pkl")):
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

    estimator = ConstantModel()
    estimator.fit(X_train, y_train)
    train_predictions = estimator.predict(X_train)
    test_predictions = estimator.predict(X_test)

    os.mkdir(os.path.join("predictions", "constant", split))
    open(os.path.join("predictions", "constant", split, "train_predictions_binary.tsv"), "w").write("\n".join(train_predictions.astype(np.str)))
    open(os.path.join("predictions", "constant", split, "test_predictions_binary.tsv"), "w").write("\n".join(test_predictions.astype(np.str)))
    c.dump(estimator, open(os.path.join("predictions", "constant", split, "model.pkl"), "w"))

    print "......accuracy:", accuracy_score(y_true=y_test, y_pred=test_predictions)
