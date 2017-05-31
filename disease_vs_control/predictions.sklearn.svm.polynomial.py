"""
Polynomial kernel SVM predictions for each train/test split

"""
import cPickle as c
import h5py as h
import numpy as np
import os

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, ParameterGrid


n_folds = 10
random_state = np.random.RandomState(42)

dataset = h.File("data/numpy_dataset.h5", "r")
kernel_matrix = h.File("data/linear.kernel.h5", "r")["kernel_matrix"][...]
y = dataset["labels"][...]
idx_by_patient_id = {id: i for i, id in enumerate(dataset["example_ids"])}

if not os.path.exists(os.path.join("predictions", "poly_svm")):
    os.mkdir(os.path.join("predictions", "poly_svm"))

for split in os.listdir("splits"):
    print "...{0!s}".format(split)

    if os.path.exists(os.path.join("predictions", "poly_svm", split, "parameters.pkl")):
        continue

    train_idx = np.array([idx_by_patient_id[l.strip()] for l in open(os.path.join("splits", split, "train_ids.tsv"), "r")])
    test_idx = np.array([idx_by_patient_id[l.strip()] for l in open(os.path.join("splits", split, "test_ids.tsv"), "r")])

    params = ParameterGrid(dict(C=np.logspace(-6, 3, 10),
                                gamma=[1. / kernel_matrix.shape[0]],
                                degree=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                coef0=[1]))

    # Select the best HP by cross-validation
    hp_results = []
    for hps in params:
        folds = np.arange(len(train_idx)) % n_folds
        random_state.shuffle(folds)

        fold_scores = []
        for fold in np.unique(folds):
            fold_train_idx = train_idx[folds != fold]
            fold_test_idx = train_idx[folds == fold]

            poly_kernel_matrix = (float(hps["gamma"]) * kernel_matrix + hps["coef0"])**hps["degree"]
            fold_K_train = poly_kernel_matrix[fold_train_idx][:, fold_train_idx]
            fold_y_train = y[fold_train_idx]
            fold_K_test = poly_kernel_matrix[fold_test_idx][:, fold_train_idx]
            fold_y_test = y[fold_test_idx]

            estimator = SVC(kernel="precomputed", C=hps["C"], probability=True, class_weight="balanced", max_iter=1000000)
            estimator.fit(fold_K_train, fold_y_train)
            fold_scores.append(estimator.score(fold_K_test, fold_y_test))
        print hps, fold_scores[-1]

        hp_results.append(np.mean(fold_scores))  # Cross-validation score (average accuracy)

    # Refit on entire training set and predict on testing set using the best hyperparameters
    best_hps = params[np.argmax(hp_results)]

    poly_kernel_matrix = (best_hps["gamma"] * kernel_matrix + best_hps["coef0"]) ** best_hps["degree"]
    K_train = poly_kernel_matrix[train_idx][:, train_idx]
    y_train = y[train_idx]
    K_test = poly_kernel_matrix[test_idx][:, train_idx]
    y_test = y[test_idx]

    estimator = SVC(C=best_hps["C"], kernel="precomputed", probability=True, class_weight="balanced", max_iter=1000000)
    estimator.fit(K_train, y_train)
    train_predictions = estimator.predict(K_train)
    train_predictions_proba = estimator.predict_proba(K_train)[:, 1]
    test_predictions = estimator.predict(K_test)
    test_predictions_proba = estimator.predict_proba(K_test)[:, 1]

    os.mkdir(os.path.join("predictions", "poly_svm", split))
    open(os.path.join("predictions", "poly_svm", split, "train_predictions_binary.tsv"), "w").write("\n".join(train_predictions.astype(np.str)))
    open(os.path.join("predictions", "poly_svm", split, "test_predictions_binary.tsv"), "w").write("\n".join(test_predictions.astype(np.str)))
    open(os.path.join("predictions", "poly_svm", split, "train_predictions_proba.tsv"), "w").write("\n".join(train_predictions_proba.astype(np.str)))
    open(os.path.join("predictions", "poly_svm", split, "test_predictions_proba.tsv"), "w").write("\n".join(test_predictions_proba.astype(np.str)))
    c.dump(best_hps, open(os.path.join("predictions", "poly_svm", split, "parameters.pkl"), "w"))
    c.dump(estimator, open(os.path.join("predictions", "poly_svm", split, "model.pkl"), "w"))
    c.dump(zip(params, hp_results), open(os.path.join("predictions", "poly_svm", split, "cv_results.pkl"), "w"))

    print "......accuracy:", accuracy_score(y_true=y_test, y_pred=test_predictions)
    print "......auc:", roc_auc_score(y_true=y_test, y_score=test_predictions_proba)
