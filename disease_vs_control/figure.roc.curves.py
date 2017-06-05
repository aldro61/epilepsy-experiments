import h5py as h
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy import interp

from sklearn.metrics import roc_curve

color_map = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33']

dataset = h.File("data/numpy_dataset.h5", "r")
y = dataset["labels"][...]
idx_by_patient_id = {id: i for i, id in enumerate(dataset["example_ids"])}
del dataset

methods = os.listdir("predictions")
splits = os.listdir("splits")


def plot_model_roc(method, color):
    tprs = []
    base_fpr = np.linspace(0, 1, 101)

    for i, split in enumerate(splits):
        train_idx = np.array(
            [idx_by_patient_id[l.strip()] for l in open(os.path.join("splits", split, "train_ids.tsv"), "r")])
        test_idx = np.array(
            [idx_by_patient_id[l.strip()] for l in open(os.path.join("splits", split, "test_ids.tsv"), "r")])
        y_test = y[test_idx]

        if os.path.exists(os.path.join("predictions", method, split, "test_predictions_proba.tsv")):
            test_predictions = np.array([l.strip() for l in open(os.path.join("predictions", method, split, "test_predictions_proba.tsv"), "r")], dtype=np.float64)
        else:
            test_predictions = np.array([l.strip() for l in open(os.path.join("predictions", method, split, "test_predictions_binary.tsv"), "r")], dtype=np.float64)

        fpr, tpr, _ = roc_curve(y_test, test_predictions)

        plt.plot(fpr, tpr, alpha=0.15, color=color)
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    plt.plot(base_fpr, mean_tprs, color=color, label=method.title())
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color=color, alpha=0.3)


plt.figure(figsize=(8, 8))

for i, method in enumerate(methods):
    plot_model_roc(method, color_map[i])

plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.axes().set_aspect('equal', 'datalim')
plt.legend(ncol=len(methods), bbox_to_anchor=(1.05, 1.05))
plt.suptitle("Receiver operating characteristic curves averaged over 10 validation sets")
plt.savefig("figure.roc.curves.pdf", bbox_inches="tight")