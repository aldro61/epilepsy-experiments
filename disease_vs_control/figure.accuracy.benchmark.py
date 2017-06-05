"""

"""
import h5py as h
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from collections import defaultdict


color_map = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33']

dataset = h.File("data/numpy_dataset.h5", "r")
y = dataset["labels"][...]
idx_by_patient_id = {id: i for i, id in enumerate(dataset["example_ids"])}
del dataset

methods = os.listdir("predictions")
splits = os.listdir("splits")

plt.clf()
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.set_size_inches(7.5, 6)

method_is_plotted = defaultdict(bool)
method_scores = defaultdict(list)
for i, split in enumerate(splits):
    train_idx = np.array([idx_by_patient_id[l.strip()] for l in open(os.path.join("splits", split, "train_ids.tsv"), "r")])
    test_idx = np.array([idx_by_patient_id[l.strip()] for l in open(os.path.join("splits", split, "test_ids.tsv"), "r")])
    y_train = y[train_idx]
    y_test = y[test_idx]

    ax1.axvline(i, linestyle="--", color="lightgrey")

    for j, method in enumerate(methods):
        if not os.path.exists(os.path.join("predictions", method, split)):
            continue

        train_predictions = np.array([l.strip() for l in open(os.path.join("predictions", method, split, "train_predictions_binary.tsv"), "r")], dtype=np.float64)
        test_predictions = np.array([l.strip() for l in open(os.path.join("predictions", method, split, "test_predictions_binary.tsv"), "r")], dtype=np.float64)

        from sklearn.metrics import accuracy_score
        score = accuracy_score(y_true=y_test, y_pred=test_predictions)
        method_scores[method].append(score)
        ax1.scatter([i], [score], edgecolor="black", linewidth=1,
                    facecolor=color_map[j], s=50, alpha=0.7, label=method.title() if not method_is_plotted[method] else None,
                    zorder=10)
        method_is_plotted[method] = True

for j, method in enumerate(methods):
    ax1.axhline(np.mean(method_scores[method]), color=color_map[j])

ax1.set_ylabel("Accuracy")

method_is_plotted = defaultdict(bool)
method_scores = defaultdict(list)
for i, split in enumerate(splits):
    train_idx = np.array([idx_by_patient_id[l.strip()] for l in open(os.path.join("splits", split, "train_ids.tsv"), "r")])
    test_idx = np.array([idx_by_patient_id[l.strip()] for l in open(os.path.join("splits", split, "test_ids.tsv"), "r")])
    y_train = y[train_idx]
    y_test = y[test_idx]

    ax2.axvline(i, linestyle="--", color="lightgrey")

    for j, method in enumerate(methods):
        if not os.path.exists(os.path.join("predictions", method, split)):
            continue

        if os.path.exists(os.path.join("predictions", method, split, "train_predictions_proba.tsv")):
            train_predictions = np.array([l.strip() for l in open(os.path.join("predictions", method, split, "train_predictions_proba.tsv"), "r")], dtype=np.float64)
            test_predictions = np.array([l.strip() for l in open(os.path.join("predictions", method, split, "test_predictions_proba.tsv"), "r")], dtype=np.float64)
        else:
            train_predictions = np.array([l.strip() for l in open(os.path.join("predictions", method, split, "train_predictions_binary.tsv"), "r")], dtype=np.float64)
            test_predictions = np.array([l.strip() for l in open(os.path.join("predictions", method, split, "test_predictions_binary.tsv"), "r")], dtype=np.float64)

        from sklearn.metrics import roc_auc_score
        score = roc_auc_score(y_true=y_test, y_score=test_predictions)
        method_scores[method].append(score)
        ax2.scatter([i], [score], edgecolor="black", linewidth=1,
                    facecolor=color_map[j], s=50, alpha=0.7, label=method.title() if not method_is_plotted[method] else None,
                    zorder=10)
        method_is_plotted[method] = True

for j, method in enumerate(methods):
    ax2.axhline(np.mean(method_scores[method]), color=color_map[j])

ax2.set_ylabel("ROC AUC")

plt.xlim([-1, 10])
plt.xticks(np.arange(10))
plt.xlabel("Validation Set")

plt.legend(bbox_to_anchor=(1.1, 2.4), ncol=len(methods))
plt.savefig("figure.accuracy.benchmark.pdf", bbox_inches="tight")