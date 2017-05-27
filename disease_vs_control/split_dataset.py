"""
Create random train/test splits

"""
import h5py as h
import numpy as np
import os

n_splits = 10
train_size = 0.66  # proportion
random_seed = 42

if os.path.exists("splits"):
    print "Warning: splits have already been created. If you really want to recreate them, delete the splits directory."
    exit()
else:
    os.mkdir("splits")

dataset = h.File("data/numpy_dataset.h5", "r")
n_train = int(train_size * dataset["features"].shape[0])
example_ids = dataset["example_ids"][...]

random_generator = np.random.RandomState(random_seed)

for split_id in xrange(n_splits):
    idx = np.arange(dataset["features"].shape[0])
    random_generator.shuffle(idx)
    train_idx = idx[: n_train]
    test_idx = idx[n_train: ]

    split_name = "seed_{0:d}".format(split_id)
    os.mkdir(os.path.join("splits", split_name))
    open(os.path.join("splits", split_name, "train_ids.tsv"), "w").write("\n".join(example_ids[train_idx]))
    open(os.path.join("splits", split_name, "test_ids.tsv"), "w").write("\n".join(example_ids[test_idx]))
