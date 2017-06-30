"""
Predictions for neural networks

"""
import h5py as h
import numpy as np
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam, SGD
from keras.regularizers import l2

dataset = h.File("data/numpy_dataset.h5", "r")
idx_by_patient_id = {id: i for i, id in enumerate(dataset["example_ids"])}

if not os.path.exists(os.path.join("predictions", "random_forest")):
    os.mkdir(os.path.join("predictions", "random_forest"))

for split in os.listdir("splits"):
    print "...{0!s}".format(split)

    if os.path.exists(os.path.join("predictions", "random_forest", split, "parameters.pkl")):
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

    print X_train.shape, X_test.shape, y_train.shape, y_test.shape

    from keras.utils import to_categorical

    y_train = to_categorical(y_train)

    model = Sequential()

    model.add(Dense(units=200, input_dim=X_train.shape[1], activation="relu"))
    model.add(Dropout(0.75))
    model.add(Dense(units=200, activation="relu"))
    model.add(Dropout(0.75))
    model.add(Dense(units=2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.0001),
                  metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, epochs=2000, validation_split=0.2)
