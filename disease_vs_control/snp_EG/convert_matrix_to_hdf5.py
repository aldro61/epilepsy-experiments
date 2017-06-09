"""
Reads the data (haplotypes and phenotypes) and converts it to numpy arrays.
The data is then saved in an HDF5 file.

"""
import h5py as h
import numpy as np
import pandas as pd

variant_file = "data/genotype.matrix"
phenotype_file = "data/phenotypes.tsv"

label_by_patient_id = {l.strip().split("\t")[0]: int(l.strip().split("\t")[1]) for l in open(phenotype_file, "r")}

feature_values = []
feature_names = []
with open(variant_file, "r") as f:
    patient_ids = f.next().strip().split("\t")[1:]
    for i, l in enumerate(f):
        spt = l.strip().split("\t")
        feature_name = spt[0]
        patient_vals = np.asarray(spt[1:], dtype=np.uint8)
        feature_values.append(patient_vals)
        feature_names.append(feature_name)
        if i % 10000 == 0:
            print "Line", i

X = np.array(feature_values).T.copy()
y = [label_by_patient_id[id] for id in patient_ids]

with h.File("numpy_dataset.h5", "w") as f:
    f.create_dataset(name="example_ids", data=patient_ids)
    f.create_dataset(name="features", data=X)
    f.create_dataset(name="feature_names", data=feature_names)
    f.create_dataset(name="labels", data=y)
