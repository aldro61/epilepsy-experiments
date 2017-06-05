"""
A bit of data exploration

"""
import matplotlib; matplotlib.use('PDF')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set_style("white")

patient_info = pd.read_table("/exec5/GROUP/pacoss/COMMUN/claudia/machine_learning/data/ibd_clusters.pheno.txt", names=["id", "disease"])

patient_info["family"], patient_info["patient_id"] = zip(*(id.split("_") for id in patient_info["id"].values))
del patient_info["id"]

print "> Example counts"
print "Patients: {0:d}".format(patient_info.shape[0])
print "Case: {0:d}".format((patient_info["disease"].values == 1).sum())
print "Control: {0:d}".format((patient_info["disease"].values == 0).sum())
print

print "> Making family plot"
plt.clf()
ax = sns.distplot(patient_info["family"].value_counts(), kde=False)
ax.set_yscale('log')
plt.xlabel("Number of individuals")
plt.ylabel("Number of families")
plt.title("Histogram of the number of individuals per family")
plt.savefig("overview_family_histogram.pdf", bbox_inches="tight")
print
