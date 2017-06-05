"""
Compute the linear kernel matrix for the entire data set

Row order is the same as features in the numpy data set

"""
import h5py as h

from sklearn.metrics.pairwise import linear_kernel


save_path = "data/linear.kernel.h5"

print "Loading the feature vectors..."
dataset = h.File("data/numpy_dataset.h5", "r")
X = dataset["features"][...]

print "Computing the kernel matrix..."
kernel_matrix = linear_kernel(X, X)

print "Saving to HDF5 ({0!s}) ...".format(save_path)
f = h.File(save_path, "w")
f.create_dataset("kernel_matrix", data=kernel_matrix)
f.close()