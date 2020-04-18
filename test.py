import numpy as np

"""Calculate cross correlation efficiently"""


def cross_correlate(a, f):
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape=s, strides=a.strides * 2)
    # print("Sub matrix")
    # print(subM)
    return np.einsum('ijm,ijmkln->kl', f, subM)


A = np.array([[[5, 5], [0, 0], [0, 0]], [
             [0, 0], [5, 4], [0, 0]], [[0, 0], [0, 0], [5, 3]]])
f = np.array([[[1, 1], [2, 0]], [[0, 1], [0, 0]]])

print(A.shape)
print(f.shape)
res = cross_correlate(A, f)
print(res.shape)
print(res)
