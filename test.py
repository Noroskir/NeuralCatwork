import cupy as np

from conv_layer import ConvLayer

m = 10
X = np.random.rand(m, 700, 400, 3)

l = ConvLayer(X.shape, 3, 6, 1, 1)
