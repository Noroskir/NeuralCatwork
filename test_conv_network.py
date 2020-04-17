import unittest

import numpy as np
import conv_network


class TestConvNeuralNetwork(unittest.TestCase):
    def test_conv_forward(self):
        # test example from coursera deeplearning.ai
        np.random.seed(1)
        im = np.random.randn(10, 5, 7, 4)
        F = np.random.randn(3, 3, 4, 8)
        b = np.random.randn(1, 1, 1, 8)
        param = {"pad": 1,
                 "stride": 2}

        cnn = conv_network.ConvNeuralNetwork((4, 4, 1), ("relu", "sigmoid"))
        Z, cache = cnn.conv_forward(im, F, b, param)
        self.assertAlmostEqual(np.mean(Z), 0.692360880758, places=8)
        self.assertAlmostEqual(Z[3, 2, 1, 0], -1.28912231, places=8)
        self.assertAlmostEqual(Z[3, 2, 1, 4], 8.25132576, places=8)

    def test_conv_backward(self):
        # test example from coursera deeplearning.ai
        np.random.seed(1)
        A_prev = np.random.randn(10, 4, 4, 3)
        W = np.random.randn(2, 2, 3, 8)
        b = np.random.randn(1, 1, 1, 8)
        hparameters = {"pad": 2,
                       "stride": 2}

        cnn = conv_network.ConvNeuralNetwork((4, 4, 1), ("relu", "sigmoid"))
        Z, cache = cnn.conv_forward(A_prev, W, b, hparameters)

        # Test conv_backward
        dA, dW, db = cnn.conv_backward(Z, cache)
        self.assertAlmostEqual(np.mean(dA), 1.45243777754, places=8)
        self.assertAlmostEqual(np.mean(dW), 1.72699145831, places=8)
        self.assertAlmostEqual(np.mean(db), 7.83923256462, places=8)

    def test_pool_forward(self):
        np.random.seed(1)
        A_prev = np.random.randn(2, 5, 5, 3)
        hparameters = {"stride": 1, "f": 3}

        cnn = conv_network.ConvNeuralNetwork((4, 4, 1), ("relu", "sigmoid"))
        A, cache = cnn.pool_forward(A_prev, hparameters)
        self.assertAlmostEqual(A[0, 0, 0, 0], 1.74481176, places=8)
        self.assertAlmostEqual(A[1, 0, 0, 2], 0.82797464, places=8)
        self.assertAlmostEqual(A[1, 2, 2, 2], 0.79280687, places=8)
        A, cache = cnn.pool_forward(A_prev, hparameters, mode="average")
        self.assertAlmostEqual(A[0, 2, 1, 0], -2.47157416e-01, places=8)
        self.assertAlmostEqual(A[0, 0, 0, 0], -3.01046719e-02, places=8)
        self.assertAlmostEqual(A[1, 1, 1, 1], -2.34937338e-01, places=8)

    def test_pool_backward(self):
        np.random.seed(1)
        A_prev = np.random.randn(5, 5, 3, 2)
        hparameters = {"stride": 1, "f": 2}
        dA = np.random.randn(5, 4, 2, 2)

        cnn = conv_network.ConvNeuralNetwork((4, 4, 1), ("relu", "sigmoid"))
        A, cache = cnn.pool_forward(A_prev, hparameters)
        dA_prev = cnn.pool_backward(dA, cache, mode="max")
        self.assertAlmostEqual(dA_prev[1, 1, 0, 0], 0.0, places=8)
        self.assertAlmostEqual(dA_prev[1, 1, 1, 0], 5.05844394, places=8)
        self.assertAlmostEqual(dA_prev[1, 1, 1, 1],  -1.68282702, places=8)
        dA_prev = cnn.pool_backward(dA, cache, mode="average")
        self.assertAlmostEqual(dA_prev[1, 1, 0, 0], 0.08485462, places=8)
        self.assertAlmostEqual(dA_prev[1, 1, 1, 0], 1.26461098, places=8)
        self.assertAlmostEqual(dA_prev[1, 1, 1, 1], -0.25749373, places=8)


# def test_init_parameters(self):
#     dim_layers = [2, 4, 4, 1]
#     activations = ["relu", "lrelu", "sigmoid"]
#     nn = network.NeuralNetwork(dim_layers, activations)
#     nn.init_parameters()
#     self.assertEqual(len(nn.parameters), 2*(len(dim_layers)-1))
#     self.assertEqual(nn.parameters["W1"].shape, (4, 2))
#     self.assertEqual(nn.parameters["b3"].shape, (1, 1))
#     self.assertEqual(nn.parameters["b2"].shape, (4, 1))

# def test_forward_propagation(self):
#     # test example from coursera deeplearning.ai
#     np.random.seed(6)
#     X = np.random.randn(5, 4)
#     W1 = np.random.randn(4, 5)
#     b1 = np.random.randn(4, 1)
#     W2 = np.random.randn(3, 4)
#     b2 = np.random.randn(3, 1)
#     W3 = np.random.randn(1, 3)
#     b3 = np.random.randn(1, 1)
#     parameters = {"W1": W1,
#                   "b1": b1,
#                   "W2": W2,
#                   "b2": b2,
#                   "W3": W3,
#                   "b3": b3}

#     nn = network.NeuralNetwork([5, 4, 3, 1], ["relu", "relu", "sigmoid"])
#     nn.parameters = parameters
#     A, caches = nn.forward_propagation(X)
#     self.assertEqual(len(caches), 3)
#     self.assertAlmostEqual(0.03921668, A[0, 0], places=7)
#     self.assertAlmostEqual(0.19734387, A[0, 2], places=7)

# def test_backward_propagation(self):
#     # test example from coursera deeplearning.ai
#     np.random.seed(3)
#     AL = np.random.randn(1, 2)
#     Y = np.array([[1, 0]])

#     A1 = np.random.randn(4, 2)
#     W1 = np.random.randn(3, 4)
#     b1 = np.random.randn(3, 1)
#     Z1 = np.random.randn(3, 2)
#     linear_cache_activation_1 = (Z1, A1)

#     A2 = np.random.randn(3, 2)
#     W2 = np.random.randn(1, 3)
#     b2 = np.random.randn(1, 1)
#     Z2 = np.random.randn(1, 2)
#     linear_cache_activation_2 = (Z2, A2)

#     caches = (linear_cache_activation_1, linear_cache_activation_2)
#     nn = network.NeuralNetwork([3, 3, 1], ["relu", "sigmoid"])
#     nn.parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
#     grads = nn.backward_propagation(AL, Y, caches)
#     self.assertAlmostEqual(0.41010002, grads["dW1"][0, 0], places=7)
#     self.assertAlmostEqual(0.01005865, grads["dW1"][2, 1], places=7)
#     self.assertAlmostEqual(-0.02835349, grads["db1"][2, 0], places=7)
