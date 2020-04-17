import unittest

import numpy as np
import layer


class TestNeuralNetwork(unittest.TestCase):
    # def test_init_parameters(self):
    #     dim_layers = [2, 4, 4, 1]
    #     activations = ["relu", "lrelu", "sigmoid"]
    #     m = 10
    #     dim_in = (m, 4)
    #     dim_out = (m, 8)
    #     lay = layer.Layer(dim_in, dim_out, activation='relu')
    #     lay.init_parameters()
    #     self.assertEqual(len(nn.parameters), 2*(len(dim_layers)-1))
    #     self.assertEqual(nn.parameters["W1"].shape, (4, 2))
    #     self.assertEqual(nn.parameters["b3"].shape, (1, 1))
    #     self.assertEqual(nn.parameters["b2"].shape, (4, 1))

    def test_forward_propagation(self):
        # test example from coursera deeplearning.ai
        np.random.seed(6)
        X = np.random.randn(5, 4)
        W1 = np.random.randn(4, 5)
        b1 = np.random.randn(4, 1)
        W2 = np.random.randn(3, 4)
        b2 = np.random.randn(3, 1)
        W3 = np.random.randn(1, 3)
        b3 = np.random.randn(1, 1)

        lay1 = layer.Layer((1, 5), (1, 4), activation='relu')
        lay1.W = W1
        lay1.b = b1
        lay2 = layer.Layer((1, 4), (1, 3), activation='relu')
        lay2.W = W2
        lay2.b = b2
        lay3 = layer.Layer((1, 3), (1, 1), activation='sigmoid')
        lay3.W = W3
        lay3.b = b3
        A = lay1.forward(X)
        A = lay2.forward(A)
        A = lay3.forward(A)
        self.assertAlmostEqual(0.03921668, A[0, 0], places=7)
        self.assertAlmostEqual(0.19734387, A[0, 2], places=7)

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
