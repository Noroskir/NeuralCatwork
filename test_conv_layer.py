import unittest
import numpy as np

import conv_layer


class TestConvLayer(unittest.TestCase):
    def test_forward_propagation(self):
        np.random.seed(1)
        im = np.random.randn(10, 5, 7, 4)
        F = np.random.randn(3, 3, 4, 8)
        b = np.random.randn(1, 1, 1, 8)
        param = {"pad": 1,
                 "stride": 2}

        l1 = conv_layer.ConvLayer(im.shape, 3, 8, 2, 1, activation='relu')
        l1.W = F
        l1.b = b
        Z = l1.forward(im)
        self.assertAlmostEqual(Z[3, 2, 1, 4], 8.25132576, places=8)

    def test_backward_propagation(self):
        # test example from coursera deeplearning.ai
        np.random.seed(1)
        A_prev = np.random.randn(10, 4, 4, 3)
        W = np.random.randn(2, 2, 3, 8)
        b = np.random.randn(1, 1, 1, 8)

        l1 = conv_layer.ConvLayer(A_prev.shape, 2, 8, 2, 2, activation='relu')
        l1.W = W
        l1.b = b
        Z = l1.forward(A_prev)
        dA = l1.backward(Z)
        # self.assertAlmostEqual(np.mean(dA), 1.45243777754, places=8)
        # self.assertAlmostEqual(np.mean(l1.dW), 1.72699145831, places=8)
        # self.assertAlmostEqual(np.mean(l1.db), 7.83923256462, places=8)
