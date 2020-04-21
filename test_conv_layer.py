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
        self.assertAlmostEqual(Z[0, 0, 0, 0], 3.55928104)

    def test_backward_propagation(self):
        # test example from coursera deeplearning.ai
        np.random.seed(1)
        A_prev = np.random.randn(10, 4, 4, 3)
        W = np.random.randn(2, 2, 3, 8)
        b = np.random.randn(1, 1, 1, 8)
        print()
        l1 = conv_layer.ConvLayer(A_prev.shape, 2, 8, 2, 1, activation='relu')
        l2 = conv_layer.ConvLayer(A_prev.shape, 2, 8, 2, 1, activation='relu')
        l1.W = W.copy()
        l1.b = b.copy()
        l2.W = W
        l2.b = b
        Z1 = l1.forward(A_prev)
        Z2 = l2.forward(A_prev)
        dA1 = l1.conv_backward(Z1)
        dA2 = l2.backward(Z2)
        self.assertAlmostEqual(np.mean(dA2), np.mean(dA1), places=8)
        self.assertAlmostEqual(np.mean(l1.dW), np.mean(l2.dW), places=8)
        self.assertAlmostEqual(np.mean(l1.db), np.mean(l2.db), places=8)
