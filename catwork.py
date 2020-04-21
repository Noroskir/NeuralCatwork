import numpy as np
import matplotlib.pyplot as plt
import h5py

from layer import Layer
from new_conv_layer import ConvLayer
from pool_layer import PoolLayer


with h5py.File("data/train_catvnoncat.h5", "r") as f:
    X = np.array(f["train_set_x"][:])
    Y = np.array(f["train_set_y"][:])
    classes = np.array(f["list_classes"])
    train_X = X/225
    train_Y = Y
with h5py.File("data/test_catvnoncat.h5", "r") as f:
    X_test = np.array(f["test_set_x"])/255
    Y_test = np.array(f["test_set_y"])

X_train = np.zeros((X.shape[0], X.shape[-1], X.shape[1], X.shape[2]))
for i in range(X.shape[0]):
    for c in range(X.shape[-1]):
        X_train[i, c] = train_X[i, :, :, c]


plt.imshow(X_train[0, 2])
plt.show()


def compute_cost(AL, Y):
    m = Y.shape[0]
    cost = -1/m * np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))
    return np.squeeze(cost)


def train_network(X, Y, rate, iterations):
    cost = []
    t = 1
    l1 = ConvLayer(X.shape, 3, 8, 1, 1, activation='relu')
    l2 = ConvLayer(l1.out_shape, 6, 16, 2, 2, activation='relu')
    dim_in3 = (l2.out_shape[1]*l2.out_shape[2] *
               l2.out_shape[3], l2.out_shape[0])
    l3 = Layer(dim_in3, (16, 1), activation='relu')
    l4 = Layer((16, 1), (8, 1), activation='relu')
    l5 = Layer((8, 1), (1, 1), activation='sigmoid')
    for i in range(iterations):
        print('Iteration:', i)
        X1 = l1.forward(X)
        X2 = l2.forward(X1)
        X2 = X2.reshape(X.shape[0], -1).T
        X3 = l3.forward(X2)
        X4 = l4.forward(X3)
        X5 = l5.forward(X4)
        cost.append(compute_cost(X5, Y))
        print("Cost:", cost[-1])
        dA = - (Y/X5 - (1-Y)/(1-X5))
        dA = l5.backward(dA)
        dA = l4.backward(dA)
        dA = l3.backward(dA)
        dA = dA.reshape(l2.out_shape)
        dA = l2.backward(dA)
        dA = l1.backward(dA)
        l1.update_parameters(rate, t)
        l2.update_parameters(rate, t)
        l3.update_parameters(rate, t)
        l4.update_parameters(rate, t)
        l5.update_parameters(rate, t)
        t += 1

    return cost, (l1, l2, l3, l4, l5)


print("Shape training set X:", train_X.shape)
print("Shape training set Y:", train_Y.shape)

cost, nn = train_network(
    train_X[:32], train_Y[:32], rate=0.00075, iterations=2000)

plt.plot(np.arange(len(cost)), cost)
plt.show()
