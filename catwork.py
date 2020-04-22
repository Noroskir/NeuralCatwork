import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle
from layer import Layer
from conv_layer import ConvLayer
from pool_layer import PoolLayer


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        """Add a layer to the network.
        Args:
            layer (Layer): layer of the NN
        Returns:
            None
        """
        self.layers.append(layer)

    def create_mini_batches(self, X, Y, size):
        """Create mini batches from the input data."""
        m = X.shape[0]
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation]
        shuffled_Y = Y[:, permutation]

        mini_batches = []

        N_full_batches = int(m/size)
        for i in range(N_full_batches):
            mini_X = shuffled_X[i*size:(i+1)*size]
            mini_Y = shuffled_Y[:, i*size:(i+1)*size]
            mini_batches.append((mini_X, mini_Y))

        if m % size != 0:
            mini_X = shuffled_X[N_full_batches*size:]
            mini_Y = shuffled_Y[:, N_full_batches*size:]
            mini_batches.append((mini_X, mini_Y))

        return mini_batches

    def compute_cost(self, AL, Y, lamb):
        """Compute cross entropy cost with L2 regularisation
        Args:
            AL (np.array): array of predictions from the last layer
            Y (np.array): true labels
            lamb (float): L2 regularisation factor
        Returns:
            float: cross entropy cost
        """
        m = AL.shape[1]
        L2 = 0
        print(AL)
        print(Y)
        for l in self.layers:
            L2 += lamb/m/2*np.sum(np.square(l.W))
        cost = -1/m * np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL)) + L2
        return np.squeeze(cost)

    def deriv_cost(self, AL, Y):
        """Derivative of the cross entropy cost
        Args:
            AL (np.array): array of predictions from the last layer
            Y (np.array): true labels
        Returns:
            np.array: dAL derivative of last activation.
        """
        m = AL.shape[1]
        return - 1 / m * (Y/AL - (1-Y)/(1-AL))

    def train_network(self, X, Y, rate, epochs, lamb=0.1):
        cost = []
        t = 1
        mini_batches = self.create_mini_batches(X, Y, size=32)
        for i in range(epochs):
            costSum = 0
            for mX, mY in mini_batches:
                for l in self.layers:
                    mX = l.forward(mX)
                costSum += self.compute_cost(mX, mY, lamb=lamb)
                dA = self.deriv_cost(mX, mY)
                for l in reversed(self.layers):
                    dA = l.backward(dA)
                    l.update_parameters(rate, t)
                t += 1
            cost.append(costSum)
            print("Cost of epoch {:}: {:.5f}".format(i, cost[-1]))
        return cost

    def save_network(self, filename):
        """Save the trained layers of the network."""
        filehandler = open(filename, 'wb')
        pickle.dump(self.layers, filehandler)

    def load_network(self, filename):
        """Load trained layers from file."""
        filehandler = open(filename, 'r')
        self.layers = pickle.load(filehandler)


with h5py.File("data/train_catvnoncat.h5", "r") as f:
    X = np.array(f["train_set_x"][:])
    Y = np.array(f["train_set_y"][:])
    classes = np.array(f["list_classes"])
    train_X = X/225
    train_Y = Y.reshape(Y.shape[0], -1).T
with h5py.File("data/test_catvnoncat.h5", "r") as f:
    X_test = np.array(f["test_set_x"])/255
    Y_test = np.array(f["test_set_y"])
    Y_test = Y_test.reshape(Y_test.shape[0], -1).T


def compute_cost(AL, Y, layers, lamb):
    """Compute cost with L2 regularisation."""
    m = Y.shape[0]
    L2 = 0
    for l in layers:
        L2 += lamb/m/2*np.sum(np.square(l.W))
    cost = -1/m * np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL)) + L2
    return np.squeeze(cost)


print("Shape training set X:", train_X.shape)
print("Shape training set Y:", train_Y.shape)

m = train_X.shape[0]
batch_size = 209
in_1 = (batch_size, *train_X.shape[1:])
lamb = 0.01

nn = NeuralNetwork()
# nn.add_layer(ConvLayer(in_1, 3, 8, 1, 1, activation='relu', lamb=lamb))
# nn.add_layer(ConvLayer(nn.layers[0].dim_out,
#                        5, 16, 1, 0, activation='relu', lamb=lamb))
# in_3 = (nn.layers[1].dim_out[1]*nn.layers[1].dim_out[2]
#         * nn.layers[1].dim_out[3], nn.layers[1].dim_out[0])
#nn.add_layer(Layer(in_3, (16, batch_size), activation='relu', lamb=lamb))

nn.add_layer(Layer((64*64*3, batch_size), (128, batch_size),
                   activation='relu', lamb=lamb))
nn.add_layer(Layer(nn.layers[0].dim_out,
                   (64, batch_size), activation='relu', lamb=lamb))
nn.add_layer(Layer(nn.layers[1].dim_out,
                   (32, batch_size), activation='relu', lamb=lamb))
nn.add_layer(Layer(nn.layers[2].dim_out, (8, batch_size),
                   activation='sigmoid', lamb=lamb))

cost = nn.train_network(train_X, train_Y, rate=0.0075, epochs=3000, lamb=0)
nn.save_network('catwork.obj')

plt.plot(np.arange(len(cost)), cost)
plt.show()
