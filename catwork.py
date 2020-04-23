import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle
from layer import Layer
from conv_layer import ConvLayer
from pool_layer import PoolLayer


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

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

    def predict_testset(self, X, Y):
        """Predict the result of input X and compare to labels y.
        Args:
            X (np.array): input testset (m, nx, ny, nc)
            y (np.array): true output
        Returns:
            Accuracy.
        """
        m = Y.shape[1]
        for l in self.layers:
            X = l.forward(X)
        pred = np.float32(X > 0.5)
        a = np.sum(pred == Y)/m
        print("Accuracy on test set: {:.2f}".format(a))
        return pred

    def predict(self, X, classes):
        """Predict the result for a single input and plots the image.
        Args:
            X (np.array): input (nx, ny, nc)
            classes (dict): labels
        Returns:
            None.
        """
        prob = np.array(X).reshape(1, 64, 64, 3)
        for l in self.layers:
            prob = l.forward(prob)
        pred = (prob > 0.5)[0, 0]
        plt.imshow(X)
        plt.title("{:} with prop: {:.3f}".format(classes[pred], prob[0, 0]))
        plt.show()

    def save_network(self, filename):
        """Save the trained layers of the network."""
        filehandler = open(filename, 'wb')
        pickle.dump(self.layers, filehandler)

    def load_network(self, filename):
        """Load trained layers from file."""
        filehandler = open(filename, 'rb')
        self.layers = pickle.load(filehandler)

    def train_network(self, X, Y, rate, epochs, lamb=0.1):
        cost = []
        t = 1
        mini_batches = self.create_mini_batches(X, Y, size=209)
        for i in range(epochs):
            costSum = 0
            for mX, mY in mini_batches:
                for l in self.layers:
                    mX = l.forward(mX)
                print(self.layers[3].W)
                costSum += self.compute_cost(mX, mY, lamb=lamb)
                dA = self.deriv_cost(mX, mY)
                for l in reversed(self.layers):
                    dA = l.backward(dA)
                    l.update_parameters(rate, t)
                t += 1
            cost.append(costSum)
            print("Cost of epoch {:}: {:.5f}".format(i, cost[-1]))
            if i % 100 == 0:
                self.predict_testset(X_test, Y_test)
        return cost


with h5py.File("data/train_catvnoncat.h5", "r") as f:
    X = np.array(f["train_set_x"][:])
    Y = np.array(f["train_set_y"][:])
    classes = list(f["list_classes"])
    train_X = X/225
    train_Y = Y.reshape(Y.shape[0], -1).T
with h5py.File("data/test_catvnoncat.h5", "r") as f:
    X_test = np.array(f["test_set_x"])/255
    Y_test = np.array(f["test_set_y"])
    Y_test = Y_test.reshape(Y_test.shape[0], -1).T

print("Shape training set X:", train_X.shape)
print("Shape training set Y:", train_Y.shape)

m = train_X.shape[0]
batch_size = 209
dim_in = (batch_size, *train_X.shape[1:])
lamb = 0.0000

# l1 = ConvLayer(dim_in, f=3, c=8, stride=1, pad=1, activation='relu', lamb=lamb)
# l2 = ConvLayer(l1.dim_out, f=5, c=16, stride=2,
#                pad=0, activation='relu', lamb=lamb)
# dim_in3 = l2.dim_out[1]*l2.dim_out[2]*l2.dim_out[3]
# l3 = Layer(dim_in3, 100, activation='relu', lamb=lamb)
# l4 = Layer(l3.dim_out, 32, activation='relu', lamb=lamb)
# l5 = Layer(l4.dim_out, 16, activation='relu', lamb=lamb)
# l6 = Layer(l5.dim_out, 1, activation='sigmoid', lamb=lamb)

# nn = NeuralNetwork((l1, l2, l3, l4, l5, l6))

# cost = nn.train_network(train_X, train_Y, rate=0.02, epochs=200, lamb=lamb)
# nn.save_network('catwork2.obj')
# nn.load_network('catwork.obj')

l0 = ConvLayer((209, 64, 64, 3), 3, 4, 1, 1, activation='relu', lamb=lamb)
dim_out = l0.dim_out[1]*l0.dim_out[2]*l0.dim_out[3]
print(dim_out)
l1 = Layer(dim_out, 20, activation='relu', lamb=lamb)
l2 = Layer(l1.dim_out, 7, activation='relu', lamb=lamb)
l3 = Layer(l2.dim_out, 5, activation='relu', lamb=lamb)
l4 = Layer(l3.dim_out, 1, activation='sigmoid', lamb=lamb)

nn = NeuralNetwork((l0, l1, l2, l3, l4))

cost = nn.train_network(train_X, train_Y, rate=0.005, epochs=500, lamb=0.01)

nn.predict_testset(X_test, Y_test)
nn.predict_testset(train_X, train_Y)
# for i in range(X_test.shape[0]):
#     nn.predict(X_test[i], classes)

# plt.plot(np.arange(len(cost)), cost)
# plt.show()
