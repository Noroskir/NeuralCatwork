import numpy as np
import matplotlib.pyplot as plt
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
                costSum += self.compute_cost(mX, mY, lamb=lamb)
                dA = self.deriv_cost(mX, mY)
                for l in reversed(self.layers):
                    dA = l.backward(dA)
                    l.update_parameters(rate, t)
                t += 1
            cost.append(costSum)
            print("Cost of epoch {:}: {:.5f}".format(i, cost[-1]))
        return cost
