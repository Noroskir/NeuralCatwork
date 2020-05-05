import gzip
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

from fc_layer import FCLayer
from conv_layer import ConvLayer
from pool_layer import PoolLayer


class Network:

    def __init__(self, layers, lamb):
        self.layers = layers
        self.lamb = lamb
        self._set_lamb()

    def _set_lamb(self):
        """Set the L2 regularisation parameter in the layers."""
        for l in self.layers:
            l.lamb = self.lamb

    def create_mini_batches(self, X, Y, size):
        """Create mini batches from the input data."""
        m = X.shape[0]
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation]
        shuffled_Y = Y[:, permutation]

        mini_batches = []

        N_full_batches = int(m/size)
        for i in range(N_full_batches):
            mini_X = cp.array(shuffled_X[i*size:(i+1)*size])
            mini_Y = cp.array(shuffled_Y[:, i*size:(i+1)*size])
            mini_batches.append((mini_X, mini_Y))

        if m % size != 0:
            mini_X = cp.array(shuffled_X[N_full_batches*size:])
            mini_Y = cp.array(shuffled_Y[:, N_full_batches*size:])
            mini_batches.append((mini_X, mini_Y))

        return mini_batches

    def softmax(self, X):
        """Forward propagation of the softmax layer.
        Args:
            X (np.array): input values
        Returns:
            np.array: softmax applied values
        """
        A = cp.exp(X-cp.max(X))/cp.sum(cp.exp(X-cp.max(X)), axis=0)
        return A

    def compute_cost(self, A, Y):
        """Cross entropy cost for the softmax layer"""
        m = Y.shape[-1]
        L2 = 0
        for l in self.layers:
            L2 += self.lamb/m/2*cp.sum(cp.square(l.W))
        cost = -1/m*cp.sum(Y*cp.log(A)) + L2
        return cost

    def compute_dA(self, A, Y):
        """Compute dA for the softmax layer.
        Args:
            A (np.array): predicted labels
            Y (np.array): true labels
        Returns:
            np.array: dA
        """
        dA = (A-Y)
        return dA

    def forward_prop(self, X):
        """Forward propagation"""
        for l in self.layers:
            X = l.forward(X)
        A = self.softmax(X)
        return A

    def backward_prop(self, A, Y, rate, t):
        """Backward propagation and update of parameters"""
        dA = self.compute_dA(A, Y)
        for l in reversed(self.layers):
            dA = l.backward(dA)
            l.update_parameters(rate, t)

    def predict(self, X):
        """Predict single image.
        Args:
            X (np.array): (nx, ny, c)"""
        X_F = cp.array([X])
        for l in self.layers:
            X_F = l.forward(X_F)
        A = self.softmax(X_F)
        print(A)
        res = cp.where(A == cp.max(A))
        plt.imshow(X[:, :, 0])
        plt.title("Is a {:} with prob {:.2f}%".format(
            int(res[0][0]), float(A[res][0]*100)))
        plt.show()

    def test_accuracy(self, X, Y):
        """Test the accuracy of the network on test sample.
        Args:
            X (np.array): (m, nx, ny, c) input sample
            Y (np.array): (1, 10, m) true labels
        Returns:
            None.
        """
        X = cp.array(X)
        Y = cp.array(Y)
        for l in self.layers:
            X = l.forward(X)
        A = self.softmax(X)
        pred = (A == cp.max(A, axis=0)[None, :]).astype(cp.float32)
        acc = (cp.all(pred == Y, axis=0)).astype(cp.float32)
        acc = cp.sum(acc)/len(acc)
        print("Accuracy: {:.2f}%".format(float(acc*100)))
        return acc

    def train_network(self, X, Y, epochs, rate, mb_size):
        """"""
        cost = []
        acc = []
        t = 1
        mini_batches = self.create_mini_batches(X, Y, mb_size)
        for i in range(epochs):
            cost_sum = 0
            for mX, mY in mini_batches:
                A = self.forward_prop(mX)
                c = self.compute_cost(A, mY)
                cost_sum += c
                self.backward_prop(A, mY, rate, t)
                t += 1

            cost.append(cost_sum/len(mini_batches))
            if i % 1 == 0:
                print("Cost of epoch {:}: {:.5f}".format(i, float(cost[-1])))
                acc.append(self.test_accuracy(X_test, Y_test))
                print()
        return cost, acc


def load_data(num_images=5, mode="train", path="data/"):
    ft = gzip.open(path+mode+'-images-idx3-ubyte.gz', 'r')
    image_size = 28
    ft.read(16)
    buf = ft.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size, 1)

    fl = gzip.open(path+mode+'-labels-idx1-ubyte.gz', 'r')
    fl.read(8)
    buf = fl.read(num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    Y = labels.reshape(labels.shape[0], -1).T
    Y_vect = np.zeros((Y.shape[0], 10, Y.shape[1]))
    for i in range(len(Y[0])):
        for j in range(len(Y_vect[0])):
            if Y[0, i] == j:
                Y_vect[0, j, i] = 1
    Y = Y_vect[0]

    return data/255, Y


X, Y = load_data(num_images=30000, mode="train")
X_test, Y_test = load_data(num_images=1000, mode="t10k")
X_test = cp.array(X_test)
Y_test = cp.array(Y_test)

lamb = 0.001

mb_size = 1000

in_shape = (mb_size, *X.shape[1:])

l1 = ConvLayer(in_shape, 3, 10, 1, 0, activation='relu')
l2 = PoolLayer(l1.dim_out, 2, 2, mode=max)
l3 = FCLayer(l2.dim_out, 100, activation='relu')
l4 = FCLayer(l3.dim_out, 10, activation='sigmoid')


net = Network((l1, l2, l3, l4), lamb=lamb)

cost = net.train_network(X, Y, 30, 0.0007, mb_size)
