import numpy as np
import matplotlib.pyplot as plt
import h5py


class NeuralNetwork:
    """L-Layer neural network with He initialisation and L2 regularisation."""

    def __init__(self, dim_layers, activations, lamb=0.0):
        """Initialise the neural network.
        Args:
            dim_layers (list): list with sizes of the layers
            activations (list): list of strings with activation functions
                                'relu' or 'lrelu'
            lamb (float): L2 regularisation parameter 0 == no regularisation.
        """
        self.dim_layers = dim_layers
        self.activations = activations
        self.lamb = lamb
        self.N_layers = len(dim_layers)
        self.parameters = {}
        self.v = {}
        self.s = {}
        self.activ_functions = {"relu": self.relu,
                                "sigmoid": self.sigmoid, "lrelu": self.lrelu}
        self.dActiv_functions = {"relu": self.dRelu,
                                 "sigmoid": self.dSigmoid, "lrelu": self.dLrelu}

    def init_parameters(self):
        """He initialization of the weights and biases
        Returns:
           None.
        """
        param = {}
        for l in range(1, self.N_layers):
            param['W'+str(l)] = np.random.randn(self.dim_layers[l], self.dim_layers[l-1]) \
                * np.sqrt(2/self.dim_layers[l-1])
            param['b'+str(l)] = np.random.randn(self.dim_layers[l],
                                                1) * np.sqrt(2/self.dim_layers[l-1])
            self.v['dW'+str(l)] = np.zeros(param['W'+str(l)].shape)
            self.v['db'+str(l)] = np.zeros(param['b'+str(l)].shape)
            self.s['dW'+str(l)] = np.zeros(param['W'+str(l)].shape)
            self.s['db'+str(l)] = np.zeros(param['b'+str(l)].shape)
        self.parameters = param

    def relu(self, z):
        """ReLu activation function
        Args:
            z (np.array): input
        Returns:
            np.array.
        """
        return np.maximum(0, z)

    def dRelu(self, z):
        """Derivative of ReLu function
        Args:
            z (np.array): input values
        Returns:
            np.array: derivative at z.
        """
        return np.float64(z > 0)

    def lrelu(self, z, a=0.01):
        """Leaky ReLu activation function
        Args:
            z (np.array): input
            a (float): slope on negative for negative z values
        Returns:
            np.array.
        """
        return np.maximum(0, z) + a * np.minimum(0, z)

    def dLrelu(self, z):
        """To implement"""
        print("TODO: implement!")

    def sigmoid(self, z):
        """Sigmoid activation function
        Args:
            z (np.array): input
        Returns:
            np.array.
        """
        return 1/(1+np.exp(-z))

    def dSigmoid(self, z):
        """Derivative of the sigmoid function
        Args:
            z (np.array): input
        Returns:
            np.array.
        """
        return np.exp(-z)/(1+np.exp(-z))**2

    def forward_propagation(self, x):
        """Implementation of the forward propagation
        Args:
            x (np.array): array of input data
        Returns:
            np.array: array of the output layer
            list: list of linear and activation cache as tuples
        """
        A = x
        caches = []
        for l in range(1, self.N_layers):
            W = self.parameters["W"+str(l)]
            b = self.parameters["b"+str(l)]
            Z = np.dot(W, A) + b
            caches.append((Z, A))
            A = self.activ_functions[self.activations[l-1]](Z)
        return A, caches

    def compute_cost(self, AL, Y):
        """Compute the cross entropy cost with L2 regularisation.
        Args:
            AL (np.array): output layer, corresponding to the prediction vector
            Y (np.array): true label vector
        Returns:
            float: cross-entropy cost
        """
        m = Y.shape[1]
        extra = 0
        for l in range(1, self.N_layers):
            extra += 1/m * self.lamb/2 * \
                np.sum(np.square(self.parameters["W"+str(l)]))
        cost = -1/m * np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL)) + extra
        return np.squeeze(cost)

    def backward_propagation(self, AL, Y, caches):
        """Calculate the gradient of the parameters with L2 regularisation.
        Args:
            AL (np.array): array of predicted labels
            Y (np.array): array of true labels
            caches (list): list of (Z_l, A_l) layer activation values
        Returns:
            dict: {'dWl': dWl, 'dbl': dbl,...} dict of gradients.
        """
        grads = {}
        m = AL.shape[1]
        # cost function derivative
        dA = - (Y/AL - (1-Y)/(1-AL))
        for l in range(self.N_layers-1, 0, -1):
            Z, A = caches[l-1]
            m = A.shape[1]
            dZ = dA * self.dActiv_functions[self.activations[l-1]](Z)
            grads["dW"+str(l)] = 1/m * np.dot(dZ, A.T) + \
                self.lamb/m*self.parameters["W"+str(l)]
            grads["db"+str(l)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
            dA = np.dot(self.parameters["W"+str(l)].T, dZ)

        return grads

    def update_parameters(self, grads, learning_rate, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Update the network parameters using the gradient descent.
        Args:
            grads (dict): dictionary of gradients {'dWl': dWl, 'dbl': dbl,...}
            learning_rate (float): learning rate
            t (int): Adam counter
            beta1 (float): first parameter in Adam optimisation
            beta2 (float): second parameter in Adam optimisation
        Returns:
            None.
        """
        for l in range(1, self.N_layers):
            self.v['dW'+str(l)] = beta1*self.v['dW'+str(l)] + \
                (1-beta1)*grads['dW'+str(l)]
            self.v['db'+str(l)] = beta1*self.v['db'+str(l)] + \
                (1-beta1)*grads['db'+str(l)]
            self.s['dW'+str(l)] = beta2*self.s['dW'+str(l)] + \
                (1-beta2)*grads['dW'+str(l)]**2
            self.s['db'+str(l)] = beta2*self.s['db'+str(l)] + \
                (1-beta2)*grads['db'+str(l)]**2
            vW = self.v['dW'+str(l)] / (1-beta1**t)
            vb = self.v['db'+str(l)] / (1-beta1**t)
            sW = self.s['dW'+str(l)] / (1-beta2**t)
            sb = self.s['db'+str(l)] / (1-beta2**t)
            self.parameters["W"+str(l)] -= learning_rate * \
                vW/(np.sqrt(sW)+epsilon)
            self.parameters["b"+str(l)] -= learning_rate * \
                vb/(np.sqrt(sb)+epsilon)

    def train_network(self, X, Y, learning_rate, iterations):
        """Train the network on dataset (X,Y)
        Args:
            X (np.array): shape=(nx, m), nx=#input values, m=#examples
            Y (np.array): shape=(1, m)
        Returns:
            None.
        """
        self.init_parameters()
        cost = []
        t = 1
        for i in range(iterations):
            AL, caches = self.forward_propagation(X)
            cost.append(self.compute_cost(AL, Y))

            grads = self.backward_propagation(AL, Y, caches)
            self.update_parameters(grads, learning_rate, t)
            t += 1

        return cost

    def predict(self, X, y):
        """Predict the result of input X and compare to labels y.
        Args:
            X (np.array): input
            y (np.array): true output
        Returns:
            Accuracy.
        """
        m = X.shape[1]
        prob, cache = self.forward_propagation(X)
        pred = np.float32(prob > 0.5)
        a = np.sum(pred == y)/m
        print("Accuracy on test set: {:.2f}".format(a))
        return pred

    def print_misslabeld(self, X, y, pred, classes):
        a = pred + y
        mislabeled_ind = np.asarray(np.where(a == 1))
        num_images = len(mislabeled_ind[0])
        for i in range(num_images):
            index = mislabeled_ind[1][i]

            plt.subplot(2, num_images, i + 1)
            plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
            plt.axis('off')
            plt.title("Prediction: " + classes[int(pred[0, index])].decode("utf-8") +
                      " \n Class: " + classes[y[0, index]].decode("utf-8"))
        plt.show()


if __name__ == "__main__":
    with h5py.File("data/train_catvnoncat.h5", "r") as f:
        X = np.array(f["train_set_x"][:])
        Y = np.array(f["train_set_y"][:])
        classes = np.array(f["list_classes"])
    train_X = X.reshape(X.shape[0], -1).T/225
    train_Y = Y.reshape(Y.shape[0], -1).T
    with h5py.File("data/test_catvnoncat.h5", "r") as f:
        X_test = np.array(f["test_set_x"])
        Y_test = np.array(f["test_set_y"])
    X_test_flat = X_test.reshape(X_test.shape[0], -1).T/255
    Y_test_flat = Y_test.reshape(Y_test.shape[0], -1).T

    # Network architecture
    layer_dims = (12288, 16, 8, 5, 1)
    activations = ("relu", "relu", "relu", "sigmoid")
    nn = NeuralNetwork(layer_dims, activations, lamb=0.1)
    learning_rate = 0.001
    iterations = 2000
    cost = nn.train_network(train_X, train_Y, learning_rate, iterations)
    pred_train = nn.predict(train_X, train_Y)
    pred_test = nn.predict(X_test_flat, Y_test_flat)
    nn.print_misslabeld(X_test_flat, Y_test_flat, pred_test, classes)
    plt.plot(np.arange(len(cost)), cost)
    plt.show()
