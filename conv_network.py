import numpy as np
import h5py
import matplotlib.pyplot as plt


class ConvNeuralNetwork:
    """Convolutional neural network with He initialisation and L2 regularisation.
    TODO: - batch normalisation
          - softmax implementation
    """

    def __init__(self, input_shape, filters, dim_layers, activations, lamb=0.1):
        """Initialise the neural network.
        Args:
            input_shape (tuple): m, x, y, c
            filters (list): list with (f, channels, stride, padding)
            dim_layers (list): list with sizes of the layers
            activations (list): list of strings with activation functions
                                'relu' or 'lrelu'
            lamb (float): L2 regularisation parameter 0 == no regularisation.
        """
        self.input_shape
        self.filt_params = filters
        self.dim_layers = dim_layers
        self.activations = activations
        self.lamb = lamb
        self.N_layers = len(dim_layers)
        self.parameters = {}
        self.filters = {}
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
        filt = {}
        f_prev = self.input_shape[1]
        c_prev = self.input_shape[-1]
        for l in range(1, len(self.filt_parms)):
            f = self.filt_params[0]
            c = self.filt_params[1]
            filt['F'+str(l)] = np.random.randn(f, f, c_prev, c) * \
                np.sqrt(2/(f_prev**2*c_prev))
            filt['b'+str(l)] = np.zeros((1, 1, 1, c))
            self.v['dF'+str(l)] = np.zeros(filt['F'+str(l)].shape)
            self.v['dfb'+str(l)] = np.zeros(filt['b'+str(l)].shape)
            self.s['dF'+str(l)] = np.zeros(filt['F'+str(l)].shape)
            self.s['dfb'+str(l)] = np.zeros(filt['b'+str(l)].shape)
            f_prev = f
            c_prev = c
        self.filters = filt

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

    def conv_forward(self, im, F, b, param):
        """Cross correlate image with filters.
        Args:
            im (np.array): array of images with 3 color channels shape: (m, x, y, c)
            F (np.array): filter, shape (f, f, n_c_p, n_c)
            b (np.array): biases, shape (1, 1 ,1, n_c)
            param (dict): dictionary of {'stride': ..., 'pad': ...}
        Returns
            np.array: output of the convolution layer
            tuple: tuple of (im, F, b, param)
        """
        m, n_h_p, n_w_p, n_c_p = im.shape
        pad = param['pad']
        stride = param['stride']
        f = F.shape[0]
        n_c = F.shape[-1]
        n_h = int((n_h_p - f + 2*pad)/stride) + 1
        n_w = int((n_w_p - f + 2*pad)/stride) + 1
        Z = np.zeros((m, n_h, n_w, n_c))
        im_pad = np.pad(im, ((0, 0), (pad, pad), (pad, pad), (0, 0)),
                        mode='constant', constant_values=(0, 0))
        for i in range(m):
            for h in range(n_h):
                v_s = h*stride
                v_e = v_s + f
                for w in range(n_w):
                    h_s = w*stride
                    h_e = h_s + f
                    for c in range(n_c):
                        Z[i, h, w, c] = np.sum(
                            im_pad[i, v_s:v_e, h_s:h_e]*F[:, :, :, c]) + b[:, :, :, c]

        cache = (im, F, b, param)
        return Z, cache

    def conv_backward(self, dZ, cache):
        """Implementation of backward propagation of a convolution
        Args:
            dZ"""
        (A_p, F, b, param) = cache
        (m, n_h_p, n_w_p, n_c_p) = A_p.shape
        (f, f, n_C_prev, n_C) = F.shape

        stride = param["stride"]
        pad = param["stride"]

        (m, n_h, n_w, n_c) = dZ.shape

        dA_p = np.zeros(A_p.shape)
        dF = np.zeros(F.shape)
        db = np.zeros(b.shape)

        A_p_pad = np.pad(A_p, ((0, 0), (pad, pad), (pad, pad),
                               (0, 0)), mode='constant', constant_values=(0, 0))
        dA_p_pad = np.pad(dA_p, ((0, 0), (pad, pad), (pad, pad),
                                 (0, 0)), mode='constant', constant_values=(0, 0))
        for i in range(m):
            for h in range(n_h):
                v_s = h*stride
                v_e = h*stride + f
                for w in range(n_w):
                    h_s = w*stride
                    h_e = w*stride + f
                    for c in range(n_c):
                        dA_p_pad[i, v_s:v_e, h_s:h_e,
                                 :] += F[:, :, :, c] \
                            * dZ[i, h, w, c]
                        dF[:, :, :, c] += A_p_pad[i, v_s:v_e, h_s:h_e] \
                            * dZ[i, h, w, c]
                        db[:, :, :, c] += dZ[i, h, w, c]
            dA_p[i, :, :, :] = dA_p_pad[i, pad:-pad, pad:-pad, :]

        return dA_p, dF, db

    def pool_forward(self, im, param, mode='max'):
        """Pooling layer with max or average pooling.
        Args:
            im (np.array): array of images, shape (m, x, y, c)
            param (dict): dictionary of {'f': ..., 'stride': ...}
        Returns:
            np.array: output of the pooling layer
            cache
        """
        m, n_h_p, n_w_p, n_c = im.shape
        f = param['f']
        stride = param['stride']

        n_h = int(1 + (n_h_p - f)/stride)
        n_w = int(1 + (n_w_p - f)/stride)

        Z = np.zeros((m, n_h, n_w, n_c))
        for i in range(m):
            for h in range(n_h):
                v_s = h*stride
                v_e = v_s + f
                for w in range(n_w):
                    h_s = w*stride
                    h_e = h_s + f
                    for c in range(n_c):
                        if mode == "max":
                            Z[i, h, w, c] = np.max(im[i, v_s:v_e, h_s:h_e, c])
                        elif mode == "average":
                            Z[i, h, w, c] = np.mean(im[i, v_s:v_e, h_s:h_e, c])

        cache = (im, param)
        return Z, cache

    def pool_backward(self, dA, cache, mode='max'):
        """Backwards pool"""
        (A_p, param) = cache
        stride = param["stride"]
        f = param['f']
        m, n_h_p, n_w_p, n_c_p = A_p.shape
        m, n_h, n_w, n_c = dA.shape

        dA_p = np.zeros(A_p.shape)

        for i in range(m):
            for h in range(n_h):
                v_s = h*stride
                v_e = h*stride+f
                for w in range(n_w):
                    h_s = w*stride
                    h_e = w*stride+f
                    for c in range(n_c):
                        if mode == "max":
                            mask = np.max(
                                A_p[i, v_s:v_e, h_s:h_e, c]) == A_p[i, v_s:v_e, h_s:h_e, c]
                            dA_p[i, v_s:v_e, h_s:h_e, c] += mask * \
                                dA[i, h, w, c]
                        elif mode == "average":
                            da = dA[i, h, w, c]
                            dA_p[i, v_s:v_e, h_s: h_e,
                                 c] += np.ones((f, f))*da/f**2
        return dA_p

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

    def update_filters(self, grads, learning_rate, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Update convolution filter parameters with Adam optimisation.
        Args:
            grads (dict): dictionary of gradients {'dF1': dF1, 'db1': db1, ...}
            learning_rate (float): learning_rate
            t (int): adam counter
            beta1 (float): first parameter in Adam optimisation
            beta2 (float): second parameter in Adam optimisation
        Returns:
            None.
        """
        for l in range(1, len(self.filt_params)):
            self.v['dF'+str(l)] = beta1*self.v['dF'+str(l)] + \
                (1-beta1)*grads['dF'+str(l)]
            self.v['dfb'+str(l)] = beta1*self.v['dfb'+str(l)] + \
                (1-beta1)*grads['db'+str(l)]
            self.s['dF'+str(l)] = beta2*self.s['dF'+str(l)] + \
                (1-beta2)*grads['dF'+str(l)]**2
            self.s['dfb'+str(l)] = beta2*self.s['dfb'+str(l)] + \
                (1-beta2)*grads['db'+str(l)]**2
            vF = self.v['dF'+str(l)] / (1-beta1**t)
            vb = self.v['dfb'+str(l)] / (1-beta1**t)
            sF = self.s['dF'+str(l)] / (1-beta2**t)
            sb = self.s['dfb'+str(l)] / (1-beta2**t)
            self.filters["F"+str(l)] -= learning_rate * \
                vF/(np.sqrt(sF)+epsilon)
            self.filters["b"+str(l)] -= learning_rate * \
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
