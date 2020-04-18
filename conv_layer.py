import numpy as np


class ConvLayer:
    def __init__(self, in_shape, f, c, stride, pad, activation='relu'):
        """Initialise the convolutional layer of the neural network.
        """
        self.in_shape = in_shape
        self.out_shape = (in_shape[0],
                          1+int((in_shape[1]-f + 2*pad)/stride),
                          1+int((in_shape[2]-f + 2*pad)/stride),
                          c)
        self.f = f
        self.n_c = c
        self.stride = stride
        self.pad = pad
        self.activation = activation
        self.X = np.zeros(in_shape)
        self.Z = np.zeros(self.out_shape)
        self.W = self.__init_weights(f, c, in_shape)
        self.b = np.zeros((1, 1, 1, c))
        self.dX = np.zeros(self.X.shape)
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)
        self.vW = np.zeros(self.W.shape)
        self.vb = np.zeros(self.b.shape)
        self.sW = np.zeros(self.W.shape)
        self.sb = np.zeros(self.b.shape)

    def __init_weights(self, f, c, in_shape):
        """Initialise parameters He initialisation."""
        return np.random.randn(f, f, in_shape[-1], c) \
            * np.sqrt(2/(in_shape[0]*in_shape[1]*in_shape[-1]))

    def _relu(self, z):
        """ReLu activation function
        Args:
            z (np.array): input
        Returns:
            np.array.
        """
        return np.maximum(0, z)

    def _deriv_relu(self, z):
        """Derivative of ReLu function
        Args:
            z (np.array): input values
        Returns:
            np.array: derivative at z.
        """
        return np.float64(z > 0)

    def forward(self, x):
        """Forward propagation
        Args:
            x (np.array): array of dimension in_shape (m, n_h_p, n_w_p, n_c_p)
        Returns:
            np.array: output.
        """
        def cross_correlate(a, f):
            s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
            strd = np.lib.stride_tricks.as_strided
            subM = strd(a, shape=s, strides=a.strides * 2)
            return np.einsum('ij,ijkl->kl', f, subM)

        self.X = x.copy()
        m, n_h, n_w, n_c = self.out_shape
        if self.pad != 0:
            x_pad = np.pad(x, ((0, 0), (self.pad, self.pad),
                               (self.pad, self.pad), (0, 0)),
                           mode='constant', constant_values=(0, 0))
        else:
            x_pad = x
        myZ = np.zeros(self.Z.shape)
        myZ = cross_correlate(x_pad[1, :, :], self.W[:, :, :, 0]) + self.b
        print(myZ.shape)
        print(self.Z.shape)
        for h in range(n_h):
            v_s = h*self.stride
            v_e = v_s + self.f
            for w in range(n_w):
                h_s = w*self.stride
                h_e = h_s + self.f
                for c in range(self.n_c):
                    self.Z[:, h, w, c] = np.sum(
                        x_pad[:, v_s:v_e, h_s:h_e] *
                        self.W[:, :, :, c],
                        axis=(1, 2, 3))+self.b[:, :, :, c]

        return self._relu(self.Z)

    def backward(self, dA):
        """Backward propagation
        Args:
            dA (np.array): gradient of output values
        Returns:
            np.array: dX gradient of input values
        """
        (m, n_h, n_w, n_c) = dA.shape
        x_pad = np.pad(self.X, ((0, 0), (self.pad, self.pad), (self.pad, self.pad),
                                (0, 0)), mode='constant', constant_values=(0, 0))
        dx_pad = np.pad(self.dX, ((0, 0), (self.pad, self.pad), (self.pad, self.pad),
                                  (0, 0)), mode='constant', constant_values=(0, 0))
        dZ = dA * self._deriv_relu(self.Z)
        for h in range(n_h):
            v_s = h*self.stride
            v_e = h*self.stride + self.f
            for w in range(n_w):
                h_s = w*self.stride
                h_e = w*self.stride + self.f
                for c in range(n_c):
                    dx_pad[:, v_s:v_e, h_s:h_e, :] += self.W[:, :, :, c] \
                        * dZ[:, h, w, c].reshape(dZ.shape[0], 1, 1, 1)
                    self.dW[:, :, :, c] += np.sum(x_pad[:, v_s:v_e, h_s:h_e]
                                                  * dZ[:, h, w, c].reshape(dZ.shape[0], 1, 1, 1),
                                                  axis=0)
                    self.db[:, :, :, c] += np.sum(dZ[:, h, w, c])
        self.dX[:, :, :, :] = dx_pad[:, self.pad:-
                                     self.pad, self.pad:-self.pad, :]

        return self.dX

    def update_parameters(self, rate, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Update parameters"""
        self.vW = beta1*self.vW + (1-beta1)*self.dW
        self.vb = beta1*self.vb + (1-beta1)*self.db
        self.sW = beta2*self.sW + (1-beta2)*self.dW**2
        self.sb = beta2*self.sb + (1-beta2)*self.db**2
        vW = self.vW / (1-beta1**t)
        vb = self.vb / (1-beta1**t)
        sW = self.sW / (1-beta2**t)
        sb = self.sb / (1-beta2**t)
        self.W -= rate * vW/(np.sqrt(sW)+epsilon)
        self.b -= rate * vb/(np.sqrt(sb)+epsilon)
