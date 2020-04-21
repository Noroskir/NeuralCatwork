import numpy as np
from opt_einsum import contract


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
            * np.sqrt(2/(in_shape[1]*in_shape[2]))

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

    def forward(self, X):
        """Forward propagation
        Args:
            x (np.array): array of dimension in_shape (m, n_h_p, n_w_p, n_c_p)
        Returns:
            np.array: output.
        """
        self.X = X.copy()
        n_h = int((X.shape[1] - self.f + 2*self.pad) / self.stride) + 1
        n_w = int((X.shape[2] - self.f + 2*self.pad) / self.stride) + 1

        if self.pad != 0:
            x_pad = np.pad(X, ((0, 0), (self.pad, self.pad),
                               (self.pad, self.pad), (0, 0)),
                           mode='constant', constant_values=(0, 0))
        else:
            x_pad = X

        # compute Z for multiple input images and multiple filters
        shape = (self.f, self.f, self.in_shape[-1], X.shape[0], n_h, n_w, 1)
        strides = (x_pad.strides * 2)[1:]
        strides = (*strides[:-3], strides[-3]*self.stride,
                   strides[-2]*self.stride, strides[-1])
        M = np.lib.stride_tricks.as_strided(
            x_pad, shape=shape, strides=strides, writeable=False)
        self.Z = contract('pqrs,pqrtbmn->tbms', self.W, M)
        self.Z = self.Z + self.b
        if self.activation == 'relu':
            return self._relu(self.Z)
        elif self.activation == 'none':
            return self.Z

    def conv_backward(self, dA):
        """Naive backward propagation implementation
        Args:
            dA (np.array): gradient of output values
        Returns:
            np.array: dX gradient of input values
        """
        self.dW[:, :, :, :] = 0
        self.db[:, :, :, :] = 0
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

    def backward(self, dA):
        """Numpy einsum and stride tricks backward propagation implementation.
        Args:
            dA (np.array): gradient of output values
        Returns:
            np.array: dX gradient of input values

            """
        dZ = dA * self._deriv_relu(self.Z)
        self.dW[:, :, :, :] = 0
        self.db[:, :, :, :] = 0
        (m, n_H_prev, n_W_prev, n_C_prev) = self.X.shape
        (f, f, n_C_prev, n_C) = self.W.shape
        stride = self.stride
        pad = self.pad
        (m, n_H, n_W, n_C) = dZ.shape
        pad_dZ = f-(pad+1)
        in_h = self.X.shape[1] + (f-1)
        in_w = self.X.shape[2] + (f-1)
        W_rot = np.rot90(self.W, 2)

        dZ_pad = np.zeros((dZ.shape[0], in_h, in_w, dZ.shape[-1]))
        if pad_dZ == 0:
            dZ_pad[:, 0::stride, 0::stride] = dZ
        else:
            dZ_pad[:, pad_dZ:-pad_dZ:stride, pad_dZ:-pad_dZ:stride, :] = dZ

        input = dZ_pad[:, :, :, :]
        kernel = W_rot[:, :, :, :]
        shape = (input.shape[0],                        # m
                 input.shape[1] - kernel.shape[0] + 1,  # X_nx
                 input.shape[2] - kernel.shape[1] + 1,  # X_ny
                 input.shape[3],                        # dZ_nc
                 kernel.shape[0],                       # f
                 kernel.shape[1])                       # f
        strides = (input.strides[0],
                   input.strides[1],
                   input.strides[2],
                   input.strides[3],
                   input.strides[1],
                   input.strides[2])
        M = np.lib.stride_tricks.as_strided(
            input, shape=shape, strides=strides, writeable=False,)
        self.dX = contract('pqrs,bmnspq->bmnr', W_rot, M)

        del dZ_pad

        X_pad = np.pad(self.X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant',
                       constant_values=(0, 0))
        shape_Z = (f, f, n_C_prev, m, n_H, n_W)
        strides_Z = (X_pad.strides)[1:] + (X_pad.strides)[0:3]
        strides_Z = (*strides_Z[:-2], strides_Z[-2]
                     * stride, strides_Z[-1]*stride)

        M = np.lib.stride_tricks.as_strided(
            X_pad, shape=shape_Z, strides=strides_Z, writeable=False)
        self.dW = contract('abcd,pqsabc->pqsd', dZ, M)
        assert(self.dW.shape == self.W.shape)

        self.db = contract('abcd->d', dZ).reshape(1, 1, 1, n_C)

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
