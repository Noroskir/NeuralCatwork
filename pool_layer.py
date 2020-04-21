import numpy as np


class PoolLayer:
    def __init__(self, in_shape, f,  stride, mode='max'):
        """Initialise the pooling layer.
        Args:
            in_shape (tuple): (m, n_x, n_y, n_c)
            f (int): filter size
            stride (int): stride
            pad (int): padding
            mode (str): max or average pooling
        """
        self.in_shape = in_shape
        self.f = f
        self.stride = stride
        self.mode = mode
        self.out_shape = (in_shape[0],
                          1 + int((in_shape[1] - f)/stride),
                          1 + int((in_shape[2] - f)/stride),
                          in_shape[-1])
        self.X = np.zeros(in_shape)
        self.dX = np.zeros(in_shape)
        self.Z = np.zeros(self.out_shape)

    def forward(self, x):
        """Forward implementation of the pooling
        Args:
            x (np.array): input values (m, n_x, n_y, n_c)
        Returns:
            np.array: output values
        """
        m, n_h, n_w, n_c = self.out_shape
        self.X = x.copy()
        for i in range(m):
            for h in range(n_h):
                v_s = h*self.stride
                v_e = v_s + self.f
                for w in range(n_w):
                    h_s = w*self.stride
                    h_e = h_s + self.f
                    for c in range(n_c):
                        if self.mode == "max":
                            self.Z[i, h, w, c] = np.max(
                                x[i, v_s:v_e, h_s:h_e, c])
                        elif self.mode == "average":
                            self.Z[i, h, w, c] = np.mean(
                                x[i, v_s:v_e, h_s:h_e, c])
        return self.Z

    def backward(self, dA):
        """Implementation of backward pooling.
        Args:
            dA (np.array): derivative of output values
        Returns:
            np.array: derivative of intput values
        """
        self.dX[:, :, :, :] = 0
        m, n_h, n_w, n_c = self.out_shape
        for i in range(m):
            for h in range(n_h):
                v_s = h*self.stride
                v_e = h*self.stride+self.f
                for w in range(n_w):
                    h_s = w*self.stride
                    h_e = w*self.stride+self.f
                    for c in range(n_c):
                        if self.mode == "max":
                            mask = np.max(
                                self.X[i, v_s:v_e, h_s:h_e, c]) == self.X[i, v_s:v_e, h_s:h_e, c]
                            self.dX[i, v_s:v_e, h_s:h_e, c] += mask * \
                                dA[i, h, w, c]
                        elif self.mode == "average":
                            da = dA[i, h, w, c]
                            self.dX[i, v_s:v_e, h_s: h_e,
                                    c] += np.ones((self.f, self.f))*da/self.f**2
        return self.dX
