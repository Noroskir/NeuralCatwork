import numpy as np


class Layer:
    def __init__(self, dim_in, dim, activation='relu', lamb=0.0):
        """Initialise"""
        self.dim_in = dim_in
        self.dim = dim
        self.activation = activation
        self.lamb = lamb
