from hashlib import sha256
from itertools import accumulate

import numpy as np


class Signature:
    """
    An array of randomly generated weights and their respective indices.
    The indices corresponds to the model's flattened array of weights.
    """

    def __init__(self, data, length, hash_fn=sha256):
        """
        Note :
        If indices and weights are taken from the same prng, the indices are not reproducible
        when the signature size changes. The workaround is to initialize two prng with the 
        same seed, so both idx and weights are "deterministic" no matter the signature size.
        More detail :

        >>> gen = np.random.RandomState(42)
        >>> x1 = gen
        >>> x2 = gen
        >>> print(x1.randn(4))
        >>> print(x2.randn(4))
        [ 0.49671415 -0.1382643   0.64768854  1.52302986]
        [-0.23415337 -0.23413696  1.57921282  0.76743473]

        VS

        >>> x1 = np.random.RandomState(42)
        >>> x2 = np.random.RandomState(42)
        >>> print(x1.uniform(-1, 1, 5))
        >>> print(x2.uniform(-1, 1, 5))
        [ 0.49671415 -0.1382643   0.64768854  1.52302986]
        [ 0.49671415 -0.1382643   0.64768854  1.52302986]

        This behaviour is desirable for testing purposes, but maybe not for real life application.
        """ 
        self._idx_gen = self._gen_state(data, hash_fn)
        self._weights_gen = self._gen_state(data, hash_fn)
        self.length = length
        self.weights = self._gen_weights()
        self.masks = []
        self.frozen = []

    def _gen_state(self, data, hash_fn):
        """
        Generates a random state that can be used as reproducible number generator from a bytes object
        Note : 
        Numpy's RandomState needs a integer seed between 0 and 2^32-1.
        This is undesirable but can be fixed using a "real" PRNG, or chaining them.
        """
        h = hash_fn(data).hexdigest()
        seed = int(h, 16) % pow(2, 32)
        return np.random.RandomState(seed)

    def _gen_weights(self, distribution="uniform"):
        """
        Generates n random weights from a uniform distribution
        """
        if distribution == "uniform":
            return self._weights_gen.uniform(-1, 1, self.length).astype("float32")
        else:
            raise NotImplementedError()
    
    def _gen_indices(self, n, distribution="uniform"):
        """
        Generates random indices in the interval [0, n-1] from a given distribution
        """
        arr = np.arange(n)
        if distribution == "uniform":
            return self._idx_gen.choice(arr, self.length, replace=False)
        else:
            raise NotImplementedError()

    def verify(self, model, tol=1e-7):
        """
        Verifies the signature of the model (within floating point accuracy)
        """
        if not self.frozen:
            self._fit(model)
        if self.length == 0:
            return True
        
        for i, tensor in enumerate(model.trainable_weights):
            if self.masks[i].any():
                if np.max((tensor - self.frozen[i])[self.masks[i]]) > tol:
                    return False

        return True

    def _fit(self, model):
        """
        Fits the signature to the model structure
        """
        layer_params = []
        layer_shapes = []

        for tensor in model.trainable_weights:
            layer_shapes.append(tensor.shape)
            params_at_layer = np.multiply.reduce(tensor.shape)
            layer_params.append(params_at_layer)

        acc = np.add.accumulate(layer_params)
        rand_idx = self._gen_indices(acc[-1])

        mask = np.zeros(acc[-1], dtype=bool)
        mask[rand_idx] = True

        frozen = np.zeros(acc[-1], dtype='float32')
        frozen[mask] = self.weights

        self.masks = np.hsplit(mask, acc[:-1])
        self.frozen = np.hsplit(frozen, acc[:-1])

        for i, shape in enumerate(layer_shapes):
            self.masks[i] = self.masks[i].reshape(shape)
            self.frozen[i] = self.frozen[i].reshape(shape)

    def sign(self, model):
        """
        Replaces the model weights by the signature weights.
        The signature tensors are computed lazily so the signature can be
        instantiated independently of the model.
        """
        if not self.frozen:
            self._fit(model)

        for i, tensor in enumerate(model.trainable_weights):
            tensor.assign((1-self.masks[i])*tensor + self.masks[i]*self.frozen[i])
