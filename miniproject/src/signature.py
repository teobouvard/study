from hashlib import sha256
from itertools import accumulate

import numpy as np


class Signature:
    """
    An array of randomly generated weights and their respective indices.
    The indices corresponds to the model's flattened array of weights.
    """

    def __init__(self, data, length, hash_fn=sha256):
        self.rand_gen = self.gen_state(data, hash_fn)
        self.length = length
        self.weights = self.gen_weights()
        self.idx = None
        print(f"Create signature with {self.length} parameters")

    def gen_state(self, data, hash_fn):
        """
        Generates a random state that can be used as reproducible number generator from a bytes object
        Note : numpy's RandomState needs a integer seed between 0 and 2^32-1
        This is undesirable but can be fixed using a "real" PRNG
        """
        h = hash_fn(data).hexdigest()
        seed = int(h, 16) % pow(2, 32)
        return np.random.RandomState(seed)

    def gen_weights(self, distribution="uniform"):
        """
        Generates n random weights from a uniform distribution fixed with seed
        """
        if distribution == "uniform":
            return self.rand_gen.uniform(-1, 1, self.length).astype("float32")
        else:
            raise NotImplementedError()

    def gen_indices(self, model, distribution="uniform"):
        """
        Generates n random indices in the interval [0, self.model_size-1] from a uniform distribution fixed with seed
        """
        idx = np.arange(model.count_params())
        if distribution == "uniform":
            return self.rand_gen.choice(idx, self.length, replace=False)
        else:
            raise NotImplementedError()

    def verify(self, model, tol=1e-8):
        """
        Verifies the signature of the model
        """
        if self.idx is None:
            self.idx = self.gen_indices(model)

        flat_weights = [layer.flatten() for layer in model.get_weights()]
        flat_weights = np.concatenate(flat_weights)
        return np.abs(flat_weights[self.idx] - self.weights).max() < tol

    def apply(self, model):
        """
        Replaces the model weights by the signature weights.
        Note : the indices have to be computed the first time the signature is applied to the model, 
        because we don't know in advance the number of parameters which is the range of possible indices
        """
        if self.idx is None:
            self.idx = self.gen_indices(model)

        model_weights = model.get_weights()
        shapes = [layer.shape for layer in model_weights]
        flat_weights = [layer.flatten() for layer in model_weights]
        # keeping the last cut index creates a last empty split, so we do not consider it
        cuts = list(accumulate(len(layer) for layer in flat_weights[:-1]))
        flat_weights = np.concatenate(flat_weights)
        flat_weights[self.idx] = self.weights
        rebuilt_weights = np.hsplit(flat_weights, cuts)
        rebuilt_weights = [
            layer.reshape(shapes[i]) for i, layer in enumerate(rebuilt_weights)
        ]
        model.set_weights(rebuilt_weights)
