from tensorflow.keras.callbacks import Callback


class WeightsFreezer(Callback):
    """
	Callback used to freeze the signature weights of a model during training.
	"""

    def __init__(self, signature):
        super().__init__()
        self.sig = signature

    def on_batch_end(self, batch, logs):
        self.sig.sign(self.model)
