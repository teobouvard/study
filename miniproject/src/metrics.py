import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score

tf.config.experimental_run_functions_eagerly(True)

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, average, name="f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.average = average
        self.batch_scores = []
        self.f1 = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.reshape(y_true, (-1))
        self.batch_scores.append(f1_score(y_true, y_pred, average=self.average))

    def result(self):
        return np.mean(self.batch_scores)

    def reset_states(self):
        self.batch_scores = []