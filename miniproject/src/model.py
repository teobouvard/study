import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras import models
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    ReLU,
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from .callback import WeightsFreezer


class SimpleNet(models.Sequential):
    def __init__(self):
        super().__init__()
        self.initialize_layers()
        # opt = Adam()
        loss_fn = SparseCategoricalCrossentropy()
        self.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
        print(f"Compile model with {self.count_params():,} parameters")
        self.training_records = []

    def fit(self, *args, signature=None, **kwargs):
        if signature is not None:
            if signature.length > self.count_params():
                raise ValueError(
                    f"Signature length is {signature.length} but model has only {self.count_params()} parameters"
                )
            wf = WeightsFreezer(signature)
            kwargs.setdefault("callbacks", []).append(wf)
            print(f"Train model with signature of size {signature.length}")
        new_history = super().fit(*args, **kwargs)
        self.training_records.append(new_history)

    def plot_training(self):
        if not self.training_records:
            raise ValueError("Model has not been trained yet")
        else:
            records = self.merge_records()
            fig, ax = plt.subplots(1, 2)

            ax[0].plot(records["accuracy"], label="Train")
            ax[0].plot(records["val_accuracy"], label="Test")
            ax[0].set(xlabel="Epoch", ylabel="Accuracy", title="Model accuracy")
            ax[0].legend(loc="upper left")

            ax[1].plot(records["loss"], label="Train")
            ax[1].plot(records["val_loss"], label="Test")
            ax[1].set(xlabel="Epoch", ylabel="Loss", title="Model loss")
            ax[1].legend(loc="upper right")

            return fig, ax

    def merge_records(self):
        records = {}
        for record in self.training_records.copy():
            for metric, values in record.history.items():
                records.setdefault(metric, []).extend(values)
        return records

    def save_training_plot(self, fname):
        fig, ax = self.plot_training()
        fig.savefig(fname)

    def save_training_history(self, fname):
        pd.DataFrame(self.history.history).to_csv(fname)

    def initialize_layers(self):
        kernel_size = 3
        pool_size = 2

        for i in range(4):
            drop_rate = (i+2) / 10 # 0.2, 0.3, 0.4
            n_filters = (2**i) * 32 # 32, 64, 128
            self.add(Conv2D(n_filters, kernel_size, padding="same", input_shape=(32, 32, 3)))
            self.add(BatchNormalization())
            self.add(ReLU())
            self.add(Conv2D(n_filters, kernel_size, padding="same", input_shape=(32, 32, 3)))
            self.add(BatchNormalization())
            self.add(ReLU())
            self.add(MaxPooling2D(pool_size))
            self.add(Dropout(drop_rate))

        # Classifier block
        self.add(Flatten())
        self.add(Dense(128, activation="relu"))
        self.add(BatchNormalization())
        self.add(Dropout(0.5))
        self.add(Dense(10, activation="softmax"))

