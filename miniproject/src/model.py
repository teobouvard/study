# https://arxiv.org/pdf/1608.06037.pdf

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

    def old_initialize_layers(self):
        kernel_size = 3
        drop_rate = 0.2
        pool_size = 2

        # Block 1
        self.add(Conv2D(64, kernel_size, padding="same", input_shape=(32, 32, 3)))
        self.add(BatchNormalization())
        self.add(ReLU())
        self.add(Dropout(drop_rate))

        # Block 2
        for _ in range(3):
            self.add(Conv2D(128, kernel_size, padding="same"))
            self.add(BatchNormalization())
            self.add(ReLU())
            self.add(Dropout(drop_rate))
        self.add(MaxPooling2D())

        # Block 3
        for _ in range(2):
            self.add(Conv2D(128, kernel_size, padding="same"))
            self.add(BatchNormalization())
            self.add(ReLU())
            self.add(Dropout(drop_rate))

        # Block 4
        self.add(Conv2D(128, kernel_size, padding="same"))
        self.add(BatchNormalization())
        self.add(ReLU())
        self.add(Dropout(drop_rate))
        self.add(MaxPooling2D())

        # Block 5
        for _ in range(2):
            self.add(Conv2D(128, kernel_size, padding="same"))
            self.add(BatchNormalization())
            self.add(ReLU())
            self.add(Dropout(drop_rate))
        self.add(MaxPooling2D())

        # Block 6
        self.add(Conv2D(128, kernel_size, padding="same"))
        self.add(BatchNormalization())
        self.add(ReLU())
        self.add(Dropout(drop_rate))

        # Block 7
        self.add(Conv2D(128, 1, padding="same"))
        self.add(BatchNormalization())
        self.add(ReLU())
        self.add(Dropout(drop_rate))

        # Block 8
        self.add(Conv2D(128, 1, padding="same"))
        self.add(BatchNormalization())
        self.add(ReLU())
        self.add(Dropout(drop_rate))
        self.add(MaxPooling2D())

        # Block 9
        self.add(Conv2D(128, kernel_size, padding="same"))
        self.add(BatchNormalization())
        self.add(ReLU())
        self.add(Dropout(drop_rate))

        # Classifier block
        self.add(Flatten()),
        self.add(Dense(10, activation="softmax"))

    def initialize_layers(self):
        n_filters = 32
        kernel_size = 3
        drop_rate = 0.25
        pool_size = 2

        for i in range(3):
            self.add(Conv2D(n_filters, kernel_size, padding="same", input_shape=(32, 32, 3)))
            self.add(BatchNormalization())
            self.add(ReLU())
            self.add(Conv2D(n_filters, kernel_size, padding="same", input_shape=(32, 32, 3)))
            self.add(BatchNormalization())
            self.add(ReLU())
            self.add(MaxPooling2D(pool_size))
            self.add(Dropout(drop_rate))
            n_filters *= 2

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
        self.add(Dense(512, activation="relu"))
        self.add(Dropout(2*drop_rate))
        self.add(Dense(10, activation="softmax"))

