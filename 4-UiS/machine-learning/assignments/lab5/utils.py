import numpy as np


def confusion_matrix(predicted_labels, true_labels, normalize=True):
    labels = list(np.unique(true_labels))
    cm = np.zeros((len(labels), len(labels)), np.int64)
    for x, y in zip(predicted_labels, true_labels):
        cm[labels.index(x)][labels.index(y)] += 1
    if normalize:
        return cm / cm.sum(axis=0)
    return cm


def accuracy_score(predicted_labels, true_labels):
    cm = confusion_matrix(predicted_labels, true_labels, normalize=False)
    return cm.trace() / cm.sum()


def error_score(predicted_labels, true_labels):
    return 1 - accuracy_score(predicted_labels, true_labels)


def normalize_data(x, y):
    x_train = np.vstack([_.T for _ in x])
    x_test = np.vstack([_.T for _ in y])

    y_train = np.concatenate([[i + 1] * len(samples.T) for i, samples in enumerate(x)])
    y_test = np.concatenate([[i + 1] * len(samples.T) for i, samples in enumerate(y)])

    return x_train, x_test, y_train, y_test
