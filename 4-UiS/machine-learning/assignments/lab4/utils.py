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
    cm = confusion_matrix(predicted_labels, true_labels)
    return cm.trace() / cm.sum()


def error_score(predicted_labels, true_labels):
    return 1 - accuracy_score(predicted_labels, true_labels)
