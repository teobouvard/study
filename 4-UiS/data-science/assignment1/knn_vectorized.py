import pandas as pd
import numpy as np
from collections import Counter

RANDOM = 42
K = 5

def train_test_split(dataset, test_ratio=0.34):

    # suffle dataframe rows
    shuffled_dataset = dataset.sample(frac=1, random_state=RANDOM)

    # compute number of elements in test split
    n_test = int(len(dataset)*test_ratio)

    # split dataset
    x_train = shuffled_dataset[n_test:].drop(columns='class', axis=1)
    y_train = shuffled_dataset[n_test:]['class']

    x_test = shuffled_dataset[:n_test].drop(columns='class', axis=1)
    y_test = shuffled_dataset[:n_test]['class']
            
    return x_train, y_train, x_test, y_test


def predict(x_train, y_train, x_test):
    y_pred = []

    for unknown_flower in x_test.iterrows():
        x_train['distance'] = x_train.apply(lambda x:np.linalg.norm(x.values-unknown_flower[1].values), axis=1)
        x_train.sort_values(by='distance', inplace=True)
        neighbors = x_train.index.values[:K]
        y_pred.append(Counter(y_train[neighbors]).most_common(1)[0][0])
        x_train.drop(columns='distance', inplace=True)

    return y_pred

def evaluate(predections, ground_truth):
    correctly_classified = 0

    for prediction, truth in zip(predections, ground_truth):
        if prediction == truth:
            correctly_classified += 1

    return correctly_classified/len(ground_truth)

if __name__ == "__main__":

    dataset = pd.read_csv('iris.csv', names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
    
    x_train, y_train, x_test, y_test = train_test_split(dataset)

    y_pred = predict(x_train, y_train, x_test)

    score = evaluate(y_pred, y_test)

    print("Accuracy : {0:.4f}".format(score))
