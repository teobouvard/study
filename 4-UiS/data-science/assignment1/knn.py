import pandas as pd
import numpy as np
from collections import Counter

RANDOM = 42
K = 5

def train_test_split(dataset, test_ratio=0.34):

    # suffle dataframe rows
    suffled_dataset = dataset.sample(frac=1, random_state=RANDOM)

    # compute n of elements in test split
    n_test = int(len(dataset)*test_ratio)

    # split dataset
    x_train = suffled_dataset[n_test:].drop(columns='class', axis=1)
    y_train = suffled_dataset[n_test:]['class']

    x_test = suffled_dataset[:n_test].drop(columns='class', axis=1)
    y_test = suffled_dataset[:n_test]['class']
            
    return x_train, y_train, x_test, y_test


def predict(x_train, y_train, x_test):
    y_pred = []

    for unknown_flower in x_test.iterrows():
        distances = []

        for known_flower in x_train.iterrows():
            distance = np.linalg.norm(unknown_flower[1]-known_flower[1])
            distances.append((known_flower[0], distance))

        distances.sort(key=lambda x: x[1])

        neighbors = {}

        for distance in distances[:K]:
            if y_train[distance[0]] not in neighbors:
                neighbors[y_train[distance[0]]] = 1
            else :
                neighbors[y_train[distance[0]]] += 1
        
        y_pred.append(Counter(neighbors).most_common(1)[0][0])

    
    return y_pred

def evaluate(predections, ground_truth):
    correctly_classified = 0

    for pred, truth in zip(predections, ground_truth):
        if pred == truth:
            correctly_classified += 1

    return correctly_classified/len(ground_truth)

if __name__ == "__main__":

    dataset = pd.read_csv('iris.csv', names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
    
    x_train, y_train, x_test, y_test = train_test_split(dataset)

    y_pred = predict(x_train, y_train, x_test)

    score = evaluate(y_pred, y_test)

    print(score)
