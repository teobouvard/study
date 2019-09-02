import pandas as pd
import numpy as np

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


def predict_set(x_train, y_train, x_test):
    y_pred = []

    for unknown_flower in x_test.iterrows():
        distances = []

        for known_flower in x_train.iterrows():
            distance = np.linalg.norm(unknown_flower[1]-known_flower[1])
            distances.append((known_flower[0], distance))

        distances.sort(key=lambda x: x[1])

        most_likely = {}

        for distance in distances[:K]:
            if y_train[distance[0]] not in most_likely:
                most_likely[y_train[distance[0]]] = 1
            else :
                most_likely[y_train[distance[0]]] += 1

            print(most_likely))
        #print(y_train[distances[0][0]])

    
    return y_pred


if __name__ == "__main__":

    dataset = pd.read_csv('iris.csv', names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
    
    x_train, y_train, x_test, y_test = train_test_split(dataset)

    y_pred = predict_set(x_train, y_train, x_test)

    #print(y_pred)
