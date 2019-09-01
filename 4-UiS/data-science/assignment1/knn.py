import pandas as pd
import numpy as np



if __name__ == "__main__":

    dataset = pd.read_csv('iris.csv', names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
    print(dataset)