import os
import pickle
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

class Main:
    def __init__(self, dataset_path):
        if not os.path.exists("data"):
            os.makedirs("data")

        self.dataset_path = dataset_path

    def execute(self):
        dataset = load_dataset_from(self.dataset_path)
        remove_outliers(dataset)

        input_train, input_test, label_train, label_test = extract_inputs_and_labels_from(dataset)

        model = train(input_train, label_train)

        evaluate(model, input_test, label_test)

        save(model)

def load_dataset_from(path):
    return pd.read_csv(path, usecols=["time", "value"])

def remove_outliers(dataset):
    z = np.abs(stats.zscore(dataset))
    threshold = 5
    outliers = np.where(z > threshold)[0]
    dataset.drop(outliers, axis=0, inplace=True)

def extract_inputs_and_labels_from(dataset):
    inputs = dataset["time"]
    labels = dataset["value"]
    return split_train_and_test(inputs, labels)

def split_train_and_test(inputs, labels):
    input_train, input_test, label_train, label_test = train_test_split(inputs, labels, test_size=0.1)
    input_train = polynomialize(input_train)
    input_test = polynomialize(input_test)
    return input_train, input_test, label_train, label_test

def polynomialize(input):
    transformer = PolynomialFeatures(degree=10)
    input = input.values.reshape(-1, 1)
    input = transformer.fit_transform(input)
    return input

def evaluate(model, input_test, label_test):
    predictions = model.predict(input_test)
    print("R2 Score: {}".format(model.score(input_test, label_test)))
    print("MSE: {}".format(mean_squared_error(label_test, predictions)))

def train(input_train, label_train):
    model = linear_model.LinearRegression()
    return model.fit(input_train, label_train)

def save(model):
    file_path = "./data/model.sav"
    pickle.dump(model, open(file_path, "wb"))

if __name__ == '__main__':
    app = Main("../ml-workflow-data-layer/data/dataset.csv")

    app.execute()
