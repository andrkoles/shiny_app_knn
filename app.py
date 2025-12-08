from shiny.express import ui, render, input
from shiny import reactive

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, 
                             classification_report)
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    train_size=0.8,
                                                    random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with ui.card():
    ui.card_header("A k-NN Model Trained on Sklearn Breast Cancer Data")

    with ui.layout_sidebar():
        with ui.sidebar():
            ui.input_numeric("k", "Number of Neighbors", value=3, min=1,
                             max=115, step=2),
            ui.input_checkbox("n", "Min-Max Normalization"),
            ui.input_select("w", "Weights", choices=['uniform', 'distance']),
            ui.input_numeric("p", "Minkowski power parameter", value=2, min=1,
                             max=100)

        @reactive.calc
        def train_set():
            if input.n() == True:
                training = X_train_scaled
            else:
                training = X_train
            return training
        
        @reactive.calc
        def test_set():
            if input.n() == True:
                testing = X_test_scaled
            else:
                testing = X_test
            return testing

        
        @reactive.calc
        def classifier():
            classifier = KNeighborsClassifier(n_neighbors=input.k(),
                                              weights=input.w(), 
                                              metric='minkowski', p=input.p())
            classifier.fit(train_set(), y_train)
            return classifier

        @reactive.calc
        def accuracy():
            y_pred = classifier().predict(test_set())
            return accuracy_score(y_test, y_pred)

        @render.plot
        def plot():
            ConfusionMatrixDisplay.from_estimator(classifier(), test_set(), y_test)

        @render.text
        def metric():
            return f"Accuracy: {accuracy()}"
