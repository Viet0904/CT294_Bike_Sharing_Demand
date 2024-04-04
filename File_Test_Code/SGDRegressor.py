from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dt = pd.read_csv("train.csv")
X = dt[
    [
        "season",
        "holiday",
        "workingday",
        "weather",
        "temp",
        "atemp",
        "humidity",
        "windspeed",
    ]
]
y = dt["count"]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


sgdr = SGDRegressor(
    max_iter=100,
    alpha=0.01,
    fit_intercept=False,
    shuffle=True,
)
print(sgdr)
sgdr.fit(X_train, y_train)
score = sgdr.score(X_train, y_train)
print("R-squared:", score)

ypred = sgdr.predict(X_test)
mse = mean_squared_error(y_test, ypred)
print("MSE: ", mse)
print("RMSE: ", mse ** (1 / 2.0))
