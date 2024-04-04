import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# Load data from csv file
dt = pd.read_csv("train.csv")
# Chuyển cột 'datetime' sang định dạng datetime
dt["datetime"] = pd.to_datetime(dt["datetime"])

# Tạo các cột year, month, hour, dayofweek
dt["year"] = dt["datetime"].dt.year
dt["month"] = dt["datetime"].dt.month
dt["hour"] = dt["datetime"].dt.hour
dt["dayofweek"] = dt["datetime"].dt.dayofweek

# Chia features và nhãn

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
        "year",
        "month",
        "hour",
        "dayofweek",
    ]
]
y = dt["count"]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train linear regression model
lm = linear_model.LinearRegression()
lm.fit(X_train, y_train)

# Predictions and evaluation
y_pred = lm.predict(X_test)
err = mean_squared_error(y_test, y_pred)
print("MSE =", str(err))
rmse_err = math.sqrt(err)
print("RMSE =", str(round(rmse_err, 3)))

# Train decision tree regressor
treeRegressor = DecisionTreeRegressor()
treeRegressor.fit(X_train, y_train)

# Predictions and evaluation for decision tree regressor
y_pred_tree = treeRegressor.predict(X_test)
err_tree = mean_squared_error(y_test, y_pred_tree)
print("MSE of tree =", str(err_tree))
rmse_err_tree = math.sqrt(err_tree)
print("RMSE of tree =", str(round(rmse_err_tree, 3)))

# Train KNN regressor
knnRegressor = KNeighborsRegressor()
knnRegressor.fit(X_train, y_train)

# Predictions and evaluation for KNN regressor
y_pred_knn = knnRegressor.predict(X_test)
err_knn = mean_squared_error(y_test, y_pred_knn)
print("MSE of KNN =", str(err_knn))
rmse_err_knn = math.sqrt(err_knn)
print("RMSE of KNN =", str(round(rmse_err_knn, 3)))

# Create ensemble with VotingRegressor
voting_reg = VotingRegressor(
    estimators=[
        ("tree_reg", treeRegressor),
        ("knn_reg", knnRegressor),
    ]
)
voting_reg.fit(X_train, y_train)

# Predictions and evaluation for ensemble
y_pred_ensemble = voting_reg.predict(X_test)
err_ensemble = mean_squared_error(y_test, y_pred_ensemble)
print("MSE of Ensemble =", str(err_ensemble))
rmse_err_ensemble = math.sqrt(err_ensemble)
print("RMSE of Ensemble =", str(round(rmse_err_ensemble, 3)))
