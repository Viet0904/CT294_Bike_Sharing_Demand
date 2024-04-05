import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load dữ liệu từ file csv
dt = pd.read_csv("./train.csv")

print(dt.head())
# Loại bỏ cột datetime
dt.drop(columns=["datetime"], inplace=True)
dt.drop(columns=["casual"], inplace=True)
dt.drop(columns=["registered"], inplace=True)

print(dt)

# Tách thuộc tính và nhãn
X = dt.iloc[:, 0:-1]  # Lấy các cột từ cột thứ 1 trở đi
print(X)
y = dt.iloc[:, -1]  # Lấy cột đầu tiên
print("Tap Thuoc Tinh")
print(X)
print("Tap Nhan")
print(y)


# This code snippet is performing the following tasks:
# Tách dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Huấn luyện mô hình Linear Regression
lm = LinearRegression()
lm.fit(X_train, y_train)

# Đánh giá mô hình Linear Regression trên tập kiểm tra
y_pred_lm = lm.predict(X_test)
err_lm = mean_squared_error(y_test, y_pred_lm)
print("MSE of LinearRegression  = ", mean_squared_error(y_test, y_pred_lm))
print("RMSE of LinearRegression = ", np.sqrt(mean_squared_error(y_test, y_pred_lm)))

# Huấn luyện mô hình Decision Tree Regression
treeRegressor = DecisionTreeRegressor()
treeRegressor.fit(X_train, y_train)

# Đánh giá mô hình Decision Tree Regression trên tập kiểm tra
y_pred_tree = treeRegressor.predict(X_test)
err_tree = mean_squared_error(y_test, y_pred_tree)
print("MSE of Decision Tree = " + str(err_tree))
rmse_err_tree = math.sqrt(err_tree)
print("RMSE of Decision Tree = " + str(round(rmse_err_tree, 3)))

# Huấn luyện mô hình KNN Regression
knnRegressor = KNeighborsRegressor()
knnRegressor.fit(X_train, y_train)

# Đánh giá mô hình KNN Regression trên tập kiểm tra
y_pred_knn = knnRegressor.predict(X_test)
err_knn = mean_squared_error(y_test, y_pred_knn)
print("MSE of KNN = " + str(err_knn))
rmse_err_knn = math.sqrt(err_knn)
print("RMSE of KNN = " + str(round(rmse_err_knn, 3)))
