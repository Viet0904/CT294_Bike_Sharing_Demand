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
dt = pd.read_csv("BikeSharingDemand.csv")

# Tạo cột year, month, day, hour từ cột datetime
dt["datetime"] = pd.to_datetime(dt["datetime"])
dt["year"] = dt["datetime"].dt.year
dt["month"] = dt["datetime"].dt.month
dt["weekday"] = dt["datetime"].dt.day
dt["hour"] = dt["datetime"].dt.hour

# Chia dữ liệu thành X và y
X = dt.drop(columns=["casual", "registered", "count", "datetime"])
y = dt["count"] 

# Tách dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

# Huấn luyện mô hình KNN Regression
knnRegressor = KNeighborsRegressor()
knnRegressor.fit(X_train, y_train)

# Đánh giá mô hình KNN Regression trên tập kiểm tra
y_pred_knn = knnRegressor.predict(X_test)
err_knn = mean_squared_error(y_test, y_pred_knn)
print("MSE of KNN = " + str(err_knn))
rmse_err_knn = math.sqrt(err_knn)
print("RMSE of KNN = " + str(round(rmse_err_knn, 3)))
