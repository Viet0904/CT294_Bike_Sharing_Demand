import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
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

# Số lượng fold
n_splits = 100

# Khởi tạo KFold
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Danh sách để lưu trữ MSE và RMSE cho từng mô hình
mse_results = []
rmse_results = []

# Lặp qua các fold
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Huấn luyện mô hình Linear Regression
    lm = LinearRegression()
    lm.fit(X_train, y_train)

    # Đánh giá mô hình Linear Regression trên tập kiểm tra
    y_pred_lm = lm.predict(X_test)
    err_lm = mean_squared_error(y_test, y_pred_lm)
    mse_results.append(err_lm)
    rmse_results.append(math.sqrt(err_lm))
    
# Tính trung bình của MSE và RMSE từ các fold
avg_mse = sum(mse_results) / len(mse_results)
avg_rmse = sum(rmse_results) / len(rmse_results)

# In ra kết quả trung bình của MSE và RMSE
print("Average MSE of LinearRegression  = ", avg_mse)
print("Average RMSE of LinearRegression = ", avg_rmse)


