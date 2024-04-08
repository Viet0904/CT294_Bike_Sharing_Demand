import math
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import VotingRegressor

# Load dữ liệu từ file CSV
dt = pd.read_csv("./BikeSharingDemand.csv")
# tạo cột year, month, day, hour từ cột datetime
dt["datetime"] = pd.to_datetime(dt["datetime"])

dt["month"] = dt["datetime"].dt.month
dt["weekday"] = dt["datetime"].dt.weekday
dt["hour"] = dt["datetime"].dt.hour
dt["year"] = dt["datetime"].dt.year
print(dt.head())

# Chia dữ liệu thành X và y
X = dt.drop(columns=["casual", "registered", "count", "datetime"])
y = dt["count"]

# Số lượng fold
n_splits = 50

# Khởi tạo KFold
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Danh sách để lưu trữ MSE và RMSE cho từng mô hình
mse_results = []
rmse_results = []

# Lặp qua các fold
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    ## Bagging
    # Mô hình cơ sở là Decision Tree
    tree = DecisionTreeRegressor(max_depth=10, min_samples_split=5)
    # Tạo mô hình Bagging với 10 mô hình cơ sở
    baggingTree = BaggingRegressor(estimator=tree, n_estimators=100, random_state=42)
    # Huấn luyện mô hình
    baggingTree.fit(X_train, y_train)
    # Dự đoán kết quả
    y_pred = baggingTree.predict(X_test)
    # Đánh giá mô hình
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    mse_results.append(mse)
    rmse_results.append(rmse)

# Tính trung bình của MSE và RMSE từ các fold
avg_mse = sum(mse_results) / len(mse_results)
avg_rmse = sum(rmse_results) / len(rmse_results)

# In ra kết quả trung bình của MSE và RMSE
print("Average MSE =", avg_mse)
print("Average RMSE =", avg_rmse)

import matplotlib.pyplot as plt

# Vẽ giá trị dự đoán so với giá trị thực tế
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color="blue")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--",
    color="red",
)
plt.title("Giá trị dự đoán vs. Giá trị thực tế")
plt.xlabel("Giá trị thực tế")
plt.ylabel("Giá trị dự đoán")
plt.show()
