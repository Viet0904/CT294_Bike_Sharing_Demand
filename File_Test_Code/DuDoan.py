import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import math

warnings.filterwarnings("ignore")
# Load dữ liệu từ file CSV
dt = pd.read_csv("BikeSharingDemand.csv")
print("Hiển thị 5 dòng đầu của tập data\n ", dt.head())
print("Xem kiểu dữ liệu của từng thuộc tính ", dt.dtypes)
print("Xem thông tin của dữ liệu ", dt.info())
print("Kiểm tra giá trị thiếu ", dt.isnull().sum())

# tạo cột year, month, day, hour,weekday từ cột datetime
dt["datetime"] = pd.to_datetime(dt["datetime"])
dt["year"] = dt["datetime"].dt.year
dt["month"] = dt["datetime"].dt.month
dt["weekday"] = dt["datetime"].dt.weekday
dt["hour"] = dt["datetime"].dt.hour
print("Hiển thị 5 dòng đầu của tập data\n ", dt.head())
print("Xem kiểu dữ liệu của từng thuộc tính ", dt.dtypes)
print("Xem thông tin của dữ liệu ", dt.info())
print("Kiểm tra giá trị thiếu ", dt.isnull().sum())


# Chia dữ liệu thành X và y
X = dt.drop(columns=["casual", "registered", "count", "datetime"])
print(X)
y = dt["count"]
print(y)
# Số lượng fold
n_splits = 10

# Khởi tạo KFold
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)


# Tạo figure và axes
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))
# Khởi tạo index fold
fold_index = 0
# Danh sách để lưu trữ MSE và RMSE cho từng mô hình
mse_results = []
rmse_results = []
# Lặp qua các fold
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # Xây dựng và huấn luyện mô hình KNN Regression trên fold hiện tại
    model = RandomForestRegressor(
        n_estimators=100, max_depth=None, min_samples_leaf=1, min_samples_split=2
    )
    model.fit(X_train, y_train)

    # Dự đoán kết quả trên fold kiểm tra
    y_pred = model.predict(X_test)
    # Đánh giá mô hình
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    mse_results.append(mse)
    rmse_results.append(rmse)
    # Tính toán hàng và cột của subplot
    row = fold_index // 5
    col = fold_index % 5

    # Vẽ scatter plot cho fold hiện tại
    axes[row, col].scatter(y_pred, y_test, color="blue")
    axes[row, col].plot(
        [min(y_test), max(y_test)],
        [min(y_test), max(y_test)],
        linestyle="--",
        color="red",
    )
    axes[row, col].set_xlabel("Dự đoán")
    axes[row, col].set_ylabel("Thực tế")
    axes[row, col].set_title("Fold {}".format(fold_index + 1))
    # Tăng index fold
    fold_index += 1


# Hiển thị hình
plt.tight_layout()
plt.show()
# Tính trung bình của MSE và RMSE từ các fold
avg_mse = sum(mse_results) / len(mse_results)
avg_rmse = sum(rmse_results) / len(rmse_results)

# In ra kết quả trung bình của MSE và RMSE
print("RandomForestRegressor Average MSE =", avg_mse)
print("RandomForestRegressor Average RMSE =", avg_rmse)


data_new = pd.read_csv("test.csv")

data_new["datetime"] = pd.to_datetime(data_new["datetime"])
data_new["year"] = data_new["datetime"].dt.year
data_new["month"] = data_new["datetime"].dt.month
data_new["weekday"] = data_new["datetime"].dt.weekday
data_new["hour"] = data_new["datetime"].dt.hour
data_new = data_new.drop(columns=["datetime"])
print(data_new)
y_pred = model.predict(data_new)
data_new["Predict Count"] = y_pred
data_new.to_csv("test.csv", index=False)
print(data_new)

