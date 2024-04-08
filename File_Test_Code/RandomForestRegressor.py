import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
# pd.options.display.max_columns = 999

# Đọc dữ liệu từ file "train.csv" và lưu vào DataFrame df
df = pd.read_csv("BikeSharingDemand.csv")

# Tạo cột year, month, day, hour từ cột datetime
df["datetime"] = pd.to_datetime(df["datetime"])
df["month"] = df["datetime"].dt.month
df["weekday"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour

# Xóa cột "datetime" khỏi DataFrame df
df = df.drop(columns=["datetime"])

# Chia dữ liệu thành X và y
X = df.drop(columns=["casual", "registered", "count"])
print(X)
y = df["count"]  # Áp dụng logarithm cho cột "count"

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# # Xây dựng và huấn luyện mô hình Random Forest Regression
# model = RandomForestRegressor(n_estimators=1000, random_state=42)
# model.fit(X_train, y_train)

# # Dự đoán kết quả trên tập kiểm tra
# y_pred = model.predict(X_test)

# # Tính và in ra giá trị MSE và RMSE
# print("MSE = ", mean_squared_error(y_test, y_pred))
# print("RMSE = ", np.sqrt(mean_squared_error(y_test, y_pred)))


from sklearn.model_selection import GridSearchCV

# Định nghĩa lưới các tham số bạn muốn thử nghiệm
param_grid = {
    "n_estimators": [
        10,
        20,
        50,
        100,
    ],  # Số lượng cây quyết định trong rừng
    "max_depth": [
        None,
        10,
        20,
        30,
        50,
        100,
    ],  # Độ sâu tối đa của các cây quyết định
    "min_samples_split": [
        2,
        5,
        10,
        20,
        50,
        100,
    ],  # Số lượng mẫu tối thiểu để chia một nút
    "min_samples_leaf": [
        1,
        2,
        4,
        10,
        20,
        50,
        100,
    ],  # Số lượng mẫu tối thiểu ở mỗi lá
}

# Tạo một đối tượng GridSearchCV với mô hình RandomForestRegressor và lưới tham số
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
)

# Tiến hành tìm kiếm trên lưới tham số sử dụng tập huấn luyện
grid_search.fit(X_train, y_train)

# In ra tham số tốt nhất đã tìm được
print("Best parameters:", grid_search.best_params_)

# Dự đoán kết quả trên tập kiểm tra sử dụng mô hình đã được điều chỉnh
y_pred_grid = grid_search.predict(X_test)

# Tính và in ra giá trị MSE và RMSE của mô hình tối ưu
print("MSE (optimized) =", mean_squared_error(y_test, y_pred_grid))
print("RMSE (optimized) =", np.sqrt(mean_squared_error(y_test, y_pred_grid)))

plt.figure(figsize=(20, 15))
plot_tree(
    model, max_depth=3, feature_names=X.columns, filled=True, rounded=True, fontsize=8
)
plt.title("Decision Tree Regression (Max Depth = 10)", fontsize=16)
plt.show()
