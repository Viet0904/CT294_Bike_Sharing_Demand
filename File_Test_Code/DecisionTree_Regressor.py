import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

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
y = df["count"]

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 3, random_state=42
)

from sklearn.model_selection import GridSearchCV

# Định nghĩa lưới các tham số cần tinh chỉnh
param_grid = {
    "min_samples_split": [None, 100, 500, 1000, 2000],
    "max_depth": [None, 10, 20, 30, 50, 100],
    "min_samples_leaf": [1, 2, 5, 10, 20, 30, 50, 100],
    "splitter": ["best", "random"],
}

# Khởi tạo GridSearchCV với mô hình DecisionTreeRegressor và lưới tham số
grid_search = GridSearchCV(
    estimator=DecisionTreeRegressor(random_state=42),
    param_grid=param_grid,
)

# Tiến hành tìm kiếm trên lưới tham số
grid_search.fit(X_train, y_train)

# In ra tham số tốt nhất và điểm số tốt nhất
print("Best parameters found:", grid_search.best_params_)
print("Best score found (negative MSE):", grid_search.best_score_)

# Lấy mô hình tốt nhất đã được điều chỉnh từ GridSearchCV
best_model = grid_search.best_estimator_

# Dự đoán trên tập test với mô hình tốt nhất
y_pred_best = best_model.predict(X_test)

# Tính MSE và RMSE với mô hình tốt nhất
mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)

print("Best Model - Mean Squared Error:", mse_best)
print("Best Model - Root Mean Squared Error:", rmse_best)


# # Xây dựng mô hình Decision Tree Regressor
# model = DecisionTreeRegressor(min_samples_split=1000)
# model.fit(X_train, y_train)

# # Dự đoán trên tập test
# y_pred = model.predict(X_test)

# # Tính MSE và RMSE
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)

# print("Mean Squared Error:", mse)
# print("Root Mean Squared Error:", rmse)

# # Ghi kết quả dự đoán vào file sampleSubmission
# sample_submission = pd.DataFrame({"datetime": test_data["datetime"], "count": y_pred})
# sample_submission.to_csv("DecisionTreeRegressor.csv", index=False)
