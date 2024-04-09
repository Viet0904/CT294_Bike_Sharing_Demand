import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Đọc dữ liệu từ tập tin CSV
dt = pd.read_csv('C:/Users/tuanp/Downloads/train.csv')

# Chuyển cột datetime thành các đặc trưng phù hợp
dt["datetime"] = pd.to_datetime(dt["datetime"])
dt["year"] = dt["datetime"].dt.year
dt["month"] = dt["datetime"].dt.month
dt["weekday"] = dt["datetime"].dt.weekday
dt["hour"] = dt["datetime"].dt.hour

# Tạo DataFrame chứa các đặc trưng và nhãn
X = dt.drop(columns=["casual", "registered", "count", "datetime"])
y = dt["count"]

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=42)

# Thiết lập siêu tham số để tìm kiếm
param_grid = {
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [2],
    'splitter': ['best']
}

# Tạo GridSearchCV object
grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit GridSearchCV vào dữ liệu
grid_search.fit(X_train, y_train)

# Lấy các siêu tham số tốt nhất
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Sử dụng mô hình tốt nhất từ GridSearchCV
best_model = grid_search.best_estimator_

# Dự đoán với mô hình tốt nhất
y_pred_best = best_model.predict(X_test)

# Tính MSE với mô hình tốt nhất
mse_best = mean_squared_error(y_test, y_pred_best)

# Tính RMSE từ MSE
rmse_best = np.sqrt(mse_best)

print(f"Mean Squared Error with Best Model: {mse_best:.2f}")
print(f"Root Mean Squared Error with Best Model: {rmse_best:.2f}")
