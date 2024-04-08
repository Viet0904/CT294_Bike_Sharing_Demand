import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
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

# Định nghĩa các giá trị tham số cần tìm kiếm
param_grid = {
    'fit_intercept': [True, False],
    'positive': [True, False],
    'copy_X': [True, False],
    'n_jobs': [None, -1]
    
}

# Tạo một đối tượng GridSearchCV
grid_search = GridSearchCV(estimator=LinearRegression(), param_grid=param_grid)

# Tiến hành tìm kiếm trên tập huấn luyện
grid_search.fit(X_train, y_train)

# Lấy ra các thông số tốt nhất
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# In ra các thông số tốt nhất
print("Best parameters:", best_params)
print("Best score:", -best_score)

# Sử dụng các thông số tốt nhất để huấn luyện mô hình
best_lm = LinearRegression(**best_params)
best_lm.fit(X_train, y_train)

# Đánh giá mô hình Linear Regression sử dụng các thông số tốt nhất trên tập kiểm tra
y_pred_best = best_lm.predict(X_test)
err_best = mean_squared_error(y_test, y_pred_best)
print("MSE of Linear Regression (best params) = ", err_best)
print("RMSE of Linear Regression (best params) = ", np.sqrt(err_best))
