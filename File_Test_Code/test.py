import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV

# Load dữ liệu từ file CSV
dt = pd.read_csv("BikeSharingDemand.csv")
print("Hiển thị 5 dòng đầu của tập data\n ", dt.head())
print("Xem kiểu dữ liệu của từng thuộc tính ", dt.dtypes)
print("Xem thông tin của dữ liệu ", dt.info())
print("Kiểm tra giá trị thiếu ", dt.isnull().sum())

# Tạo cột year, month, day, hour, weekday từ cột datetime
dt["datetime"] = pd.to_datetime(dt["datetime"])
dt["year"] = dt["datetime"].dt.year
dt["month"] = dt["datetime"].dt.month
dt["weekday"] = dt["datetime"].dt.weekday
dt["hour"] = dt["datetime"].dt.hour
print("Hiển thị 5 dòng đầu của tập data\n", dt.head())
print("Xem kiểu dữ liệu của từng thuộc tính\n", dt.dtypes)
print("Xem thông tin của dữ liệu\n", dt.info())
print("Kiểm tra giá trị thiếu\n", dt.isnull().sum())

# Chia dữ liệu thành X và y
X = dt.drop(columns=["casual", "registered", "count", "datetime"])
print(X)
y = dt["count"]
print(y)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 3, random_state=42
)
print(X_train)
print(X_test)

# Tạo một đối tượng DecisionTreeRegressor để sử dụng trong BaggingRegressor
base_tree = DecisionTreeRegressor()

# Định nghĩa các giá trị tham số bạn muốn tìm kiếm
param_grid = {
    "n_estimators": [10, 20, 50, 100],
    "max_samples": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    "estimator__min_samples_split": [10, 20, 30, 50, 100],
    "estimator__max_depth": [None, 10, 20, 30, 50, 100],
    "estimator__min_samples_leaf": [1, 2, 5, 10, 20, 50, 100],
    "estimator__splitter": ["best", "random"],
}

# Tạo GridSearchCV với mô hình BaggingRegressor và các giá trị tham số
grid_search = GridSearchCV(
    estimator=BaggingRegressor(estimator=base_tree, random_state=42),
    param_grid=param_grid,
)

# Huấn luyện GridSearchCV để tìm ra bộ tham số tốt nhất
grid_search.fit(X_train, y_train)

# In ra bộ tham số tốt nhất
print("Best parameters found:", grid_search.best_params_)

# Lấy mô hình tốt nhất
best_model = grid_search.best_estimator_

# Dự đoán kết quả trên tập kiểm tra bằng mô hình tốt nhất
y_pred_best = best_model.predict(X_test)

# Đánh giá mô hình tốt nhất
best_err_DT = mean_squared_error(y_test, y_pred_best)
print("Best DecisionTreeRegressor MSE =", str(best_err_DT))
best_rmse_err_DT = math.sqrt(best_err_DT)
print("Best DecisionTreeRegressor RMSE =", str(round(best_rmse_err_DT, 3)))
