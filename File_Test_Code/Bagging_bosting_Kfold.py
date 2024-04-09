import math
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Load dữ liệu từ file CSV
dt = pd.read_csv("./BikeSharingDemand.csv")
# tạo cột year, month, day, hour từ cột datetime
dt["datetime"] = pd.to_datetime(dt["datetime"])
dt["year"] = dt["datetime"].dt.year
dt["month"] = dt["datetime"].dt.month
dt["weekday"] = dt["datetime"].dt.weekday
dt["hour"] = dt["datetime"].dt.hour
# Chia dữ liệu thành X và y
X = dt.drop(columns=["casual", "registered", "count", "datetime"])
y = dt["count"]

# Số lượng fold
n_splits = 5

# Khởi tạo KFold
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Danh sách để lưu trữ MSE và RMSE cho từng mô hình
mse_results = []
rmse_results = []

# Lặp qua các fold
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Tạo và huấn luyện mô hình từ cơ sở Decision Tree
    treeRegressor = DecisionTreeRegressor()
    treeRegressor.fit(X_train, y_train)
    y_pred_tree = treeRegressor.predict(X_test)
    mse_tree = mean_squared_error(y_test, y_pred_tree)
    rmse_tree = math.sqrt(mse_tree)
    mse_results.append(mse_tree)
    rmse_results.append(rmse_tree)

    # Tạo và huấn luyện mô hình từ cơ sở Linear Regression
    lmRegressor = LinearRegression()
    lmRegressor.fit(X_train, y_train)
    y_pred_lm = lmRegressor.predict(X_test)
    mse_lm = mean_squared_error(y_test, y_pred_lm)
    rmse_lm = math.sqrt(mse_lm)
    mse_results.append(mse_lm)
    rmse_results.append(rmse_lm)

    # Tạo và huấn luyện mô hình từ cơ sở KNN
    knnRegressor = KNeighborsRegressor()
    knnRegressor.fit(X_train, y_train)
    y_pred_knn = knnRegressor.predict(X_test)
    mse_knn = mean_squared_error(y_test, y_pred_knn)
    rmse_knn = math.sqrt(mse_knn)
    mse_results.append(mse_knn)
    rmse_results.append(rmse_knn)

    # Tạo và huấn luyện mô hình Gradient Boosting
    gradient_boosting_reg = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, loss="squared_error", random_state=42
    )
    gradient_boosting_reg.fit(X_train, y_train)
    y_pred_gradient_boosting = gradient_boosting_reg.predict(X_test)
    mse_gradient_boosting = mean_squared_error(y_test, y_pred_gradient_boosting)
    rmse_gradient_boosting = math.sqrt(mse_gradient_boosting)
    mse_results.append(mse_gradient_boosting)
    rmse_results.append(rmse_gradient_boosting)

# Tính trung bình của MSE và RMSE từ các fold
avg_mse = sum(mse_results) / len(mse_results)
avg_rmse = sum(rmse_results) / len(rmse_results)

# In ra kết quả trung bình của MSE và RMSE
print("Average MSE =", avg_mse)
print("Average RMSE =", avg_rmse)
