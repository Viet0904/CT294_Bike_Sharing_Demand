import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Load dữ liệu từ file CSV
dt = pd.read_csv("./BikeSharingDemand.csv")
# tạo cột year, month, day, hour từ cột datetime
# dt["datetime"] = pd.to_datetime(dt["datetime"])
# dt["year"] = dt["datetime"].dt.year
# dt["month"] = dt["datetime"].dt.month
# dt["weekday"] = dt["datetime"].dt.weekday
# dt["hour"] = dt["datetime"].dt.hour


# Chia dữ liệu thành X và y
X = dt.drop(columns=["casual", "registered", "count", "datetime"])
y = dt["count"]

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 3, random_state=42
)

# Tạo mô hình từ cơ sở Decision Tree
treeRegressor = DecisionTreeRegressor()
treeRegressor.fit(X_train, y_train)

# Tạo mô hình từ cơ sở Linear Regression
lmRegressor = LinearRegression()
lmRegressor.fit(X_train, y_train)

# Tạo mô hình từ cơ sở KNN
knnRegressor = KNeighborsRegressor()
knnRegressor.fit(X_train, y_train)

# Tạo mô hình Gradient Boosting với các mô hình cơ sở là Decision Tree, Linear Regression và KNN
gradient_boosting_reg = GradientBoostingRegressor(
    n_estimators=100,  # Số lượng cây
    learning_rate=0.1,  # Tốc độ học
    loss="squared_error",  # Hàm mất mát: squared error (least squares regression)
)

# Huấn luyện mô hình Gradient Boosting
gradient_boosting_reg.fit(X_train, y_train)

# Dự đoán kết quả trên tập kiểm tra cho các mô hình cơ sở
y_pred_tree = treeRegressor.predict(X_test)
y_pred_lm = lmRegressor.predict(X_test)
y_pred_knn = knnRegressor.predict(X_test)

# Dự đoán kết quả trên tập kiểm tra cho mô hình Gradient Boosting
y_pred_gradient_boosting = gradient_boosting_reg.predict(X_test)

# Đánh giá mô hình cho các mô hình cơ sở
mse_tree = mean_squared_error(y_test, y_pred_tree)
mse_lm = mean_squared_error(y_test, y_pred_lm)
mse_knn = mean_squared_error(y_test, y_pred_knn)

# Đánh giá mô hình cho Gradient Boosting
mse_gradient_boosting = mean_squared_error(y_test, y_pred_gradient_boosting)

# In ra kết quả đánh giá
print("MSE of Decision Tree =", mse_tree)
print("RMSE of Decision Tree =", math.sqrt(mse_tree))
print("MSE of Linear Regression =", mse_lm)
print("RMSE of Linear Regression =", math.sqrt(mse_lm))
print("MSE of KNN =", mse_knn)
print("RMSE of KNN =", math.sqrt(mse_knn))
print("MSE of Gradient Boosting =", mse_gradient_boosting)
print("RMSE of Gradient Boosting =", math.sqrt(mse_gradient_boosting))
