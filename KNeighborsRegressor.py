import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

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
y = df["count"]

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 3, random_state=42
)

# Xây dựng và huấn luyện mô hình KNN Regression
model = KNeighborsRegressor(n_neighbors=9, p=1, weights="distance")
model.fit(x_train, y_train)
# Dự đoán kết quả trên tập kiểm tra
y_pred = model.predict(x_test)

# Tính và in ra giá trị MSE và RMSE
print("MSE = ", mean_squared_error(y_test, y_pred))
print("RMSE = ", np.sqrt(mean_squared_error(y_test, y_pred)))

# Vẽ biểu đồ giá trị thực tế và giá trị dự đoán với đường phân chia
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],   linestyle="--", lw=2,color="red")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values with Identity Line")
plt.show()


from sklearn.model_selection import GridSearchCV

# Định nghĩa các giá trị tham số để kiểm tra tham số tốt nhất
# param_grid = {
#     "n_neighbors": [
#         1,
#         3,
#         5,
#         7,
#         9,
#         11,
#         13,
#         15,
#         17,
#         19,
#         21,
#         23,
#         25,
#         27,
#         29,
#         31,
#         33,
#         35,
#         37,
#         39,
#         41,
#         43,
#         45,
#         47,
#         49,
#         51,
#         53,
#         55,
#         57,
#         59,
#         61,
#         63,
#         65,
#         67,
#         69,
#         71,
#         73,
#         75,
#         77,
#         79,
#         81,
#         83,
#         85,
#         87,
#         89,
#         91,
#         93,
#         95,
#         97,
#         99,
#     ],
#     "weights": ["uniform", "distance"],
#     "p": [1, 2],
# }

# # Tạo GridSearchCV
# grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5)

# # Huấn luyện GridSearchCV object
# grid_search.fit(x_train, y_train)

# # In ra các tham số tốt nhất
# print("Best parameters found: ", grid_search.best_params_)

# # Dự đoán kết quả trên tập kiểm tra với các tham số tốt nhất
# y_pred_grid = grid_search.predict(x_test)

# # Tính và in ra giá trị MSE và RMSE với các tham số tốt nhất
# print("MSE with best parameters = ", mean_squared_error(y_test, y_pred_grid))
# print("RMSE with best parameters = ", np.sqrt(mean_squared_error(y_test, y_pred_grid)))
