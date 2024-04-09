import math
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
import numpy as np

# Load tap du lieu tu file csv
dt = pd.read_csv("./BikeSharingDemand.csv")
print(dt)
# tạo cột year, month, day, hour từ cột datetime
# dt["datetime"] = pd.to_datetime(dt["datetime"])
# dt["year"] = dt["datetime"].dt.year
# dt["month"] = dt["datetime"].dt.month
# dt["weekday"] = dt["datetime"].dt.weekday
# dt["hour"] = dt["datetime"].dt.hour
# Chia dữ liệu thành X và y
X = dt.drop(columns=["casual", "registered", "count", "datetime"])
print(X)
y = dt["count"]  # Áp dụng logarithm cho cột "count"

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 3, random_state=42
)
print(X_train)
print(X_test)


## Bagging
# Mo hinh co so la cay hoi quy
tree = DecisionTreeRegressor()
# Tao 1 tap hop gom 10 mo hinh cay hoi quy
baggingTree = BaggingRegressor(estimator=tree, n_estimators=10, random_state=42)
# Huan luyen mo hinh
baggingTree.fit(X_train, y_train)
# Du doan ket qua
y_pred = baggingTree.predict(X_test)

err_DT = mean_squared_error(y_test, y_pred)
print("MSE =", str(err_DT))
rmse_err_DT = math.sqrt(err_DT)
print("RMSE =" + str(round(rmse_err_DT, 3)))

# Huan luyen bagging voi mo hinh hoi quy tuyen tinh
lm = linear_model.LinearRegression()
baggingLM = BaggingRegressor(estimator=lm, n_estimators=10, random_state=42)
baggingLM.fit(X_train, y_train)
y_pred = baggingLM.predict(X_test)

err_LM = mean_squared_error(y_test, y_pred)
print("MSE =", str(err_LM))
rmse_err_LM = math.sqrt(err_LM)
print("RMSE =" + str(round(rmse_err_LM, 3)))

# Tao mo hinh tu cay hoi quy
treeRegressor = DecisionTreeRegressor()
treeRegressor.fit(X_train, y_train)
# Tao mo hinh tu hoi quy tuyen tinh
LmRegressor = LinearRegression()
LmRegressor.fit(X_train, y_train)
# Tao mo hinh hoi quy voi giai thuat k lang gieng gan nhat
knnRegressor = KNeighborsRegressor()
knnRegressor.fit(X_train, y_train)

# Du doan ket qua va danh gia loi rieng cho mo hinh cay hoi quy
y_pred_tree = treeRegressor.predict(X_test)
err_tree = mean_squared_error(y_test, y_pred_tree)
print("MSE of tree = " + str(err_tree))
rmse_err_tree = math.sqrt(err_tree)
print("RMSE of tree = " + str(round(rmse_err_tree, 3)))

# Du doan ket qua va danh gia loi rieng cho mo hinh hoi quy tuyen tinh
y_pred_lm = LmRegressor.predict(X_test)
err_lm = mean_squared_error(y_test, y_pred_lm)
print("MSE of lm = " + str(err_lm))
rmse_err_lm = math.sqrt(err_lm)
print("RMSE of lm = " + str(round(rmse_err_lm, 3)))

# Du doan ket qua va danh gia loi rieng cho mo hinh KNN
y_pred_knn = knnRegressor.predict(X_test)
err_knn = mean_squared_error(y_test, y_pred_knn)
print("MSE of KNN = " + str(err_knn))
rmse_err_knn = math.sqrt(err_knn)
print("RMSE of KNN = " + str(round(rmse_err_knn, 3)))

# Ket hop 3 mo hinh voi VotingRegressor
voting_reg = VotingRegressor(
    estimators=[
        ("tree_reg", treeRegressor),
        ("Lm_reg", LmRegressor),
        ("knn_reg", knnRegressor),
    ]
)
voting_reg.fit(X_train, y_train)

# Du doan ket qua va danh gia loi chung sau khi ket hop 3 mo hinh
y_pred_ensemble = voting_reg.predict(X_test)
err_ensemble = mean_squared_error(y_test, y_pred_ensemble)
print("MSE of Ensemble = " + str(err_ensemble))
rmse_err_ensemble = math.sqrt(err_ensemble)
print("RMSE of Ensemble = " + str(round(rmse_err_ensemble, 3)))

print(y_pred_ensemble)


# # Tạo DataFrame mới từ dữ liệu test và kết quả dự đoán
# df_test = pd.DataFrame(
#     {
#         "season": X_test["season"],
#         "holiday": X_test["holiday"],
#         "workingday": X_test["workingday"],
#         "weather": X_test["weather"],
#         "temp": X_test["temp"],
#         "atemp": X_test["atemp"],
#         "humidity": X_test["humidity"],
#         "windspeed": X_test["windspeed"],
#         "Actual_Count": (y_test),
#         "Predicted_Count": (y_pred_ensemble),
#     }
# )

# # Lưu DataFrame mới này vào tệp CSV
# df_test.to_csv("./Luu_log/Bagging_predictions.csv", index=False)
