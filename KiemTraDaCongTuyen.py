import pandas as pd
import statsmodels.api as sm

# Đọc dữ liệu từ dataframe hoặc từ file csv
data = pd.read_csv("train.csv")

# Chia dữ liệu thành biến độc lập (X) và biến phụ thuộc (y)
X = data[
    [
        "season",
        "holiday",
        "workingday",
        "weather",
        "temp",
        "atemp",
        "humidity",
        "windspeed",
    ]
]
y = data["count"]

# Thêm hạng mục hằng số vào ma trận X
X = sm.add_constant(X)

# Fit mô hình hồi quy tuyến tính
model = sm.OLS(y, X).fit()

# In summary của mô hình
print(model.summary())

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Xóa cột const từ X trước khi tính toán VIF
X_vif = X.drop("const", axis=1)

# Tính toán VIF cho mỗi biến độc lập
vif = pd.DataFrame()
vif["Feature"] = X_vif.columns
vif["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

print(vif)

print("Xóa cột atemp")
# Xóa cột const từ X trước khi tính toán VIF

X_vif = X.drop(["atemp", "const"], axis=1)

# Tính toán VIF cho mỗi biến độc lập
vif = pd.DataFrame()
vif["Feature"] = X_vif.columns
vif["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

print(vif)

print("Xóa cột humidity")
# Xóa cột const từ X trước khi tính toán VIF

X_vif = X.drop(["atemp", "humidity", "const"], axis=1)

# Tính toán VIF cho mỗi biến độc lập
vif = pd.DataFrame()
vif["Feature"] = X_vif.columns
vif["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

print(vif)
