import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Đọc dữ liệu từ file hoặc từ dataframe nếu đã có
dt = pd.read_csv(
    "BikeSharingDemand.csv"
)  


dt["datetime"] = pd.to_datetime(dt["datetime"])
dt["hour"] = dt["datetime"].dt.hour
dt["month"] = dt["datetime"].dt.month
dt["weekday"] = dt["datetime"].dt.weekday
dt["year"] = dt["datetime"].dt.year
X = dt.drop(columns=["datetime", "count"])

# Thêm một cột constant cho ma trận X để tính toán hệ số chặn
X = add_constant(X)

# Tính VIF cho từng biến độc lập
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
# Tính VIF cho từng biến độc lập
vif_data["VIF"] = [f"{variance_inflation_factor(X.values, i):.2f}" for i in range(X.shape[1])]

print(vif_data)
# Xoá cột constant trong VIF
vif_data = vif_data[vif_data["feature"] != "const"]

print(vif_data)