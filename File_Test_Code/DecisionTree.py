import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import math

# Đọc dữ liệu từ file CSV vào DataFrame
data = pd.read_csv("./train.csv")
# Chuyển cột 'datetime' sang định dạng datetime
data["datetime"] = pd.to_datetime(data["datetime"])
# Chia tập train và test
train_data = data[data["datetime"].dt.day <= 15].copy()  # Tạo một bản sao của DataFrame
test_data = data[data["datetime"].dt.day > 15].copy()    # Tạo một bản sao của DataFrame

# Các feature liên tục và feature hạng mục
# Feature liên tục = ["temp","humidity","windspeed","atemp"]
# Chuyển đổi kiểu của feature hạng mục sang kiểu category
categorical_feature_names = [
    "season",
    "holiday",
    "workingday",
    "weather",
]

for var in categorical_feature_names:
    train_data.loc[:, var] = train_data[var].astype("category")
    test_data.loc[:, var] = test_data[var].astype("category")

feature_names = [
    "season",
    "weather",
    "temp",
    "atemp",
    "workingday",
    "humidity",
    "holiday",
    "windspeed",
]

X_train = train_data[feature_names]
y_train = train_data["count"]

X_test = test_data[feature_names]
y_test = test_data["count"]

# Khởi tạo mô hình Decision Tree
decision_tree = DecisionTreeRegressor(random_state=42)

# Huấn luyện mô hình trên tập huấn luyện
decision_tree.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
predictions_decision_tree = decision_tree.predict(X_test)

# Huấn luyện mô hình trên tập huấn luyện
decision_tree.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
predictions_decision_tree = decision_tree.predict(X_test)

# Đánh giá hiệu suất của mô hình
err_decision_tree = mean_squared_error(y_test, predictions_decision_tree)
print("MSE (Decision Tree) = ", str(err_decision_tree))
rmse_err_decision_tree = math.sqrt(err_decision_tree)
print("RMSE (Decision Tree) = " + str(round(rmse_err_decision_tree, 3)))

# Lưu dự đoán vào DataFrame
predictions_df = pd.DataFrame(predictions_decision_tree, columns=["predicted_count"])

# Thêm cột 'datetime' từ test_data vào predictions_df
predictions_df["datetime"] = test_data["datetime"].values

# Lưu DataFrame vào file Excel
predictions_df.to_csv("predictions_decision_tree.csv", index=False)
