import pandas as pd

# Đọc dữ liệu từ file CSV vào DataFrame
data = pd.read_csv("./train.csv")

# Chuyển cột 'datetime' sang định dạng datetime
data["datetime"] = pd.to_datetime(data["datetime"])

# Chia tập train và test
train_data = data[data["datetime"].dt.day <= 15]
test_data = data[data["datetime"].dt.day > 15]

# In số lượng dòng của mỗi tập
print("Số lượng dòng trong tập train:", len(train_data))
print("Số lượng dòng trong tập test:", len(test_data))

# Lưu tập train và test vào các file CSV nếu cần
train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)
