import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array

# Đọc dữ liệu từ tập tin CSV
data = pd.read_csv('C:/Users/tuanp/Downloads/train.csv')

# Tiền xử lý dữ liệu (bỏ qua datetime, casual, registered)
X = data.drop(columns=['datetime','count', 'casual', 'registered'])

# Sử dụng LabelEncoder cho các biến phân loại
label_encoder = LabelEncoder()
for column in X.columns:
    if X[column].dtype == 'object':
        X[column] = label_encoder.fit_transform(X[column])

# Logarithmic transformation for the 'count' column
y = np.log(data['count'])  # Log transformation of count variable

# Kiểm tra và xử lý missing values
X = check_array(X)
y = check_array(y.to_numpy().reshape(-1, 1))  # Convert Series to NumPy array before reshaping

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Dự đoán số lượng xe đạp
y_pred = model.predict(X_test)

# Chuyển ngược lại đơn vị gốc
y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# Vẽ biểu đồ so sánh giá trị thực tế và dự đoán
plt.figure(figsize=(10, 6))
plt.scatter(y_test_original, y_pred_original, alpha=0.5)
plt.xlabel('Thực tế')
plt.ylabel('Dự đoán')
plt.title('Biểu đồ so sánh giá trị thực tế và dự đoán')
plt.show()
