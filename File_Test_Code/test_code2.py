import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

# Load and preprocess data
data = pd.read_csv("./train.csv")
data["datetime"] = pd.to_datetime(data["datetime"])
train_data = data[data["datetime"].dt.day <= 15]
test_data = data[data["datetime"].dt.day > 15]

categorical_feature_names = ["season", "holiday", "workingday", "weather"]
for var in categorical_feature_names:
    train_data[var] = train_data[var].astype("category")
    test_data[var] = test_data[var].astype("category")

feature_names = ["season", "weather", "temp", "atemp", "workingday", "humidity", "holiday", "windspeed"]
X_train = train_data[feature_names]
X_test = test_data[feature_names]
y_train = train_data["count"]
y_test = test_data["count"]

# Number of iterations
num_iterations = 100

# Lists to store MSE values
mse_values = []

# Repeat the process for multiple iterations
for i in range(num_iterations):
    # Train the model
    lm = LinearRegression(n_jobs=5)
    lm.fit(X_train, y_train)
    
    # Make predictions
    predictions = lm.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    rmse = math.sqrt(mse)
    mse_values.append(mse)
    print("Iteration", i+1, "- MSE:", mse, "RMSE:", rmse)

# Calculate the average MSE
avg_mse = np.mean(mse_values)
print("Average MSE over", num_iterations, "iterations:", avg_mse)
