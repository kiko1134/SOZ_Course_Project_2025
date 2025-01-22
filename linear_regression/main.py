import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

file_path = "./data/Sofia_Electricity_Consumption_Updated_2014_2024.csv"
data = pd.read_csv(file_path)

# Preview the data
print(data.head())

# Convert the 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Handle missing values if any
data.ffill()  # Forward fill for simplicity


# Extract time-based features
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

# Define feature columns and target variable
features = ['Year', 'Month', 'Avg_Temperature_C', 'Electricity_Price_BGN_per_MWh', 'Holiday_Period',
            'Population_Thousands']
target = 'Electricity_Consumption_MWh'

# Split into features (X) and target (y)
X = data[features]
y = data[target]


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Display model coefficients
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

future_data = {
    "Year": [2025],
    "Month": [1],
    "Avg_Temperature_C": [1.0],
    "Electricity_Price_BGN_per_MWh": [220],
    "Holiday_Period": [0],
    "Population_Thousands": [1411]
}

future_df = pd.DataFrame(future_data)
future_prediction = model.predict(future_df)

print("Predicted Electricity Consumption for Jan 2025:", future_prediction[0])