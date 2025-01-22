import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

file_path = "./data/Sofia_Electricity_Consumption_Updated_2014_2024.csv"
data = pd.read_csv(file_path)

# Провека дали данните се прочитат правилно
print(data.head())

# Конвертираме датата към datetime формат
data['Date'] = pd.to_datetime(data['Date'])

# Проверка за липсващи стойности
print("Missing values:\n", data.isnull().sum())

# При липса на стойности тук се допълват
data.ffill()  # използваме forward fill за олеснение


# Извличаме времево базираните стойности
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

# Дефинираме feature колоните и target променливите
features = ['Year', 'Month', 'Avg_Temperature_C', 'Electricity_Price_BGN_per_MWh', 'Holiday_Period',
            'Population_Thousands']
target = 'Electricity_Consumption_MWh'

# Разделяме на features (X) и target (y)
X = data[features]
y = data[target]


# Разделяме данните на тренирани (80%) и тестови (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Трениране на модел чрез линейна регресия
model = LinearRegression()
model.fit(X_train, y_train)

# Извеждане в конзолата на коефициентите на модела
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# Make predictions on the test set Създаване на предсказание на тестовите данни
y_pred = model.predict(X_test)

# Изчисляваме показателите за оценка в случая (MSE,RMSE,r2)
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

print("Предполагаема консумация на електроенергия за януари 2025:", round(future_prediction[0],2),"MWh")