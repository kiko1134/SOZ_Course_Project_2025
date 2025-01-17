import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

print("Зареждане на данните...")
data_daily = pd.read_csv('Daily_Electricity_Consumption_Data.csv')
data_daily['Date'] = pd.to_datetime(data_daily['Date'])

print("Нормализация на данните...")
# тази функция ще направи всички стойности на Consumption_kWh в диапазон от 0 до 1, за по-лесно обработване
scaler_daily = MinMaxScaler()
data_daily['Consumption_kWh'] = scaler_daily.fit_transform(data_daily[['Consumption_kWh']])

print("Данни след нормализация")
print(data_daily)

print("Създаване на входни данни за модела...")
x_daily = []
y_daily = []
for i in range(7, len(data_daily)):
    x_daily.append(data_daily['Consumption_kWh'][i-7:i].values)
    y_daily.append(data_daily['Consumption_kWh'][i])

X_daily = np.array(x_daily)
y_daily = np.array(y_daily)

print("X_daily",X_daily)
print("y_daily",y_daily)

print("Разделяне на данните на обучаващи и тестови...")
X_train_daily, X_test_daily, y_train_daily, y_test_daily = train_test_split(X_daily, y_daily, test_size=0.2, random_state=42)

print("Създаване и обучение на невронна мрежа...")
model_daily = tf.keras.models.Sequential([
    tf.keras.layers.Dense(50, activation='relu', input_shape=(X_train_daily.shape[1],)),
    tf.keras.layers.Dense(1)
])
model_daily.compile(optimizer='adam', loss='mean_squared_error')
model_daily.fit(X_train_daily, y_train_daily, epochs=50, batch_size=32, validation_split=0.2)

print("Прогнозиране на потреблението за следващата година...")
last_7_days = data_daily['Consumption_kWh'][-7:].values.reshape(1, -1)
future_consumption_daily = []
for _ in range(365):  # Прогнозиране за 365 дни
    next_day = model_daily.predict(last_7_days)
    future_consumption_daily.append(next_day[0][0])
    last_7_days = np.append(last_7_days[:, 1:], next_day).reshape(1, -1)

print("Обръщане на мащабирането на прогнозираните стойности...")
future_consumption_daily_scaled = scaler_daily.inverse_transform(np.array(future_consumption_daily).reshape(-1, 1))

# Изведи примерни прогнозирани стойности
print("Прогнозирани стойности за първата седмица на 2025 година:")
print(future_consumption_daily_scaled[:7])
