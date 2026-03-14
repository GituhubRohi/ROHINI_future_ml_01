import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("Sample - Superstore.csv", encoding="latin1")

# Convert date
data['Order Date'] = pd.to_datetime(data['Order Date'])

# Group sales by date
daily_sales = data.groupby('Order Date')['Sales'].sum().reset_index()

# Create time feature
daily_sales['day_number'] = range(len(daily_sales))

X = daily_sales[['day_number']]
y = daily_sales['Sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict future
future_days = [[len(daily_sales)+i] for i in range(30)]
forecast = model.predict(future_days)

print("Future Sales Prediction:", forecast)

# Plot
plt.plot(daily_sales['day_number'], daily_sales['Sales'], label="Actual Sales")
plt.plot(range(len(daily_sales), len(daily_sales)+30), forecast, label="Forecast")

plt.legend()
plt.title("Sales Forecast") 
plt.savefig("sales_forecast.png")
plt.show() 
