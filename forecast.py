import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("sales_data.csv")

# Create number index for dates
data['day'] = range(len(data))

X = data[['day']]
y = data['sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict next 5 days
future_days = [[len(data)+i] for i in range(5)]
predictions = model.predict(future_days)

print("Future Sales Prediction:")
print(predictions)

# Plot graph
plt.plot(data['day'], data['sales'], label="Actual Sales")
plt.plot(range(len(data), len(data)+5), predictions, label="Forecast")

plt.xlabel("Days")
plt.ylabel("Sales")
plt.title("Sales Forecast")
plt.legend()

plt.show()
