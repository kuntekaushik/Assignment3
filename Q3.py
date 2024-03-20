import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate fictitious dataset
np.random.seed(0)
n = 100  # Number of months
years = np.random.randint(2010, 2023, n)
months = np.random.randint(1, 13, n)
co2_values = np.random.uniform(300, 500, n)  # Random CO2 values

# Create DataFrame
data = pd.DataFrame({'Year': years, 'Month': months, 'CO2': co2_values})

# Sort the data by year and month
data = data.sort_values(by=['Year', 'Month']).reset_index(drop=True)

# Create lag features for past CO2 values
for i in range(1, 7):  # Using past 6 months to predict next month
    data[f'CO2_lag_{i}'] = data['CO2'].shift(i)

# Drop rows with NaN values (due to creating lag features)
data.dropna(inplace=True)

# Split data into features and target
X = data.drop(['Year', 'Month', 'CO2'], axis=1)
y = data['CO2']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build MLP model
mlp_model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=0)

# Train the model
mlp_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_train = mlp_model.predict(X_train_scaled)
y_pred_test = mlp_model.predict(X_test_scaled)

# Evaluate model
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print("Train MSE:", mse_train)
print("Test MSE:", mse_test)

# Plot predictions vs actual values for test set
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred_test, label='Predicted')
plt.xlabel('Index')
plt.ylabel('CO2 Value')
plt.title('MLP Forecasting - Test Set')
plt.legend()
plt.show()
