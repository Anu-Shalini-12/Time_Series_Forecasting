# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = r'C:\Users\0042H8744\climate timeseries data.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Convert the 'Date' column to datetime format with the correct format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)

# Visualize the time series data
plt.figure(figsize=(12, 6))
plt.plot(df['meantemp'], label='Mean Temperature')
plt.title('Time Series Data - Mean Temperature in Delhi')
plt.xlabel('Date')
plt.ylabel('Mean Temperature')
plt.legend()
plt.show()

# Split the dataset into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Train the ARIMA model
model = ARIMA(train['meantemp'], order=(5,1,0))
model_fit = model.fit()

# Make predictions on the test set
predictions = model_fit.forecast(steps=len(test))

# Evaluate the model
mse = mean_squared_error(test['meantemp'], predictions)
rmse = np.sqrt(mse)
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

# Visualize the predictions
plt.figure(figsize=(12, 6))
plt.plot(train['meantemp'], label='Training Data')
plt.plot(test['meantemp'], label='Actual Test Data')
plt.plot(test.index, predictions, label='Predictions', color='red')
plt.title('ARIMA Time Series Forecasting - Mean Temperature in Delhi')
plt.xlabel('Date')
plt.ylabel('Mean Temperature')
plt.legend()
plt.show()
