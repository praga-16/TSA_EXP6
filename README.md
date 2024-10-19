# Ex.No: 6               HOLT WINTERS METHOD
### Date: 



### AIM:

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
# Step 1: Load and Filter Data
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load the gold dataset
data = pd.read_csv('Gold Price Prediction.csv')

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Assuming 'Price Today' is the column of interest for gold price
gold_data = data[['Date', 'Price Today']]

# Step 2: Set Index and Resample
gold_data.set_index('Date', inplace=True)

# Resample data to get monthly average prices
monthly_data = gold_data['Price Today'].resample('M').mean()

# Step 3: Handle Missing Data
monthly_data_clean = monthly_data.dropna()

# Step 4: Split Data
# Train data (all but last 12 months)
train_data = monthly_data_clean[:-12]

# Test data (last 12 months)
test_data = monthly_data_clean[-12:]

# Step 5: Fit Holt-Winters Model (Only Trend, No Seasonality)
holt_winters_model = ExponentialSmoothing(
    train_data,
    trend='add',  # Additive trend
    seasonal=None # No seasonal component
).fit()

# Step 6: Test and Final Forecast
# Forecast for the test period (last 12 months)
test_predictions = holt_winters_model.forecast(steps=12)

# Forecast future prices for the next 24 months (12 test + 12 future)
final_forecast = holt_winters_model.forecast(steps=24)

# Step 7: Plot Results

# Plot for test predictions
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data, label='Training Data', color='blue')
plt.plot(test_data.index, test_data, label='Actual Test Data', color='green')
plt.plot(test_data.index, test_predictions, label='Test Predictions', color='red', linestyle='dashed')
plt.title('Test Predictions vs Actual Prices')
plt.xlabel('Date')
plt.ylabel('Gold Price')
plt.legend()
plt.grid(True)
plt.show()

# Plot for final predictions (next 12 months)
plt.figure(figsize=(12, 6))
plt.plot(monthly_data_clean.index, monthly_data_clean, label='Historical Data', color='blue')
plt.plot(test_data.index, test_predictions, label='Test Predictions', color='red', linestyle='dashed')
plt.plot(final_forecast.index[-12:], final_forecast[-12:], label='Final Forecast (Next 12 Months)', color='purple', linestyle='dashed')
plt.title('Final Forecast for Next 12 Months')
plt.xlabel('Date')
plt.ylabel('Gold Price')
plt.legend()
plt.grid(True)
plt.show()

```
### OUTPUT:


TEST_PREDICTION

![image](https://github.com/user-attachments/assets/abd42d4e-2d4d-4a32-8009-3ecac105963f)


FINAL_PREDICTION
![image](https://github.com/user-attachments/assets/e617854c-39be-483a-97f3-a31c1fb45401)

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
