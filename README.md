# Time_Series_Forecasting

## Overview
This project focuses on creating a time series forecasting model for predicting future mean temperature values based on historical weather data in Delhi, India. The dataset spans from 1st January 2013 to 24th April 2017, containing parameters such as `meantemp`, `humidity`, `wind_speed`, and `meanpressure`.

## Libraries Used
- pandas
- numpy
- matplotlib
- statsmodels
- scikit-learn

## Steps to Run

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/Anu-Shalini-12/Time_Series_Forecasting.git
    cd Time_Series_Forecasting
    ```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Code:**
    ```bash
    python report.py
    ```

4. **View Results:**
    - The script will print the Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) on the test set.
    - A plot will be generated showing the training data, actual test data, and predictions.

## Output
- **MSE and RMSE:**
    - The script will output the Mean Squared Error and Root Mean Squared Error for the model's performance on the test set.

- **Visualization:**
    - A plot will be generated showing the training data, actual test data, and predictions.

## Additional Notes
- You may need to adjust the model parameters in the `ARIMA` instantiation based on your data characteristics.
- Experiment with different forecasting techniques and hyperparameters to improve model performance.

# Author
Anu-Shalini-12
