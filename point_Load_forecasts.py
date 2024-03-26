import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

class Point_Load_Forecasts():

 # Generate some example data

 X_train = np.zeros(34560).reshape(-1, 1)

 for i in range(24):
     for j in range(24):
         for k in range(60):
             X_train[i * 24 + j * 60 + k] = j + 1

 print(X_train)

 X_test = np.zeros(10080).reshape(-1, 1)

 for i in range(7):
     for j in range(24):
        for k in range(60):
           X_test[i * 24 + j * 60 + k] = j + 1

 print(X_test)

 timeframe = np.arange(0, 10080, 1)

 y_train = pd.read_excel('German Household Load Dataset.xlsx', sheet_name='Forecast-Train Data')

 y_train = y_train.to_numpy()

 y_test = pd.read_excel('German Household Load Dataset.xlsx', sheet_name='Load-Test Data')

 y_test = y_test.to_numpy()

 Load_Point_Forecasts = np.zeros(24*4)

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
 model = RandomForestRegressor()

# Train the model
 model.fit(X_train, y_train)

# Make predictions on the test set
 y_pred = model.predict(X_test)

# Plot the results
 plt.scatter(timeframe, y_test, color='black', label='Actual Load')
 plt.plot(timeframe, y_pred, color='red', linewidth=3, label='Predicted Point Forecast')
 plt.xlabel('X')
 plt.ylabel('y')
 plt.legend()
 plt.show()

 y_pred_15_min = np.zeros(672).reshape(-1,1)

 for i in range(672):
     for j in range(15):
         y_pred_15_min[i] += y_pred[i * 15 + j] / 15


 timeframe_15min = np.arange(0, 672, 1)

 for i in range(96):
     Load_Point_Forecasts[i] = y_pred_15_min[i]

 df = pd.DataFrame(Load_Point_Forecasts)

 # Write DataFrame to Excel file
 excel_file_path = "Load_Forecast_output.xlsx"  # Specify the path where you want to save the Excel file
 df.to_excel(excel_file_path, index=False)

 plt.plot(timeframe_15min, y_pred_15_min, label='Predicted Point Forecast 15 min time res')

 plt.title('Point Load Forecasts')
 plt.xlabel('X')
 plt.ylabel('y')
 plt.legend()
 plt.show()
