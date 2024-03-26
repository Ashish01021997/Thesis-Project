import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

class Point_PV_Forecasts():

# Generate some example data
 X_train = pd.read_excel('Solar Data.xlsx', sheet_name= 'SolarPV_Xtrain')

 X_test = pd.read_excel('Solar Data.xlsx', sheet_name= 'SolarPV_Xtest')

 y_train = pd.read_excel('Solar Data.xlsx', sheet_name= 'SolarPV_ytrain')

 y_test = pd.read_excel('Solar Data.xlsx', sheet_name= 'SolarPV_ytest')

 timeframe = np.arange(0, 672, 1)

 Solar_Point_Forecasts = np.zeros(24*4)

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
 model = RandomForestRegressor()

# Train the model
 model.fit(X_train, y_train)

# Make predictions on the test set
 y_pred = model.predict(X_test)

 for i in range(96):

  Solar_Point_Forecasts[i] = y_pred[i]

 df = pd.DataFrame(Solar_Point_Forecasts)

# Write DataFrame to Excel file
 excel_file_path = "Solar_Forecast_output.xlsx"  # Specify the path where you want to save the Excel file
 df.to_excel(excel_file_path, index=False)



# Plot the results
 plt.scatter(timeframe, y_test, color='black', label='Actual PV')
 plt.plot(timeframe, y_pred, color='blue', linewidth=3, label='Predicted Point Forecast')
 plt.xlabel('X')
 plt.ylabel('y')
 plt.legend()
 plt.show()
