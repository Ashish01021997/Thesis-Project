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

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
 model = RandomForestRegressor()

# Train the model
 model.fit(X_train, y_train)

# Make predictions on the test set
 y_pred = model.predict(X_test)

# Plot the results
 plt.scatter(timeframe, y_test, color='black', label='Actual PV')
 plt.plot(timeframe, y_pred, color='blue', linewidth=3, label='Predicted Point Forecast')
 plt.xlabel('X')
 plt.ylabel('y')
 plt.legend()
 plt.show()
