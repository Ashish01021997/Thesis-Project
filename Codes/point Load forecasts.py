import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

class Point_Load_Forecasts():

 # Generate some example data

 X_train = np.zeros(2016).reshape(-1, 1)

 for i in range(7):
    for j in range(24):
        for k in range(12):
            X_train[i * 24 + j * 12 + k] = j + 1

 print(X_train)

 X_test = np.zeros(288).reshape(-1, 1)

 for i in range(24):
    for j in range(12):
        X_test[i * 12 + j] = i + 1

 print(X_test)



 y_train = pd.read_excel('German Household Load Dataset.xlsx', sheet_name= 'Forecast-Train Data')

 y_train = y_train.to_numpy()

 y_test = pd.read_excel('German Household Load Dataset.xlsx', sheet_name= 'Load-Test Data')

 y_test = y_test.to_numpy()

 y_train_5_min = np.zeros(2016)

 y_test_5_min = np.zeros(288)

 #changing temporal resolutions from 1 min to 5 min

 for i in range(2016):
    for j in range(5):
        y_train_5_min[i] += y_train[i*5+j]/5

 for i in range(288):
    for j in range(5):
        y_test_5_min[i] += y_test[i*5+j]/5

 y_train_5_min = y_train_5_min.reshape(-1, 1)
 y_test_5_min = y_test_5_min.reshape(-1, 1)

 timeframe = np.arange(0, 288, 1)

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
 model = RandomForestRegressor()

# Train the model
 model.fit(X_train, y_train_5_min)

# Make predictions on the test set
 y_pred = model.predict(X_test)

# Plot the results
 plt.scatter(timeframe, y_test_5_min, color='black', label='Actual Load')
 plt.plot(timeframe, y_pred, color='blue', linewidth=3, label='Predicted Point Forecast')
 plt.xlabel('X')
 plt.ylabel('y')
 plt.legend()
 plt.show()
