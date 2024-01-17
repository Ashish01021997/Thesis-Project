import random

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sktime.utils.plotting import plot_series
import random as random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


class EM_Price_point_forecast():

# Generate some example data

 X_train = np.zeros(168).reshape(-1, 1)


 for i in range(7):

   for j in range(24):

     X_train[i * 24 + j] = j + 1

 print(X_train)

 X_test = np.zeros(24).reshape(-1, 1)


 for i in range(1):

  for j in range(24):

    X_test[i * 24 + j] = j + 1


 df = pd.read_excel('Electricity_Market_Price.xlsx', sheet_name= 'Electricity_Market_Price')
 print(df)

 df_test = pd.read_excel('Electricity_Market_Price.xlsx', sheet_name= 'Electricity_Market_Price_test')


 y_train = df['Preis (EUR/MWh, EUR/tCO2)'].to_numpy()

 y_train = y_train.reshape(-1, 1)

 y_test = df_test['Preis (EUR/MWh, EUR/tCO2)'].to_numpy()

 y_test = y_test.reshape(-1, 1)


 #Create a  regression model
 model = RandomForestRegressor()

# Train the model
 model.fit(X_train, y_train)



 y_pred = model.predict(X_test)

 print(y_pred)

 y_pred_15_min = np.zeros(96).reshape(-1,1)

 # changing from hour to 15 min basis time res of market prices

 for i in range(24):
  for j in range(4):
   y_pred_15_min[i*4+j] = y_pred[i]

 print(y_pred_15_min)

 timeframe = np.arange(0, 24, 1)

 #Plot the results
 plt.scatter(timeframe, y_test, color='black', label='Actual Load')
 plt.plot(timeframe, y_pred, color='blue', linewidth=3, label='Predicted Point Forecast')
 plt.xlabel('X')
 plt.ylabel('y')
 plt.legend()
 plt.show()





















# Import necessary libraries


# Generate synthetic data for binary classification
# Replace this with your own data


#
 #print(y_probabilities)

# Now, y_pred is the mean prediction, and sigma is the standard deviation, providing a measure of uncertainty.

# You can plot the results to visualize the probabilistic forecast.