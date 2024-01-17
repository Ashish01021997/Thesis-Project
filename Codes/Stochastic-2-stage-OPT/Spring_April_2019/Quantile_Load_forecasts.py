import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Load data drawn from paper/link - HTW-Repraesentative-elektrische-Lastprofile-fuer-Wohngebaeude
class Load_Quantiles:


 X_train = np.zeros(34560).reshape(-1, 1)

 for i in range(24):
    for j in range(24):
        for k in range(60):
           X_train[i*24+j*60+k] = j+1



 X_test = np.zeros(8640).reshape(-1, 1)

 for i in range(6):
    for j in range(24):
        for k in range(60):
         X_test[i*24+j*60+k] = j+1


 timeframe = np.arange(0, 8640, 1) # last 6 days of April 2019

 y_train = pd.read_excel('German Household Load Dataset.xlsx', sheet_name= 'Forecast-Train Data')

 y_train = y_train.to_numpy()

 y_test = pd.read_excel('German Household Load Dataset.xlsx', sheet_name= 'Load-Test Data')

 y_test = y_test.to_numpy()

 #y_train_5_min = np.zeros(2016)

 #y_test_5_min = np.zeros(288)

 #changing temporal resolutions from 1 min to 5 min


# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 df_predictions_load = pd.DataFrame({'ten_percent_quantile': pd.Series(dtype='float'),'twenty_percent_quantile': pd.Series(dtype='float'),'thirty_percent_quantile': pd.Series(dtype='float'),'forty_percent_quantile':pd.Series(dtype='float'),
                                    'fifty_percent_quantile':pd.Series(dtype='float'), 'sixty_percent_quantile':pd.Series(dtype='float'), 'seventy_percent_quantile':pd.Series(dtype='float'), 'eighty_percent_quantile' :pd.Series(dtype='float'),
                                    'ninety_percent_quantile': pd.Series(dtype='float') })

 num_rows = 24*4*6
 num_columns = 9

 column_names = ['ten_percent_quantile', 'twenty_percent_quantile',
      'thirty_percent_quantile', 'forty_percent_quantile',
      'fifty_percent_quantile', 'sixty_percent_quantile',
      'seventy_percent_quantile', 'eighty_percent_quantile',
      'ninety_percent_quantile']

 df_predictions_load_15min = pd.DataFrame(0, index=range(num_rows), columns= column_names)

# Define quantiles of interest (e.g., 0.1, 0.5, 0.9)
 quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Train a quantile regression model for each quantile
 quantile_models = {}

 i = 0

 for q in quantiles:
    # Fit model for the specified quantile
    model = GradientBoostingRegressor(loss='quantile', alpha= q, n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    df_predictions_load.iloc[:, i] = model.predict(X_test)
    i += 1

# Make predictions for each quantile on the test set
 #quantile_predictions = {q: model.predict(X_test) for q, model in quantile_models.items()}

# Plot the results
 plt.figure(figsize=(10, 6))
 plt.scatter(timeframe, y_test, color='black', label='data')

 i = 0

 for q in quantiles:
    plt.plot(timeframe, df_predictions_load.iloc[:, i], label=f'Quantile {q}')
    i += 1


 print(df_predictions_load)

 plt.title('Quantile Regression Load Forecasts')
 plt.xlabel('X')
 plt.ylabel('y')
 plt.legend()
 plt.show()

 # changing time res of quantiles from 1 min to 15 min-

 k = 0

 for q in quantiles:
     for i in range(576):
         for j in range(15):
           df_predictions_load_15min.iloc[i, k] += df_predictions_load.iloc[i*15+j, k]/15
     k += 1

 print(df_predictions_load_15min)

 timeframe_15min = np.arange(0, 576, 1)

 i = 0

 for q in quantiles:
    plt.plot(timeframe_15min, df_predictions_load_15min.iloc[:, i], label=f'Quantile {q}')
    i += 1

 plt.title('Quantile Regression Load Forecasts_time_resolution_15min')
 plt.xlabel('X')
 plt.ylabel('y')
 plt.legend()
 plt.show()
