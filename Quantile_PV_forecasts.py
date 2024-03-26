import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

class PV_Quantiles:
 # PV data for May 2010 drawn from Solcast website

  X_train = pd.read_excel('Solar Data.xlsx', sheet_name= 'SolarPV_Xtrain')

  X_test = pd.read_excel('Solar Data.xlsx', sheet_name= 'SolarPV_Xtest')

  y_train = pd.read_excel('Solar Data.xlsx', sheet_name= 'SolarPV_ytrain')

  y_test = pd.read_excel('Solar Data.xlsx', sheet_name= 'SolarPV_ytest')

  timeframe = np.arange(0, 672, 1)


 # Split the data into training and testing sets
 #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  df_predictions_PV = pd.DataFrame({'ten_percent_quantile': pd.Series(dtype='float'),'twenty_percent_quantile': pd.Series(dtype='float'),'thirty_percent_quantile': pd.Series(dtype='float'),'forty_percent_quantile':pd.Series(dtype='float'),
                                    'fifty_percent_quantile':pd.Series(dtype='float'), 'sixty_percent_quantile':pd.Series(dtype='float'), 'seventy_percent_quantile':pd.Series(dtype='float'), 'eighty_percent_quantile' :pd.Series(dtype='float'),
                                    'ninety_percent_quantile': pd.Series(dtype='float') })

 # Define quantiles of interest (e.g., 0.1, 0.5, 0.9)
  quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

 # Train a quantile regression model for each quantile
  quantile_models = {}

  i = 0

  for q in quantiles:
    # Fit model for the specified quantile
    model = GradientBoostingRegressor(loss='quantile', alpha=q, n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    df_predictions_PV.iloc[:, i] = model.predict(X_test)
    i += 1

 # Make predictions for each quantile on the test set
  #quantile_predictions = {q: model.predict(X_test) for q, model in quantile_models.items()}

 # Plot the results
  plt.figure(figsize=(10, 6))
  plt.scatter(timeframe, y_test, color='black', label='data')

  i = 0

  for q in quantiles:

    plt.plot(timeframe, df_predictions_PV.iloc[:, i], label=f'Quantile {q}')
    i += 1


  print(df_predictions_PV)

  plt.title('Quantile Regression PV Forecasts')
  plt.xlabel('X')
  plt.ylabel('y')
  plt.legend()
  plt.show()