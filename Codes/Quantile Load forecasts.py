import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate some sample data for x train and x test (26 and 5 day timeframes)
X_train = np.zeros(624).reshape(-1, 1)

for i in range(26):
    for j in range(24):
        X_train[i*24+j] = j+1

X_test = np.zeros(120).reshape(-1, 1)

for i in range(5):
    for j in range(24):
        X_test[i*24+j] = j+1

timeframe = np.arange(0, 120, 1)

y_train = pd.read_excel('German Household Load Dataset.xlsx', sheet_name= 'Forecast-Train Data')

y_train = y_train.to_numpy()

y_test = pd.read_excel('German Household Load Dataset.xlsx', sheet_name= 'Load-Test Data')

y_test = y_test.to_numpy()

y_train_hour = np.zeros(624)

y_test_hour = np.zeros(120)

#changing temporal resolutions from 1 min to 1 hour

for i in range(624):
    for j in range(60):
        y_train_hour[i] += y_train[i*60+j]/60

for i in range(120):
    for j in range(60):
        y_test_hour[i] += y_test[i*60+j]/60

y_train_hour = y_train_hour.reshape(-1, 1)
y_test_hour = y_test_hour.reshape(-1, 1)



# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df_predictions_load = pd.DataFrame({'ten_percent_quantile': pd.Series(dtype='float'),'twenty_percent_quantile': pd.Series(dtype='float'),'thirty_percent_quantile': pd.Series(dtype='float'),'forty_percent_quantile':pd.Series(dtype='float'),
                                    'fifty_percent_quantile':pd.Series(dtype='float'), 'sixty_percent_quantile':pd.Series(dtype='float'), 'seventy_percent_quaantile':pd.Series(dtype='float'), 'eighty_percent_quantile' :pd.Series(dtype='float'),
                                    'ninety_percent_quantile': pd.Series(dtype='float') })

# Define quantiles of interest (e.g., 0.1, 0.5, 0.9)
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Train a quantile regression model for each quantile
quantile_models = {}
for q in quantiles:
    # Fit model for the specified quantile
    model = GradientBoostingRegressor(loss='quantile', alpha=q, n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train_hour)
    quantile_models[q] = model

# Make predictions for each quantile on the test set
quantile_predictions = {q: model.predict(X_test) for q, model in quantile_models.items()}

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(timeframe, y_test_hour, color='black', label='data')

i = 0

for q, predictions in quantile_predictions.items():
    df_predictions_load.iloc[:, i] = predictions
    plt.plot(timeframe, predictions, label=f'Quantile {q}')
    i += 1


print(df_predictions_load)

plt.title('Quantile Regression Load')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()