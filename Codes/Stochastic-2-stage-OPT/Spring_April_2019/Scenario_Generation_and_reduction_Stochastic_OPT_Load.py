import numpy as np
import pandas as pd
from scipy.stats import rv_discrete
from sklearn.cluster import KMeans
import Quantile_Load_forecasts
import matplotlib.pyplot as plt
class Scenario_Generatiom_Reduction_Load_Quantiles():

 Load_Quantiles = Quantile_Load_forecasts.Load_Quantiles()

 df_predictions_PV = Load_Quantiles.df_predictions_load_15min

 num_rows = 576

 columns = 3

 columns_original_scenarios = 100

 df_original_scenarios = pd.DataFrame(0, index=range(num_rows), columns=range(columns_original_scenarios))

 df_reduced_scenarios = pd.DataFrame(0, index=range(num_rows), columns= range(columns))

 df_reduced_scenarios_probabilities = pd.DataFrame(0, index=range(num_rows), columns=range(columns))

 timescale = np.arange(0, 576, 1)

 for i in range(len(timescale)):

   def generate_scenarios_with_probabilities(quantiles, num_scenarios):
    quantile_levels = sorted(quantiles.keys())
    values_at_quantiles = [quantiles[level] for level in quantile_levels]

    # Calculate probabilities for each quantile
    probabilities = [quantile_levels[i] - quantile_levels[i - 1] if i > 0 else quantile_levels[i] for i in
                     range(len(quantile_levels))]

    # Normalize probabilities to ensure they sum to 1
    probabilities /= np.sum(probabilities)

    # Create a custom discrete distribution using scipy.stats.rv_discrete
    custom_dist = rv_discrete(name='custom', values=(values_at_quantiles, probabilities))

    # Generate scenarios and their corresponding probabilities
    scenarios = custom_dist.rvs(size=num_scenarios)
    scenario_probabilities = custom_dist.cdf(scenarios)

    return scenarios, scenario_probabilities


   def reduce_scenarios(scenarios, scenario_probabilities, k):
    # Combine scenarios and probabilities into a single array
    data = np.column_stack((scenarios, scenario_probabilities))

    # Apply k-means clustering to reduce scenarios
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)

    # Retrieve cluster centers as reduced scenarios
    reduced_scenarios = kmeans.cluster_centers_[:, 0]

    # Assign probabilities to reduced scenarios based on the cluster sizes
    reduced_probabilities = np.bincount(kmeans.labels_) / len(scenarios)

    return reduced_scenarios, reduced_probabilities


# Example usage
   quantiles = {0.1: df_predictions_PV.iloc[i, 0], 0.2: df_predictions_PV.iloc[i, 1], 0.3: df_predictions_PV.iloc[i, 2], 0.4: df_predictions_PV.iloc[i, 3], 0.5: df_predictions_PV.iloc[i, 4],
               0.6: df_predictions_PV.iloc[i, 5], 0.7: df_predictions_PV.iloc[i, 6], 0.8: df_predictions_PV.iloc[i, 7], 0.9: df_predictions_PV.iloc[i, 8]}

   num_scenarios = 100

# Generate scenarios and their probabilities using CDF
   scenarios, probabilities = generate_scenarios_with_probabilities(quantiles, num_scenarios)

   df_original_scenarios.iloc[i, :] = scenarios

# Perform scenario reduction using k-means
   k = 3  # Number of clusters (adjust as needed)

   reduced_scenarios, reduced_probabilities = reduce_scenarios(scenarios, probabilities, k)

   df_reduced_scenarios.iloc[i, :] = reduced_scenarios

   df_reduced_scenarios_probabilities.iloc[i, :] = reduced_probabilities

# Display results
   print("Original Scenarios:", scenarios)
   print("Original Probabilities:", probabilities)
   print("\nReduced Scenarios:", reduced_scenarios)
   print("Reduced Probabilities:", reduced_probabilities)

 average_probabilities_scenarios = [0, 0, 0]

 for i in range(columns_original_scenarios):

  plt.plot(timescale, df_original_scenarios.iloc[:, i])

 plt.title('Original Scenarios Quantile Regression Load Forecasts')
 plt.xlabel('X')
 plt.ylabel('y')
 plt.legend()
 plt.show()

 for i in range(columns):

  average_probabilities_scenarios[i] = df_reduced_scenarios_probabilities.iloc[:, i].mean()

  plt.plot(timescale, df_reduced_scenarios.iloc[:, i], label = average_probabilities_scenarios[i] )

 plt.title('Reduced Scenarios Quantile Regression Load Forecasts')
 plt.xlabel('X')
 plt.ylabel('y')
 plt.legend()
 plt.show()

 print(average_probabilities_scenarios)

