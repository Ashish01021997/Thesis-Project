import numpy as np
from sklearn.cluster import KMeans


def generate_scenarios_with_probabilities(quantiles, n):
    quantile_levels = sorted(quantiles.keys())
    values_at_quantiles = [quantiles[level] for level in quantile_levels]

    # Calculate probabilities for each quantile
    probabilities = [quantile_levels[i] - quantile_levels[i - 1] if i > 0 else quantile_levels[i] for i in
                     range(len(quantile_levels))]

    # Normalize probabilities to ensure they sum to 1
    probabilities /= np.sum(probabilities)

    # Generate scenarios and their corresponding probabilities
    scenarios = np.random.choice(values_at_quantiles, n, p=probabilities)

    # Calculate probabilities for each generated scenario
    scenario_probabilities = [probabilities[quantile_levels.index(level)] for level in scenarios]

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
quantiles = {0.1: 5, 0.5: 10, 0.9: 15}
num_scenarios = 1000

# Generate scenarios and their probabilities
scenarios, probabilities = generate_scenarios_with_probabilities(quantiles, num_scenarios)

# Perform scenario reduction using k-means
k = 10  # Number of clusters (adjust as needed)
reduced_scenarios, reduced_probabilities = reduce_scenarios(scenarios, probabilities, k)

# Display results
print("Original Scenarios:", scenarios)
print("Original Probabilities:", probabilities)
print("\nReduced Scenarios:", reduced_scenarios)
print("Reduced Probabilities:", reduced_probabilities)
