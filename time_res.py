
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
date_rng = pd.date_range(start='2023-01-01', end='2023-01-07', freq='15T')
data = np.random.randn(len(date_rng))

# Create a DataFrame with the time series data
df = pd.DataFrame(data, index=date_rng, columns=['Value'])

print(df.index)

# Resample the data to 3-hour resolution
df_resampled = df.resample('3H').mean()

# Plot the original and resampled time series
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Value'], label='Original (15 min)')
plt.plot(df_resampled.index, df_resampled['Value'], label='Resampled (3 hours)', linestyle='dashed', marker='o')
plt.title('Time Series Resampling')
plt.xlabel('Time')
plt.ylabel('Value')

# Set custom xticks with tick positions and labels
xtick_positions = df_resampled.index
xtick_labels = [str(ts) for ts in xtick_positions]
plt.xticks(xtick_positions, xtick_labels, rotation=45, ha='right')

plt.legend()
plt.tight_layout()
plt.show()