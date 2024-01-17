import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta

# Generate some sample data for demonstration purposes
# Replace this with your actual data
data = np.arange(0, 288, 1)

# Specify the start time as 00:00
start_time = datetime.strptime('00:00', '%H:%M')

# Create a list of datetime objects for the x-axis
time_points = [start_time + timedelta(minutes=5 * i) for i in range(len(data))]

print(time_points)

# Plot the data
plt.plot(time_points, data)

# Customize the x-axis ticks to show a 3-hour gap
hours_3_interval = 3 * 60  # 3 hours in minutes
tick_positions = range(0, (len(time_points)+1) * 5, hours_3_interval)
tick_labels = [start_time + timedelta(minutes=i) for i in tick_positions]

plt.xticks(tick_labels)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# Rotate x-axis labels for better readability (optional)
plt.gcf().autofmt_xdate()

# Show the plot
plt.show()
