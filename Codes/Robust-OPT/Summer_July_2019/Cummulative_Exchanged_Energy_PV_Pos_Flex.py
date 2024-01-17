import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
from numpy import random
import pandas as pd

P_bat_min = 0 # 0.01kW # Assumption to make P_bat_min equal to zero for simplicity reasons
P_bat_max = 1250 # 1.25kW
C_bat = 4600 # in Wh
Panel_area = 4  # 2msq 2 panels area x 2 panels (1.937 sq m from a article : (https://www.skillstg.co.uk/blog/solar-photovoltaic-panel-sizes/#:~:text=The%20physical%20dimensions%20of%20most,thickness%20of%20around%201.5%20inches)
counter = 0
SOC_min = 0.20
SOC_max = 0.95 # soc limits from the research paper (https://www.researchgate.net/publication/373737720_A_PROGNOSTIC_DECISION-MAKING_APPROACH_UNDER_UNCERTAINTY_FOR_AN_ELECTRIC_VEHICLE_FLEET_ROUTING_PROBLEM)
E_bat_min = SOC_min * C_bat
E_bat_max = SOC_max * C_bat
E_bat_initial = (E_bat_max + E_bat_min)/2

timescale = np.arange(0, 96, 1)

x_array = np.arange(0, 96, 3)

iterations = len(timescale)

PV_Pos_Flex = Flexibility_Prosumer_PV_Battery_Grid_Stochastic_2stage_OPT_PV_Pos_Flex.EMS_OPT_PV_Pos_Flex()

P_PV_export_grid_opt = PV_Pos_Flex.P_PV_export_grid_opt

P_bat_charging_PV_opt = PV_Pos_Flex.P_bat_charging_PV_opt

P_PV_export_grid_opt_flex = PV_Pos_Flex.P_PV_export_grid_opt_flex

E_bat_profile = PV_Pos_Flex.E_bat_profile

E_bat_final = E_bat_initial

E_bat_flex_check = np.zeros(iterations + 1)


P_pv_positive_flex = np.zeros(iterations)
E_pv_positive_flex = np.zeros(iterations)
t_pv_positive_flex = 0.0

P_pv_negative_flex = np.zeros(iterations)
E_pv_negative_flex = np.zeros(iterations)
t_pv_negative_flex = 0.0

time_res = 15/60

E_PV_grid_optimum = np.zeros(iterations)
E_PV_grid_optimum_flex = np.zeros(iterations)

plt.figure(figsize=(12, 6))

positive_flex = 0.0

negative_flex = 0.0

for i in range(iterations):

   pv_pos_flex_iteration_list = []

   pv_neg_flex_iteration_list = []


   def PV_positive_flex():  # curtailing pv feed in to grid

       P_pv_positive_flex[i] = -abs(P_PV_export_grid_opt[i])

       t_pv_positive_flex = time_res

       E_pv_positive_flex[i] = P_pv_positive_flex[i] * t_pv_positive_flex

       print(t_pv_positive_flex)

       pv_pos_flex_iteration_list.append(i)

       for j in range(i + 1, i + 8, 1):

           if P_PV_export_grid_opt[j] >= P_pv_positive_flex[i]:
               P_pv_positive_flex[j] = P_pv_positive_flex[i]
               E_pv_positive_flex[j] = P_pv_positive_flex[i] * t_pv_positive_flex
               pv_pos_flex_iteration_list.append(j)

       return P_pv_positive_flex[i]


   def PV_negative_flex():  # not charging battery from PV

       P_pv_negative_flex[i] = -P_bat_charging_PV_opt[i]

       t_pv_negative_flex = time_res

       E_pv_negative_flex[i] = P_pv_negative_flex[i] * t_pv_negative_flex

       pv_neg_flex_iteration_list.append(i)

       for j in range(i + 1, iterations - 1, 1):

           if abs(P_bat_charging_PV_opt[j] > abs(P_pv_negative_flex[i])):
               P_pv_negative_flex[j] = P_pv_negative_flex[i]
               E_pv_negative_flex[j] = P_pv_negative_flex[i] * t_pv_negative_flex
               pv_neg_flex_iteration_list.append(j)

       return P_pv_negative_flex[i]

   if i <= 85:


    E_PV_grid_optimum[i] += (P_PV_export_grid_opt[i]) * time_res / 1000

    E_PV_grid_optimum_flex[i] += (P_PV_export_grid_opt_flex[i]) * time_res / 1000

    positive_flex = PV_positive_flex()

    negative_flex = PV_negative_flex()

    print(pv_pos_flex_iteration_list)

    print(pv_neg_flex_iteration_list)

    E_positive_flex = np.zeros(len(pv_pos_flex_iteration_list))

    E_negative_flex = np.zeros(len(pv_neg_flex_iteration_list))

    E_positive_flex[0] = E_PV_grid_optimum[i]

    for j in (np.arange(1, len(pv_pos_flex_iteration_list), 1)):
        E_positive_flex[j] = E_positive_flex[j - 1] - P_pv_positive_flex[i] * time_res / 1000

    #time_pos_flex_vector = np.arange(i, i + len(bat_pos_flex_iteration_list), 1)

    # Original time array with 15-minute resolution
    original_time_array = np.arange(i, i + len(pv_pos_flex_iteration_list), 1) * 15  # minutes

    # Convert to datetime objects
    time_pos_flex_vector = [datetime(2019, 7, 25, 0, 0) + timedelta(minutes=int(time)) for time in original_time_array]

    print(time_pos_flex_vector)

    print(E_positive_flex)

    plt.plot(time_pos_flex_vector, E_positive_flex, color='blue')

    E_negative_flex[0] = E_PV_grid_optimum[i]

    for k in (np.arange(1, len(pv_neg_flex_iteration_list), 1)):
        E_negative_flex[k] = E_negative_flex[k - 1] + P_pv_negative_flex[i] * time_res / 1000

    #time_neg_flex_vector = np.arange(i, i + len(bat_neg_flex_iteration_list), 1)

    # Original time array with 15-minute resolution
    original_time_array = np.arange(i, i + len(pv_neg_flex_iteration_list), 1) * 15  # minutes

    # Convert to datetime objects
    time_neg_flex_vector = [datetime(2019, 7, 25, 0, 0) + timedelta(minutes=int(time)) for time in original_time_array]

    print(time_neg_flex_vector)

    print(E_negative_flex)

    plt.plot(time_neg_flex_vector, E_negative_flex, color='green')

# code to plot the optimum battery-grid exchanged energy with flexibility


#change of axes
# Original time array with 15-minute resolution
original_time_array = np.arange(0, 96, 1) * 15  # minutes

# Convert to datetime objects
time_objects = [datetime(2019, 7, 25, 0, 0) + timedelta(minutes=int(time)) for time in original_time_array]

# Convert to hours with 3-hour resolution
new_time_array = np.arange(0, 96, 12)  # 3 hours * 60 minutes / 15 minutes

plt.plot(time_objects, E_PV_grid_optimum, marker='*', label= 'PV-Grid-Exchanged-Energy-Optimum')
plt.plot(time_objects, E_PV_grid_optimum_flex, marker='.', label='PV-Grid-Exchanged-Energy-Flex-Optimum')

# Set ticks and labels for the x-axis
plt.xticks([time_objects[i] for i in new_time_array], [f'{i/4:.0f}:00' for i in new_time_array], rotation=45)

plt.title('Comparison of Exchanged Energy in Basic and Flexible Optimization')
plt.xlabel('Day Time Scale with 15 min time resolution')
plt.ylabel('Cummulative Exchanged Energy in kWh')

plt.tight_layout()
plt.legend()
plt.show()