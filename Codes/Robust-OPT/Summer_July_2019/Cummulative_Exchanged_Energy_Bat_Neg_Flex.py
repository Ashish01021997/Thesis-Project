import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
#import Flexibility_Prosumer_PV_Battery_Grid_Stochastic_2stage_OPT_Battery_Neg_Flex
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

Bat_Neg_Flex = Flexibility_Prosumer_PV_Battery_Grid_Stochastic_2stage_OPT_Battery_Neg_Flex.EMS_OPT_Bat_Neg_Flex()

P_bat_charging_grid_opt = Bat_Neg_Flex.P_bat_charging_grid_opt

P_bat_discharging_grid_opt = Bat_Neg_Flex.P_bat_discharging_grid_opt

P_bat_discharging_load_opt = Bat_Neg_Flex.P_bat_discharging_load_opt

P_bat_discharging_opt = Bat_Neg_Flex.P_bat_discharging_opt

P_bat_charging_grid_opt_flex = Bat_Neg_Flex.P_bat_charging_grid_opt_flex

P_bat_discharging_grid_opt_flex = Bat_Neg_Flex.P_bat_discharging_grid_opt_flex

E_bat_profile = Bat_Neg_Flex.E_bat_profile

E_bat_final = E_bat_initial

E_bat_flex_check = np.zeros(iterations + 1)

P_bat_positive_flex = np.zeros(iterations)
E_bat_positive_flex = np.zeros(iterations)
t_bat_positive_flex = 0.0

P_bat_negative_flex = np.zeros(iterations)
E_bat_negative_flex = np.zeros(iterations)
t_bat_negative_flex = 0.0

time_res = 15/60

E_battery_grid_optimum = np.zeros(iterations)
E_battery_grid_optimum_flex = np.zeros(iterations)

plt.figure(figsize=(12, 6))

positive_flex = 0.0

negative_flex = 0.0

for i in range(iterations):

   bat_pos_flex_iteration_list = []

   bat_neg_flex_iteration_list = []

   def bat_positive_flex():  # battery charge from grid



         P_bat_positive_flex[i] = -(P_bat_max - abs(
             P_bat_charging_grid_opt[i]))  # calling variables from object of basic optimization class

         t_bat_positive_flex = time_res

         E_bat_positive_flex[i] = P_bat_positive_flex[i] * t_bat_positive_flex

         bat_pos_flex_iteration_list.append(i)

         E_bat_flex_check[i + 1] = E_bat_profile[i] - P_bat_positive_flex[i] * t_bat_positive_flex

         if (E_bat_flex_check[i + 1] - E_bat_final) <= sum(
                 abs(P_bat_discharging_load_opt[j]) * time_res for j in range(i + 1, iterations - 1, 1)):

            for k in range(i + 1, i + 8, 1):

                 if abs(P_bat_max - abs(
             P_bat_charging_grid_opt[k])) >= abs(P_bat_max - abs(
             P_bat_charging_grid_opt[i])):
                     P_bat_positive_flex[k] = P_bat_positive_flex[i]
                     E_bat_positive_flex[k] = P_bat_positive_flex[i] * t_bat_positive_flex
                     bat_pos_flex_iteration_list.append(k)

         return P_bat_positive_flex[i]



   def bat_negative_flex():  # battery not charging from grid and discharging to grid

       t_bat_negative_flex = time_res

       if P_bat_max - abs(P_bat_discharging_opt[i]) > 0 and abs(P_bat_charging_grid_opt[i]) > 0:

           P_bat_negative_flex[i] = (P_bat_max - abs(P_bat_discharging_opt[i])) + abs(P_bat_charging_grid_opt[i])

           E_bat_negative_flex[i] = P_bat_negative_flex[i] * t_bat_negative_flex

           bat_neg_flex_iteration_list.append(i)

       elif P_bat_max - abs(P_bat_discharging_opt[i]) > 0:

           P_bat_negative_flex[i] = (P_bat_max - abs(P_bat_discharging_opt[i]))

           E_bat_negative_flex[i] = P_bat_negative_flex[i] * t_bat_negative_flex

           bat_neg_flex_iteration_list.append(i)

       elif abs(P_bat_charging_grid_opt[i]) > 0:

           P_bat_negative_flex[i] = abs(P_bat_charging_grid_opt[i])

           E_bat_negative_flex[i] = P_bat_negative_flex[i] * t_bat_negative_flex

           bat_neg_flex_iteration_list.append(i)

       E_bat_flex_check[i + 1] = E_bat_profile[i] - P_bat_negative_flex[i] * t_bat_negative_flex

       if (E_bat_final - E_bat_flex_check[i + 1]) <= E_bat_max * (len(timescale) - 1 - (i + 1)):

             for k in range(i + 1, i + 8, 1): # Bat Flex check for 1 hour from the called timestep

                 if abs(P_bat_charging_grid_opt[k]) >= abs(P_bat_charging_grid_opt[i]) and (
                         P_bat_max - abs(P_bat_discharging_opt[k])) >= (P_bat_max - abs(P_bat_discharging_opt[i])):
                     if ((P_bat_max - abs(P_bat_discharging_opt[i])) + abs(P_bat_charging_grid_opt[i])) > 0:
                         P_bat_negative_flex[k] = (
                                 abs(P_bat_charging_grid_opt[i]) + (P_bat_max - abs(P_bat_discharging_opt[i])))
                         E_bat_negative_flex[k] = (abs(P_bat_charging_grid_opt[i]) + (
                                 P_bat_max - abs(P_bat_discharging_opt[i]))) * t_bat_negative_flex
                         bat_neg_flex_iteration_list.append(k)

                 elif abs(P_bat_charging_grid_opt[k]) >= abs(P_bat_charging_grid_opt[i]):
                     if abs(P_bat_charging_grid_opt[i]) > 0:
                         P_bat_negative_flex[k] = abs(P_bat_charging_grid_opt[i])
                         E_bat_negative_flex[k] = abs(P_bat_charging_grid_opt[i]) * t_bat_negative_flex
                         bat_neg_flex_iteration_list.append(k)

                 elif (P_bat_max - abs(P_bat_discharging_opt[k])) >= (P_bat_max - abs(P_bat_discharging_opt[i])):
                     if (P_bat_max - abs(P_bat_discharging_opt[i])) > 0:
                         P_bat_negative_flex[k] = (P_bat_max - abs(P_bat_discharging_opt[i]))
                         E_bat_negative_flex[k] = (P_bat_max - abs(P_bat_discharging_opt[i])) * t_bat_negative_flex
                         bat_neg_flex_iteration_list.append(k)

       return P_bat_negative_flex[i]

   if i <= 85:


    E_battery_grid_optimum[i] += (abs(P_bat_charging_grid_opt[i]) - abs(
           P_bat_discharging_grid_opt[i])) * time_res / 1000

    E_battery_grid_optimum_flex[i] += (abs(P_bat_charging_grid_opt_flex[i]) - abs(P_bat_discharging_grid_opt_flex[
                                                                                         i])) * time_res / 1000

    positive_flex = bat_positive_flex()

    negative_flex = bat_negative_flex()

    print(bat_pos_flex_iteration_list)

    print(bat_neg_flex_iteration_list)

    E_positive_flex = np.zeros(len(bat_pos_flex_iteration_list))

    E_negative_flex = np.zeros(len(bat_neg_flex_iteration_list))

    E_positive_flex[0] = E_battery_grid_optimum[i]

    for j in (np.arange(1, len(bat_pos_flex_iteration_list), 1)):
        E_positive_flex[j] = E_positive_flex[j - 1] - P_bat_positive_flex[i] * time_res / 1000

    #time_pos_flex_vector = np.arange(i, i + len(bat_pos_flex_iteration_list), 1)

    # Original time array with 15-minute resolution
    original_time_array = np.arange(i, i + len(bat_pos_flex_iteration_list), 1) * 15  # minutes

    # Convert to datetime objects
    time_pos_flex_vector = [datetime(2019, 7, 25, 0, 0) + timedelta(minutes=int(time)) for time in original_time_array]

    print(time_pos_flex_vector)

    print(E_positive_flex)

    plt.plot(time_pos_flex_vector, E_positive_flex, color='blue')

    E_negative_flex[0] = E_battery_grid_optimum[i]

    for k in (np.arange(1, len(bat_neg_flex_iteration_list), 1)):
        E_negative_flex[k] = E_negative_flex[k - 1] - P_bat_negative_flex[i] * time_res / 1000

    #time_neg_flex_vector = np.arange(i, i + len(bat_neg_flex_iteration_list), 1)

    # Original time array with 15-minute resolution
    original_time_array = np.arange(i, i + len(bat_neg_flex_iteration_list), 1) * 15  # minutes

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

plt.plot(time_objects, E_battery_grid_optimum, marker='*', label= 'Battery-Grid-Exchanged-Energy-Optimum')
plt.plot(time_objects, E_battery_grid_optimum_flex, marker='.', label='Battery-Grid-Exchanged-Energy-Flex-Optimum')

# Set ticks and labels for the x-axis
plt.xticks([time_objects[i] for i in new_time_array], [f'{i/4:.0f}:00' for i in new_time_array], rotation=45)

plt.title('Comparison of Exchanged Energy in Basic and Flexible Optimization')
plt.xlabel('Day Time Scale with 15 min time resolution')
plt.ylabel('Cummulative Exchanged Energy in kWh')

plt.tight_layout()
plt.legend()
plt.show()