import gurobipy as gp

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import Scenario_Generation_and_reduction_Stochastic_OPT_PV
import Scenario_Generation_and_reduction_Stochastic_OPT_Load
import Flexibility_Prosumer_PV_Battery_Grid_Stochastic_2stage_OPT_Basic
from numpy import random
import pandas as pd

time_steps = np.arange(0, 96, 1)

Flex_Basic_Optimization = Flexibility_Prosumer_PV_Battery_Grid_Stochastic_2stage_OPT_Basic.EMS_Basic_Optimization()

iterations = len(time_steps)

C_bat = 4600 # in Wh
Panel_area = 4  # 2msq 2 panels area x 2 panels (1.937 sq m from a article : (https://www.skillstg.co.uk/blog/solar-photovoltaic-panel-sizes/#:~:text=The%20physical%20dimensions%20of%20most,thickness%20of%20around%201.5%20inches)
counter = 0
SOC_min = 0.05
SOC_max = 0.95 # soc limits from the research paper (https://www.researchgate.net/publication/373737720_A_PROGNOSTIC_DECISION-MAKING_APPROACH_UNDER_UNCERTAINTY_FOR_AN_ELECTRIC_VEHICLE_FLEET_ROUTING_PROBLEM)
E_bat_min = SOC_min * C_bat
E_bat_max = SOC_max * C_bat
P_bat_min = 0 # 0.01kW # Assumption to make P_bat_min equal to zero for simplicity reasons
P_bat_max = 1250 # 1.5kW
P_bat_delta = P_bat_max - P_bat_min
E_bat_initial = (E_bat_max + E_bat_min)/2
E_bat_final = E_bat_max
eta_batt = 0.93 # taken 0.93 in research paper
P_grid_max = 2500
P_grid_min = -2500 # GRID LIMITS IN -1.5 kW AND 1.5 kW
P_grid_initial = (P_grid_min + P_grid_max)/2
P_grid_delta = P_grid_max - P_grid_min
P_grid = np.zeros(iterations)# Grid is stabilised at 0 kW power


PV_forecaster = Scenario_Generation_and_reduction_Stochastic_OPT_PV.Scenario_Generatiom_Reduction_PV_Quantiles()

Load_forecaster = Scenario_Generation_and_reduction_Stochastic_OPT_Load.Scenario_Generatiom_Reduction_Load_Quantiles()

load_scaling_factor = 1 # https://www.cleanenergywire.org/factsheets/germanys-energy-consumption-and-power-mix-charts-No need of scaling factor (discussed with Simon)

EM_Price = pd.read_excel('Electricity_Market_Price.xlsx', sheet_name= 'Electricity_Market_Price_July')['Preis (EUR/MWh, EUR/tCO2)'] # for the year 2015 - LINK : https://ember-climate.org/data-catalogue/european-wholesale-electricity-price-data/

#changing time resolution of energy market price from 1 hour to 15 min

EM_Price_15_min_res = np.zeros(iterations)

for i in range(24):
     for j in range(4):
         EM_Price_15_min_res[i*4 + j] = EM_Price[i] # for the year 2015 - LINK : https://ember-climate.org/data-catalogue/european-wholesale-electricity-price-data/

 #changing time resolution of energy market price from 1 hour to 5 min

Load_scenarios = Load_forecaster.df_reduced_scenarios

Load_scenarios_probabilities = Load_forecaster.df_reduced_scenarios_probabilities

average_Load_probabilities = Load_forecaster.average_probabilities_scenarios

print(average_Load_probabilities)

PV_scenarios = PV_forecaster.df_reduced_scenarios

PV_scenarios_probabilities = PV_forecaster.df_reduced_scenarios_probabilities

average_PV_probabilities = PV_forecaster.average_probabilities_scenarios

print(average_PV_probabilities)

number_of_scenarios_PV = PV_scenarios.shape[1] # 9 SCENARIOS

number_of_scenarios_Load = Load_scenarios.shape[1]

number_of_scenarios = number_of_scenarios_PV*number_of_scenarios_Load

epex_spot_import_price = np.zeros(iterations)

epex_spot_export_price = np.zeros(iterations)

average_stochastic_probabilities = [0] * number_of_scenarios

print(number_of_scenarios)

P_grid_scenario = np.zeros(number_of_scenarios)


offset = np.ones(iterations)*5.0 # offset is 5 EUR/MWh


bat_operational_cost = 0.01 # 1 cent/NO_OF_CHARGING/DISCHARGING_CYCLES


print("Flex Loop Begins...")

P_bat_positive_flex = np.zeros(iterations)
E_bat_positive_flex = np.zeros(iterations)
t_bat_positive_flex = 0.0

P_bat_negative_flex = np.zeros(iterations)
E_bat_negative_flex = np.zeros(iterations)
t_bat_negative_flex = 0.0

P_pv_positive_flex = np.zeros(iterations)
E_pv_positive_flex = np.zeros(iterations)
t_pv_positive_flex = 0.0

P_pv_negative_flex = np.zeros(iterations)
E_pv_negative_flex = np.zeros(iterations)
t_pv_negative_flex = 0.0

P_flex = np.zeros(iterations)
E_flex = np.zeros(iterations)
E_bat_flex_check = np.zeros(iterations + 1)
E_bat_flex = np.zeros(iterations)
P_grid_flex = np.zeros(iterations)
Price_flex = np.zeros(iterations)
P_grid_current_flex = np.ones(iterations + 1) * P_grid_initial

E_bat_loop_current_flex = np.zeros(iterations + 1)

E_bat_loop_current_flex[0] = E_bat_initial
E_bat_profile_flex = np.zeros(iterations)

E_bat_profile_flex[0] = 0

P_PV_export_grid_profile_flex = np.zeros(iterations)
P_grid_import_profile_flex = np.zeros(iterations)
P_grid_profile_flex = np.zeros(iterations)

epex_spot_import_price_flex = np.zeros(iterations)
epex_spot_export_price_flex = np.zeros(iterations)

P_PV_export_grid_opt_flex = np.zeros(iterations)

P_bat_charging_PV_opt_flex = np.zeros(iterations)

P_bat_charging_grid_opt_flex = np.zeros(iterations)

P_bat_discharging_grid_opt_flex = np.zeros(iterations)

P_bat_discharging_load_opt_flex = np.zeros(iterations)

E_bat_discharging_load_flex = np.zeros(iterations)

P_bat_charging_opt_flex = np.zeros(iterations)

P_bat_discharging_opt_flex = np.zeros(iterations)

P_grid_import_opt_flex = np.zeros(iterations)

P_bat_export_opt_flex = np.zeros(iterations)

 # calling variables from basic optimization class object

P_PV_export_grid_opt = Flex_Basic_Optimization.P_PV_export_grid_profile

P_bat_charging_PV_opt = Flex_Basic_Optimization.P_bat_charging_PV_opt

P_bat_charging_grid_opt = Flex_Basic_Optimization.P_bat_charging_grid_opt

P_bat_charging_opt = Flex_Basic_Optimization.P_bat_charging_opt

P_bat_discharging_grid_opt = Flex_Basic_Optimization.P_bat_discharging_grid_opt

P_bat_discharging_load_opt = Flex_Basic_Optimization.P_bat_discharging_load_opt

P_bat_discharging_opt = Flex_Basic_Optimization.P_bat_discharging_opt

P_grid_import_opt = Flex_Basic_Optimization.P_grid_import_profile


epex_spot_price = Flex_Basic_Optimization.EM_Price

battery_cycles_flex_iteration_count = np.zeros(iterations)

Bat_Positive_Flex_Price = np.zeros(iterations)

Bat_Negative_Flex_Price = np.zeros(iterations)

PV_Positive_Flex_Price = np.zeros(iterations)

PV_Negative_Flex_Price = np.zeros(iterations)

bat_pos_flex_iteration_list = []

bat_neg_flex_iteration_list = []

pv_pos_flex_iteration_list = []

pv_neg_flex_iteration_list = []

i_count_flex = 0

counter_non_optimal = 0

counter_bat_positive_flex = 0

counter_PV_positive_flex = 0
counter_PV_negative_flex = 0


E_bat_charging_profile = np.zeros(iterations)
E_bat_charging_scenario_profile = np.zeros(number_of_scenarios)

P_grid_scenario_profile = np.zeros(number_of_scenarios)

E_bat_scenario_profile = np.zeros(number_of_scenarios)

E_bat_discharging_profile = np.zeros(iterations)
E_bat_discharging_scenario_profile = np.zeros(number_of_scenarios)

E_bat_loop_current = np.zeros(iterations+1)

E_bat_profile = np.zeros(iterations+1)

P_grid_profile = np.zeros(iterations+1)

E_bat_loop_current_scenario = np.zeros(number_of_scenarios)

P_bat_charging_PV_scenario_opt = np.zeros(number_of_scenarios)

P_bat_charging_grid_scenario_opt = np.zeros(number_of_scenarios)

P_bat_discharging_grid_scenario_opt = np.zeros(number_of_scenarios)

P_bat_discharging_load_scenario_opt = np.zeros(number_of_scenarios)

P_bat_charging_scenario_opt = np.zeros(number_of_scenarios)

P_bat_discharging_scenario_opt = np.zeros(number_of_scenarios)

P_bat_export_scenario_opt = np.zeros(number_of_scenarios)

P_grid_import_scenario_profile = np.zeros(number_of_scenarios)

P_PV_export_grid_scenario_profile = np.zeros(number_of_scenarios)

time_res = (15/60) # 15 minutes in hours

non_optimal_i_j_values = np.zeros(iterations*number_of_scenarios)

p = 0

switching_operation_battery = [0, 1]

sum_cost_prosumer = 0.0


for i in range(iterations):

   counter_bat_negative_flex = 0

   bat_neg_flex = 0

   pv_pos_flex = 0

   pv_neg_flex = 0

   bat_pos_flex = 0

     # sign convention - charging/feed-in to grid curtailed or charging from grid -ve; discharging to grid +ve

   def bat_positive_flex():  # battery charge from grid

         P_bat_positive_flex[i] = -(P_bat_max - abs(
             P_bat_charging_grid_opt[i]))  # calling variables from object of basic optimization class

         t_bat_positive_flex = time_res

         E_bat_positive_flex[i] = P_bat_positive_flex[i] * t_bat_positive_flex

         bat_pos_flex_iteration_list.append(i)

         E_bat_flex_check[i + 1] = E_bat_loop_current_flex[i] - P_bat_positive_flex[i] * t_bat_positive_flex

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

         E_bat_flex_check[i + 1] = E_bat_loop_current_flex[i] - P_bat_negative_flex[i] * t_bat_negative_flex

         #if (E_bat_final - E_bat_flex_check[i + 1]) <= E_bat_max * (len(time_steps) - 1 - (i + 1)):

         for k in range(i + 1, i + 9, 1): # Bat Flex check for 1 hour from the called timestep

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

   def PV_positive_flex():  # curtailing pv feed in to grid

         P_pv_positive_flex[i] = -abs(P_PV_export_grid_opt[i])

         t_pv_positive_flex = time_res

         E_pv_positive_flex[i] = P_pv_positive_flex[i] * t_pv_positive_flex

         print(t_pv_positive_flex)

         pv_pos_flex_iteration_list.append(i)

         for j in range(i + 1, iterations - 1, 1):

             if abs(P_PV_export_grid_opt[j]) >= abs(P_pv_positive_flex[i]):
                 E_pv_positive_flex[k] = P_pv_positive_flex[i] * t_pv_positive_flex
                 pv_pos_flex_iteration_list.append(j)

         return P_pv_positive_flex[i]

   def PV_negative_flex():  # not charging battery from PV

         P_pv_negative_flex[i] = -P_bat_charging_PV_opt[i]

         t_pv_negative_flex = time_res

         E_pv_negative_flex[i] = P_pv_negative_flex[i] * t_pv_negative_flex

         pv_neg_flex_iteration_list.append(i)

         for j in range(i + 1, iterations - 1, 1):

             if abs(P_bat_charging_PV_opt[j] > abs(P_pv_negative_flex[i])):
                 E_pv_negative_flex[k] = P_pv_negative_flex[i] * t_pv_negative_flex
                 pv_neg_flex_iteration_list.append(j)

         return P_pv_negative_flex[i]

   if i == 12 :  # 3:15 AM Battery +ve  Flexibility Check

        bat_pos_flex = bat_positive_flex()

        counter_bat_positive_flex = 1

         # pv_neg_flex = PV_negative_flex()

         # counter_PV_negative_flex = 1

         # bat_pos_flex = bat_positive_flex()

         # counter_bat_positive_flex = 1

         # switch to 1 or 0

   print("Bat_Pos_Flex", bat_pos_flex)

   print("Bat_Neg_Flex", bat_neg_flex)

   print("PV_pos_flex", pv_pos_flex)

   print("PV_neg_flex", pv_neg_flex)

   P_flex[i] = pv_neg_flex + pv_pos_flex + bat_neg_flex + bat_pos_flex

   epex_spot_import_price[i] = (EM_Price_15_min_res[i] + offset[i])

   epex_spot_export_price[i] = (EM_Price_15_min_res[i] - offset[i])

   l = 0

   for j in range(number_of_scenarios_PV):

    for k in range(number_of_scenarios_Load):



      model_flex = gp.Model("Two_Stage_Stochastic_Optimization_EMS")

      E_bat_loop_current_scenario[l] = E_bat_loop_current_flex[i]

      P_grid_scenario[l] = P_grid_current_flex[i]

      # Define decision variables for the first stage

      P_PV_export_grid_flex = model_flex.addVar(lb=0, vtype=gp.GRB.CONTINUOUS,
                                                name="P_PV_export_grid_flex")

      P_bat_charging_PV_flex = model_flex.addVar(lb=0, vtype=gp.GRB.CONTINUOUS,
                                                 name="P_bat_charging_PV_flex")

      P_bat_charging_grid_flex = model_flex.addVar(lb=0, vtype=gp.GRB.CONTINUOUS,
                                                   name="P_bat_charging_grid_flex")

      P_bat_discharging_grid_flex = model_flex.addVar(lb=0, vtype=gp.GRB.CONTINUOUS,
                                                      name="P_bat_discharging_grid_flex")

      P_bat_discharging_load_flex = model_flex.addVar(lb=0, vtype=gp.GRB.CONTINUOUS,
                                                      name="P_bat_discharging_load_flex")

      P_bat_export = model_flex.addVar(lb=0, vtype=gp.GRB.CONTINUOUS,
                                       name="P_bat_export")

      P_grid_import_flex = model_flex.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="P_grid_import_flex")

      PV_quantile = PV_scenarios.iloc[i, j] * Panel_area

      Load_quantile = Load_scenarios.iloc[i, k]

      if PV_quantile < 0:
          PV_quantile = 0

      if i % 2 == 0:
          binary_battery_variable = switching_operation_battery[0]
      else:
          binary_battery_variable = switching_operation_battery[1]

      # binary_battery_variable = random.randint(0,1)



          # bat_pos_flex = bat_positive_flex() # switch to 1 or 0

          # if i == 215: # 6 PM of 8th May :- Battery Flexibility Check

          # bat_neg_flex = bat_negative_flex()  # switch to 1 or 0

          # counter_bat_negative_flex = 1



      print(binary_battery_variable)

      import_price_flex_costs = -(P_bat_positive_flex[i]) * time_res * 12
      export_price_flex_savings = (P_bat_negative_flex[i] + P_pv_positive_flex[i] - P_pv_negative_flex[
              i]) * time_res * 8   # import and export prices have a deviation of 2 EUR/MWh

      epex_spot_import_price_flex[i] = epex_spot_import_price[i]
      epex_spot_export_price_flex[i] = epex_spot_export_price[i]

      P_bat_charging_flex = (P_bat_charging_PV_flex + P_pv_negative_flex[i] - P_pv_positive_flex[i]) + (
                  P_bat_charging_grid_flex - P_bat_positive_flex[i] - P_bat_export)

      P_bat_discharging_flex = (P_bat_discharging_load_flex + P_bat_discharging_grid_flex  + P_bat_negative_flex[i])


      # OPTIMIZATION WITH THE FLEX VALUES

      # Set objective function for the first stage

      #battery_cycles = (P_bat_charging_flex + P_bat_discharging_flex) / P_bat_delta

      OBJECTIVE_FUNCTION =((P_grid_import_flex) * epex_spot_import_price_flex[i] - (P_PV_export_grid_flex) * epex_spot_export_price_flex[
                      i]) * time_res

      model_flex.setObjective(OBJECTIVE_FUNCTION, gp.GRB.MINIMIZE)

      # Battery max and min power constraints with alternate charging or discharging events (1ST STAGE CONSTRAINTS)

      if binary_battery_variable == 1:

          model_flex.addConstr(P_bat_discharging_flex == 0) # R0
          model_flex.addConstr(P_bat_charging_flex >= P_bat_min) #R1
          model_flex.addConstr(P_bat_charging_flex <= P_bat_max)#R2
          model_flex.addConstr(P_bat_charging_PV_flex == 0.25 * PV_quantile)

      else:

          model_flex.addConstr(P_bat_charging_flex == 0)
          model_flex.addConstr(P_bat_discharging_flex >= P_bat_min)
          model_flex.addConstr(P_bat_discharging_flex <= P_bat_max)

      # 0 for battery charging

      P_grid_delta_net = P_grid_scenario[l]  + P_PV_export_grid_flex - P_grid_import_flex + P_bat_discharging_grid_flex - P_bat_charging_grid_flex + (
                                          P_bat_negative_flex[i] + P_pv_positive_flex[i] - P_pv_negative_flex[i] +
                                          P_bat_positive_flex[i])

      E_bat_loop = E_bat_loop_current_scenario[l] + (P_bat_charging_flex * eta_batt - P_bat_discharging_flex / eta_batt) * (
          time_res)  # time_res is 15 min

      # 4

      #model_flex.addConstr(P_grid_delta_net <= P_grid_max)
      #model_flex.addConstr(P_grid_delta_net >= P_grid_min)
      model_flex.addConstr(E_bat_loop >= E_bat_min)  # 3
      model_flex.addConstr(E_bat_loop <= E_bat_max)  # 4

      model_flex.addConstr(P_bat_export <= abs(P_bat_positive_flex[i])) #5

      average_stochastic_probabilities[l] = (average_PV_probabilities[j] * average_Load_probabilities[k])

      print(average_stochastic_probabilities[l])

      # Integrate 2ND STAGE subproblems into the main model

      model_flex.addConstr(
          ((P_bat_discharging_flex + P_grid_import_flex) - (P_bat_charging_flex + P_PV_export_grid_flex)) == (Load_quantile - PV_quantile))

      model_flex.addConstr(P_PV_export_grid_flex + P_bat_charging_PV_flex  <= PV_quantile)



      #model_flex.addConstr(P_PV_export_grid_flex <= 0.75 * PV_quantile)


      print(i)

      model_flex.optimize()

      if model_flex.status == gp.GRB.OPTIMAL:
        counter += 1
        print("Optimal solution found")
        print("Optimal objective value =", model_flex.objVal)
        sum_cost_prosumer += model_flex.objVal*average_stochastic_probabilities[l]

        P_grid_scenario_profile[l] = P_grid_scenario[l] - P_grid_import_flex.x + P_PV_export_grid_flex.x + P_bat_discharging_grid_flex.x - P_bat_charging_grid_flex.x + (
                                    P_bat_negative_flex[i] + P_pv_positive_flex[i] - P_pv_negative_flex[i] +
                                    P_bat_positive_flex[i])

        E_bat_scenario_profile[l] = E_bat_loop_current_scenario[l] + (((P_bat_charging_PV_flex.x + P_pv_negative_flex[i]  + P_bat_charging_grid_flex.x - P_bat_positive_flex[i] - P_bat_export.x) * eta_batt - (
                                                                               P_bat_discharging_load_flex.x + P_bat_discharging_grid_flex.x + P_bat_negative_flex[i] ) / eta_batt))*time_res
        print(E_bat_scenario_profile[l])

        P_PV_export_grid_scenario_profile[l] = -abs(
            P_PV_export_grid_flex.x + P_pv_positive_flex[i] - P_pv_negative_flex[i])
        P_bat_charging_PV_scenario_opt[l] = P_bat_charging_PV_flex.x + P_pv_negative_flex[i]
        P_bat_charging_grid_scenario_opt[l] = P_bat_charging_grid_flex.x - P_bat_positive_flex[i]
        P_bat_discharging_grid_scenario_opt[l] = -(P_bat_discharging_grid_flex.x + P_bat_negative_flex[i])
        P_bat_discharging_load_scenario_opt[l] = -P_bat_discharging_load_flex.x
        P_bat_export_scenario_opt[l] = P_bat_export.x
        P_grid_import_scenario_profile[l] = P_grid_import_flex.x
        P_bat_charging_scenario_opt[l] = P_bat_charging_PV_flex.x + P_pv_negative_flex[i] + P_bat_charging_grid_flex.x - P_bat_positive_flex[i] - P_pv_positive_flex[i]
        P_bat_discharging_scenario_opt[l] = -P_bat_discharging_grid_flex.x - P_bat_discharging_load_flex.x - P_bat_negative_flex[i]

        P_grid_profile_flex[i] += P_grid_scenario_profile[l] * average_stochastic_probabilities[l]

        E_bat_profile_flex[i] += E_bat_scenario_profile[l] * average_stochastic_probabilities[l]



        P_PV_export_grid_profile_flex[i] += P_PV_export_grid_scenario_profile[l] * average_stochastic_probabilities[l]
        P_bat_charging_PV_opt_flex[i] += P_bat_charging_PV_scenario_opt[l] * average_stochastic_probabilities[l]
        P_bat_charging_grid_opt_flex[i] += P_bat_charging_grid_scenario_opt[l] * average_stochastic_probabilities[l]
        P_bat_discharging_grid_opt_flex[i] += P_bat_discharging_grid_scenario_opt[l] * average_stochastic_probabilities[l]
        P_bat_discharging_load_opt_flex[i] += P_bat_discharging_load_scenario_opt[l] * average_stochastic_probabilities[l]
        P_bat_export_opt_flex[i] += P_bat_export_scenario_opt[l]*average_stochastic_probabilities[l]
        P_grid_import_profile_flex[i] += P_grid_import_scenario_profile[l] * average_stochastic_probabilities[l]
        P_bat_charging_opt_flex[i] += P_bat_charging_scenario_opt[l]*average_stochastic_probabilities[l]
        P_bat_discharging_opt_flex[i] += P_bat_discharging_scenario_opt[l]*average_stochastic_probabilities[l]

      else:
        print("No solution found")
        model_flex.computeIIS()  # Compute the Infeasible Inconsistency Set (IIS)
        iis_constraints = [constr for constr in model_flex.getConstrs() if constr.IISConstr]
        print("Infeasible constraints:", iis_constraints)
        counter_non_optimal += 1
        P_grid_scenario_profile[l] = P_grid_scenario[l]
        E_bat_scenario_profile[l] = E_bat_loop_current_scenario[l]
        P_grid_profile[i] += P_grid_scenario_profile[l] * average_stochastic_probabilities[l]

        E_bat_profile[i] += E_bat_scenario_profile[l] * average_stochastic_probabilities[l]
        non_optimal_i_j_values[p] = i
        p += 1

      l += 1


   print(E_bat_profile_flex[i])
   E_bat_loop_current_flex[i + 1] = E_bat_profile_flex[i]





print(counter)

print(counter_non_optimal)

print(P_bat_charging_opt_flex)

print(P_bat_discharging_opt_flex)

print(P_grid_import_profile_flex)

print(P_PV_export_grid_profile_flex)

start_time = datetime.strptime('00:00', '%H:%M')

print(start_time)

date_rng = pd.date_range(start='2019-07-25', end='2019-07-26', freq='15T')

data = np.random.randn(len(date_rng))

 # Create a DataFrame with the time series data
df = pd.DataFrame(data, index=date_rng, columns=['Value'])

df = df.drop(df.tail(1).index)

 # Resample the data to 3-hour resolution
df_resampled = df.resample('3H').mean()

 # Plot the original and resampled time series
plt.figure(figsize=(12, 6))

#time_scale = [start_time + timedelta(minutes=15 * i) for i in range(len(time_steps))]


plt.plot(df.index, P_bat_charging_opt_flex, marker='*', label='Bat_charging_profile')
plt.plot(df.index, P_bat_charging_grid_opt_flex, marker='3', label='Bat_charging_grid_profile')
plt.plot(df.index, P_bat_charging_PV_opt_flex, marker='4', label='Bat_charging_PV_profile' )
plt.plot(df.index, P_bat_discharging_opt_flex, marker='+', label='Bat_discharging_profile')
plt.plot(df.index, P_bat_discharging_load_opt_flex, marker='1', label='Bat_discharging_load_profile')
plt.plot(df.index, P_bat_discharging_grid_opt_flex, marker='2', label='Bat_discharging_grid_profile')
plt.plot(df.index, P_bat_export_opt_flex, marker='.', label='P_bat_export_optimum_flow')
plt.plot(df.index, P_grid_import_profile_flex, marker='v', label='P_grid_import_optimum_flow')
plt.plot(df.index, P_PV_export_grid_profile_flex, marker='o', label='PV_export_grid_optimum_flow')

plt.title('EMS Stochastic Optimization with Battery Positive flexibility provision')
plt.xlabel('Time')
plt.ylabel('Power Flow in W')

 # Customize the x-axis ticks to show a 3-hour gap
xtick_positions = df_resampled.index
xtick_labels = [str(ts) for ts in xtick_positions]
plt.xticks(xtick_positions, xtick_labels, rotation=45, ha='right')

plt.tight_layout()
plt.legend()
plt.show()

# FLEX PRICES - positive price is income for prosumer and negative price is loss for prosumer

for i in range(iterations):

 RM = 0.15

 cost_per_cycle = 0.01  # 5  cents is the cost per battery charging/discharging cycle

 n_cycles = (P_bat_discharging_opt_flex[i] + P_bat_charging_opt_flex[i])/P_bat_delta  # to be found


    ### VVVVIIII - CHANGE OPT_FLEX to Variables to club prices in objective function
 def Bat_Pos_Flex_Price():

        C_bat_op_pos_flex = 0.5 * cost_per_cycle / n_cycles * (
                P_bat_charging_opt_flex[i] / P_bat_max)  # battery cycles flex to be replaced

        savings_bat_pos_flex = -abs((P_bat_discharging_load_opt_flex[i] - P_bat_discharging_load_opt[i]) * epex_spot_export_price_flex[
            i] + (-P_PV_export_grid_opt[i] + P_PV_export_grid_opt_flex[i]) * epex_spot_export_price_flex[i] + (
                                        - P_bat_charging_grid_opt[i] + P_bat_charging_grid_opt_flex[i]) *
                                epex_spot_import_price_flex[i]) * time_res

        price_flex_bat_pos = (savings_bat_pos_flex * (1 - RM) + C_bat_op_pos_flex) / abs(
            E_bat_positive_flex[i])

        return price_flex_bat_pos


 def Bat_Neg_Flex_Price():

        C_bat_op_neg_flex = 0.5 * cost_per_cycle / n_cycles * (
                abs(P_bat_discharging_opt_flex[i] + P_bat_export_opt_flex[i]))/ P_bat_max

        expenses = abs((P_grid_import_opt_flex[i] + P_bat_charging_grid_opt_flex[i] - P_grid_import_opt[i] - P_bat_charging_grid_opt[i]) * epex_spot_import_price_flex[i] +
                    (P_bat_charging_PV_opt_flex[i] - P_bat_charging_PV_opt[i]) * epex_spot_export_price_flex[i]) * time_res

        price_flex_bat_neg = (expenses * (1 + RM) + C_bat_op_neg_flex) / (E_bat_negative_flex[i])

        return price_flex_bat_neg


 def PV_Pos_Flex_Price():

        profit = (P_bat_charging_PV_opt_flex[i] - P_bat_charging_PV_opt[i]) * epex_spot_import_price_flex[
            i] * time_res  # Battery charging from PV at grid import price

        losses = (abs(P_PV_export_grid_opt[i]) - abs(P_PV_export_grid_opt_flex[i])) * epex_spot_export_price_flex[i] * time_res

        if losses > profit:
            price_flex_pv_pos = ((losses * (1 + RM) - profit)) / E_pv_positive_flex[i]

        elif losses == profit:
            price_flex_pv_pos = (losses * RM) / E_pv_positive_flex[i]

        else:
            price_flex_pv_pos = (profit - losses * (1 + RM)) / E_pv_positive_flex[i]

        return price_flex_pv_pos


 def PV_Neg_Flex_Price():

        savings_pv_neg_flex = (P_bat_charging_PV_opt[i] - P_bat_charging_PV_opt_flex[i]) * epex_spot_export_price_flex[
            i]*time_res

        price_flex_pv_neg = savings_pv_neg_flex * (1 + RM) / (E_pv_negative_flex[i])

        return price_flex_pv_neg


 for k in range(len(bat_pos_flex_iteration_list)):

    if i == bat_pos_flex_iteration_list[k]:

      Bat_Positive_Flex_Price[i] = Bat_Pos_Flex_Price()



 Price_flex[i] =  PV_Negative_Flex_Price[i] + PV_Positive_Flex_Price[i] + \
                    Bat_Negative_Flex_Price[i] + Bat_Positive_Flex_Price[i] + EM_Price_15_min_res[i]

    # for k in range(len(pv_neg_flex_iteration_list)):

    # if pv_pos_flex_iteration_list[k] == i:
    # PV_Negative_Flex_Price[i] = PV_Neg_Flex_Price()


print(bat_pos_flex_iteration_list)

print(P_bat_positive_flex)

print(E_bat_positive_flex)

# Price Flex Calculation

print("Battery Positive Flex Price is", Bat_Positive_Flex_Price)

print("Battery Negative Flex Price is", Bat_Negative_Flex_Price)

print("PV Positive Flex Price is", PV_Positive_Flex_Price)

print("PV Negative Flex Price is", PV_Negative_Flex_Price)

print("Flex Price is", Price_flex)

print(non_optimal_i_j_values)

print(sum_cost_prosumer)























