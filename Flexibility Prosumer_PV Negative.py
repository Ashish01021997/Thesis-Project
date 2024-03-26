import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
import Quantile_Load_forecasts
import Quantile_PV_forecasts
import random as random
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import Flexibility_Prosumer_PV_Battery_Grid_Robust_OPT_Basic

start_time = datetime.strptime('00:00', '%H:%M')

Flex_Basic_Optimization = Flexibility_Prosumer_PV_Battery_Grid_Robust_OPT_Basic.EMS_Basic_Optimization()

C_bat = 4600 # in Wh
Panel_area = 4 # 2msq 1 panel of area (1.937 sq m from a article : (https://www.skillstg.co.uk/blog/solar-photovoltaic-panel-sizes/#:~:text=The%20physical%20dimensions%20of%20most,thickness%20of%20around%201.5%20inches)
counter = 0
SOC_min = 0.20
SOC_max = 0.95 # soc limits from the research paper (https://www.researchgate.net/publication/373737720_A_PROGNOSTIC_DECISION-MAKING_APPROACH_UNDER_UNCERTAINTY_FOR_AN_ELECTRIC_VEHICLE_FLEET_ROUTING_PROBLEM)
E_bat_min = SOC_min * C_bat
E_bat_max = SOC_max * C_bat
P_bat_min = 0 # 0.5kW
P_bat_max = 2500 # 1,25 kW
P_bat_delta = P_bat_max - P_bat_min
E_bat_initial = (E_bat_max + E_bat_min)/2
E_bat_final = E_bat_initial
eta_batt = 0.93
P_grid_min = -2000
P_grid_max = 2000
P_grid_range = P_grid_max - P_grid_min
P_grid_initial = 0

time_scale = np.arange(0, 96, 1) # 15 minutes is the time resolution

iterations = len(time_scale)

PV_forecaster = Quantile_PV_forecasts.PV_Quantiles()

Load_forecaster = Quantile_Load_forecasts.Load_Quantiles()

load_scaling_factor = 1 # https://www.cleanenergywire.org/factsheets/germanys-energy-consumption-and-power-mix-charts - scaled load from 2010 to 2015

# for the year 2015 - LINK : https://ember-climate.org/data-catalogue/european-wholesale-electricity-price-data/

Load_hour_forecasts = Load_forecaster.df_predictions_load

Load_ten_percent_quantiles = Load_hour_forecasts['ten_percent_quantile']*load_scaling_factor

Load_ninety_percent_quantiles = Load_hour_forecasts['ninety_percent_quantile']*load_scaling_factor

PV_hour_forecasts = PV_forecaster.df_predictions_PV

PV_ten_percent_quantiles = PV_hour_forecasts['ten_percent_quantile']*Panel_area

PV_ninety_percent_quantiles = PV_hour_forecasts['ninety_percent_quantile']*Panel_area


offset = np.ones(iterations)*10.0 # offset is 10 EUR/MWh


bat_operational_cost = 0.1248*1000

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
E_bat_flex_check = np.zeros(iterations+1)
E_bat_flex = np.zeros(iterations)
P_grid_flex = np.zeros(iterations)
Price_flex = np.zeros(iterations)


P_grid_current_flex = np.ones(iterations+1) * P_grid_initial

E_bat_loop_current_flex = np.zeros(iterations+1)

E_bat_loop_current_flex[0] = E_bat_initial
E_bat_profile_flex = np.zeros(iterations)
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

P_bat_charging_opt_flex_effective = np.zeros(iterations)

P_bat_discharging_opt_flex_effective = np.zeros(iterations)

P_grid_import_opt_flex = np.zeros(iterations)

P_bat_export_opt = np.zeros(iterations)

# calling variables from basic optimization class object

P_PV_export_grid_opt = Flex_Basic_Optimization.P_PV_export_grid_opt

P_bat_charging_PV_opt = Flex_Basic_Optimization.P_bat_charging_PV_opt

P_bat_charging_grid_opt = Flex_Basic_Optimization.P_bat_charging_grid_opt

P_bat_charging_opt = Flex_Basic_Optimization.P_bat_charging_opt

P_bat_discharging_grid_opt = Flex_Basic_Optimization.P_bat_discharging_grid_opt

P_bat_discharging_load_opt = Flex_Basic_Optimization.P_bat_discharging_load_opt

P_bat_discharging_opt = Flex_Basic_Optimization.P_bat_discharging_opt

P_grid_import_opt = Flex_Basic_Optimization.P_grid_import_opt

E_bat_profile = Flex_Basic_Optimization.E_bat_profile

P_grid_profile = Flex_Basic_Optimization.P_grid_profile

epex_spot_import_price = Flex_Basic_Optimization.epex_spot_import_price

epex_spot_export_price = Flex_Basic_Optimization.epex_spot_export_price

epex_spot_price = Flex_Basic_Optimization.EM_Price_15_min_res

battery_cycles_flex_iteration_count = np.zeros(iterations)

time_res = float(15/60)

print(time_res)

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

non_optimal_i_values = np.zeros(iterations)

j = 0

sum_prosumer_cost = 0.0

counter_bat_positive_flex = 0
counter_bat_negative_flex = 0
counter_PV_positive_flex = 0
counter_PV_negative_flex = 0

switching_operation_battery = [0, 1]

for i in range(iterations):

    bat_neg_flex = 0

    pv_pos_flex = 0

    pv_neg_flex = 0

    bat_pos_flex = 0

    model_flex = gp.Model("linear_flex_optimization_EMS")

    P_PV_export_grid_flex = model_flex.addVar(lb= 0,   vtype=gp.GRB.CONTINUOUS,
                                              name="P_PV_export_grid_flex")

    P_bat_charging_PV_flex = model_flex.addVar(lb= 0, vtype=gp.GRB.CONTINUOUS,
                                               name="P_bat_charging_PV_flex")

    P_bat_charging_grid_flex = model_flex.addVar(lb= 0, vtype=gp.GRB.CONTINUOUS,
                                                 name="P_bat_charging_grid_flex")

    P_bat_discharging_grid_flex = model_flex.addVar(lb= 0, vtype=gp.GRB.CONTINUOUS,
                                                    name="P_bat_discharging_grid_flex")

    P_bat_discharging_load_flex = model_flex.addVar(lb= 0, vtype=gp.GRB.CONTINUOUS,
                                                    name="P_bat_discharging_load_flex")

    P_bat_export = model_flex.addVar(lb=0, vtype=gp.GRB.CONTINUOUS,
                                                    name="P_bat_export")

    P_grid_import_flex = model_flex.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="P_grid_import_flex")

    if PV_ninety_percent_quantiles[i] < 0:
        PV_ninety_percent_quantiles[i] = 0

    uncertain_param_PV = [PV_ten_percent_quantiles[i], PV_ninety_percent_quantiles[i]]

    uncertain_param_Load = [Load_ten_percent_quantiles[i], Load_ninety_percent_quantiles[i]]

    binary_battery_variable = random.randint(0, 2)

    # Battery max and min power constraints with alternate charging or discharging events

    #sign convention - charging/feed-in to grid curtailed or charging from grid -ve; discharging to grid +ve


    def bat_positive_flex():  # battery charge from grid

        P_bat_positive_flex[i] = -(P_bat_max - abs(P_bat_charging_grid_opt[i])) # calling variables from object of basic optimization class

        t_bat_positive_flex = time_res

        E_bat_positive_flex[i] = P_bat_positive_flex[i] * t_bat_positive_flex

        bat_pos_flex_iteration_list.append(i)

        E_bat_flex_check[i + 1] = E_bat_loop_current_flex[i] + P_bat_positive_flex[i] * t_bat_positive_flex

        if  (E_bat_flex_check[i + 1] - E_bat_final) < sum(abs(P_bat_discharging_load_opt[j])*time_res for j in range(i + 1, iterations-1, 1)):

            for k in range(i + 1, iterations-1, 1):

                 if P_bat_positive_flex[k] > P_bat_positive_flex[i]:
                     E_bat_positive_flex[k] = P_bat_positive_flex[i] * t_bat_positive_flex
                     bat_pos_flex_iteration_list.append(k)


        return P_bat_positive_flex[i]


    def bat_negative_flex():  # battery not charging from grid and discharging to grid

         t_bat_negative_flex = time_res

         P_bat_negative_flex[i] = (P_bat_max - abs(P_bat_discharging_opt[i])) + abs(P_bat_charging_grid_opt[i])

         E_bat_flex_check[i + 1] = E_bat_loop_current_flex[i] + P_bat_negative_flex[i] * t_bat_negative_flex

         if P_bat_max - abs(P_bat_discharging_opt[i]) > 0 and  abs(P_bat_charging_grid_opt[i])  > 0:

            E_bat_negative_flex[i] = P_bat_negative_flex[i] * t_bat_negative_flex

            bat_neg_flex_iteration_list.append(i)

         elif P_bat_max - abs(P_bat_discharging_opt[i]) > 0 :

            P_bat_negative_flex[i] = (P_bat_max - abs(P_bat_discharging_opt[i]))

            E_bat_negative_flex[i] = P_bat_negative_flex[i] * t_bat_negative_flex

            bat_neg_flex_iteration_list.append(i)

         elif abs(P_bat_charging_grid_opt[i]) > 0 :

            P_bat_negative_flex[i] = abs(P_bat_charging_grid_opt[i])

            E_bat_negative_flex[i] = P_bat_negative_flex[i] * t_bat_negative_flex

            bat_neg_flex_iteration_list.append(i)




         if (E_bat_final - E_bat_flex_check[i + 1]) <= E_bat_max * (len(time_scale) - 1 - (i + 1)):

            for k in range(i + 1, i + 9, 1):

                if abs(P_bat_charging_grid_opt[k]) >= abs(P_bat_charging_grid_opt[i]) and (P_bat_max - abs(P_bat_discharging_opt[k])) >= (P_bat_max - abs(P_bat_discharging_opt[i])) :
                    if ((P_bat_max - abs(P_bat_discharging_opt[i])) + abs(P_bat_charging_grid_opt[i])) > 0:

                     P_bat_negative_flex[k] = (abs(P_bat_charging_grid_opt[i]) + (P_bat_max - abs(P_bat_discharging_opt[i])))
                     E_bat_negative_flex[k] = (abs(P_bat_charging_grid_opt[i]) + (P_bat_max - abs(P_bat_discharging_opt[i]))) * t_bat_negative_flex
                     bat_neg_flex_iteration_list.append(k)

                elif abs(P_bat_charging_grid_opt[k]) >= abs(P_bat_charging_grid_opt[i]):
                    if abs(P_bat_charging_grid_opt[i]) > 0:

                     P_bat_negative_flex[k] = abs(P_bat_charging_grid_opt[i])
                     E_bat_negative_flex[k] = abs(P_bat_charging_grid_opt[i])*t_bat_negative_flex
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


        for j in range(i + 1, i + 9, 1):

            if abs(P_PV_export_grid_opt[j]) >= abs(P_pv_positive_flex[i]):
                P_pv_positive_flex[j] = P_pv_positive_flex[i]
                E_pv_positive_flex[j] = P_pv_positive_flex[i] * t_pv_positive_flex
                pv_pos_flex_iteration_list.append(j)


        return P_pv_positive_flex[i]


    def PV_negative_flex():  # not charging battery from PV

        P_pv_negative_flex[i] = -P_bat_charging_PV_opt[i]

        t_pv_negative_flex = time_res

        E_pv_negative_flex[i] = P_pv_negative_flex[i] * t_pv_negative_flex

        pv_neg_flex_iteration_list.append(i)

        for j in range(i + 1, i + 9, 1):

            if abs(P_bat_charging_PV_opt[j]) >= abs(P_pv_negative_flex[i]):
                P_pv_negative_flex[j] = P_pv_negative_flex[i]
                E_pv_negative_flex[j] = P_pv_negative_flex[i] * t_pv_negative_flex
                pv_neg_flex_iteration_list.append(j)

        return P_pv_negative_flex[i]


    # bat_pos_flex = bat_positive_flex() # switch to 1 or 0

    #if i == 215: # 6 PM of 8th May :- Battery Flexibility Check

     #bat_neg_flex = bat_negative_flex()  # switch to 1 or 0

     #counter_bat_negative_flex = 1

    if i == 47: # 12 PM PV positive Flexibility Check

        pv_neg_flex = PV_negative_flex()

        counter_pv_negative_flex = 1

    #pv_neg_flex = PV_negative_flex()

    #counter_PV_negative_flex = 1

    #bat_pos_flex = bat_positive_flex()

    #counter_bat_positive_flex = 1

    # switch to 1 or 0


    print("Bat_Pos_Flex", bat_pos_flex)

    print("Bat_Neg_Flex", bat_neg_flex)

    print("PV_pos_flex", pv_pos_flex)

    print("PV_neg_flex", pv_neg_flex)

    P_flex[i] = pv_neg_flex + pv_pos_flex + bat_neg_flex + bat_pos_flex



    import_price_flex_costs = -(P_bat_positive_flex[i]) * time_res * 12
    export_price_flex_savings = (P_bat_negative_flex[i] + P_pv_positive_flex[i] - P_pv_negative_flex[
        i]) * time_res * 8  # import and export prices have a deviation of 2 EUR/MWh

    epex_spot_import_price_flex[i] = epex_spot_import_price[i]
    epex_spot_export_price_flex[i] = epex_spot_export_price[i]

    P_bat_charging_flex = (P_bat_charging_PV_flex + P_pv_negative_flex[i] - P_pv_positive_flex[i] - P_bat_export) + (
            P_bat_charging_grid_flex - P_bat_positive_flex[i])

    P_bat_discharging_flex = (P_bat_discharging_load_flex + P_bat_discharging_grid_flex  + P_bat_negative_flex[i])

    if binary_battery_variable == 0:

        model_flex.addConstr(P_bat_discharging_flex == 0)
        model_flex.addConstr(P_bat_charging_PV_flex >= 0.2 * (PV_ten_percent_quantiles[i] + PV_ninety_percent_quantiles[i])/2)

    else:

        model_flex.addConstr(P_bat_charging_flex == 0)

    model_flex.addConstr(P_bat_charging_flex >= P_bat_min)
    model_flex.addConstr(P_bat_charging_flex <= P_bat_max)
     #R3

    model_flex.addConstr(P_bat_discharging_flex >= P_bat_min) #R1
    model_flex.addConstr(P_bat_discharging_flex <= P_bat_max) #R2

    # OPTIMIZATION WITH THE FLEX VALUES

    P_grid_delta_net_flex = P_grid_current_flex[
                                i] + P_PV_export_grid_flex - P_grid_import_flex + P_bat_discharging_grid_flex - P_bat_charging_grid_flex + (
                                        P_bat_negative_flex[i] + P_pv_positive_flex[i] - P_pv_negative_flex[i] + P_bat_positive_flex[i])
    # j sum

    E_bat_loop_flex = E_bat_loop_current_flex[i] + (
                P_bat_charging_flex * eta_batt - (P_bat_discharging_flex) / eta_batt) * time_res


    #model_flex.addConstr((P_grid_delta_net_flex) <= (P_grid_max)) #R3 # block i
    #model_flex.addConstr((P_grid_delta_net_flex) >= (P_grid_min)) #R4
    model_flex.addConstr((E_bat_loop_flex) >= (E_bat_min)) #R5
    model_flex.addConstr((E_bat_loop_flex) <= (E_bat_max))#R6
    model_flex.addConstr(P_PV_export_grid_flex + P_bat_charging_PV_flex  <= PV_ninety_percent_quantiles[i])#R7

    if i == (iterations - 1):
        model_flex.addConstr(E_bat_loop_flex == E_bat_final)#R9


    model_flex.addConstr(P_bat_export <= abs(P_pv_negative_flex[i])) #R10

    model_flex.addConstr(P_bat_charging_PV_flex >= -P_pv_negative_flex[i])

    model_flex.addConstr(
             ((P_bat_discharging_flex + P_grid_import_flex) - (
                         P_bat_charging_flex + P_PV_export_grid_flex)) <= (
                     uncertain_param_Load[1] - uncertain_param_PV[0]))  # R11

    model_flex.addConstr(((P_bat_discharging_flex + P_grid_import_flex) - (
                     P_bat_charging_flex + P_PV_export_grid_flex)) >= (
                                      uncertain_param_Load[0] - uncertain_param_PV[1])) #R12


    battery_cycles_flex = (P_bat_charging_flex + P_bat_discharging_flex) / P_bat_delta

    BAT_FLEX_OBJ = ((P_grid_import_flex ) * epex_spot_import_price_flex[i] - (P_PV_export_grid_flex ) * epex_spot_export_price_flex[
                      i])*time_res

    model_flex.setObjective(BAT_FLEX_OBJ, gp.GRB.MINIMIZE)

    model_flex.optimize()

    print(i)

    if model_flex.status == gp.GRB.OPTIMAL:
        print("Optimal solution found")
        print("Optimal objective value =", model_flex.objVal)
        sum_prosumer_cost += model_flex.objVal
        P_grid_current_flex[i + 1] = P_grid_current_flex[
                                         i] - P_grid_import_flex.x + P_PV_export_grid_flex.x + P_bat_discharging_grid_flex.x - P_bat_charging_grid_flex.x + (
                                    P_bat_negative_flex[i] + P_pv_positive_flex[i] - P_pv_negative_flex[i] +
                                    P_bat_positive_flex[i])

        E_bat_loop_current_flex[i + 1] = E_bat_loop_current_flex[i] + (((P_bat_charging_PV_flex.x + P_pv_negative_flex[i]  + P_bat_charging_grid_flex.x - P_bat_positive_flex[i]) * eta_batt - (
                                                                               P_bat_discharging_load_flex.x + P_bat_discharging_grid_flex.x + P_bat_negative_flex[i] - P_bat_export.x ) / eta_batt))*time_res

        P_PV_export_grid_opt_flex[i] = -abs(P_PV_export_grid_flex.x + P_pv_positive_flex[i] - P_pv_negative_flex[i])
        P_bat_charging_PV_opt_flex[i] = P_bat_charging_PV_flex.x - P_pv_positive_flex[i] + P_pv_negative_flex[i]
        P_bat_charging_grid_opt_flex[i] = P_bat_charging_grid_flex.x - P_bat_positive_flex[i]
        P_bat_discharging_grid_opt_flex[i] = -P_bat_discharging_grid_flex.x - P_bat_negative_flex[i]
        P_bat_discharging_load_opt_flex[i] = -P_bat_discharging_load_flex.x
        P_bat_export_opt[i] = P_bat_export.x
        P_bat_charging_opt_flex[i] = P_bat_charging_PV_flex.x + P_pv_negative_flex[i] + P_bat_charging_grid_flex.x - P_bat_positive_flex[i] - P_pv_positive_flex[i]  + P_pv_negative_flex[i]
        P_bat_discharging_opt_flex[i] = -P_bat_discharging_grid_flex.x - P_bat_discharging_load_flex.x - P_bat_negative_flex[i]
        P_grid_import_opt_flex[i] = P_grid_import_flex.x

        E_bat_discharging_load_flex[i] = P_bat_discharging_load_opt_flex[i] * time_res
        i_count_flex += 1

        battery_cycles_flex_iteration_count[i] = (P_bat_charging_opt_flex[i] + P_bat_discharging_opt_flex[i]) / P_bat_delta

    else:
        print("No solution found")
        model_flex.computeIIS()  # Compute the Infeasible Inconsistency Set (IIS)
        iis_constraints = [constr for constr in model_flex.getConstrs() if constr.IISConstr]
        print("Infeasible constraints:", iis_constraints)
        counter_non_optimal += 1
        E_bat_loop_current_flex[i + 1] = E_bat_initial  # battery set to initial state in case of non-optimal solution
        non_optimal_i_values[j] = i
        j += 1

    E_bat_profile_flex[i] = E_bat_loop_current_flex[i]

    P_grid_profile_flex[i] = P_grid_current_flex[i]



    print(i_count_flex)

time_points = [start_time + timedelta(minutes=15 * i) for i in range(len(time_scale))]

plt.plot(time_points, P_bat_charging_opt_flex, marker='*', label='Bat_charging_profile')

plt.plot(time_points, P_bat_charging_grid_opt_flex, marker='3', label='Bat_charging_grid_profile')
plt.plot(time_points, P_bat_charging_PV_opt_flex, marker='4', label='Bat_charging_PV_profile')
plt.plot(time_points, P_bat_discharging_opt_flex, marker='+', label='Bat_discharging_profile')

plt.plot(time_points, P_bat_discharging_load_opt_flex, marker='1', label='Bat_discharging_load_profile')
plt.plot(time_points, P_bat_discharging_grid_opt_flex, marker='2', label='Bat_discharging_grid_profile')
plt.plot(time_points, P_bat_export_opt, marker='.', label='P_bat_export_optimum_flow')
plt.plot(time_points, P_grid_import_opt_flex, marker='v', label='P_grid_import_optimum_flow')
plt.plot(time_points, P_PV_export_grid_opt_flex, marker='o', label='PV_export_grid_optimum_flow')
#plt.plot(time_points, E_bat_profile_flex, marker='8', label='E_battery_optimum_flow')


hours_3_interval = 3 * 60  # 3 hours in minutes
tick_positions = range(0, (len(time_points)+1) * 15, hours_3_interval)
tick_labels = [start_time + timedelta(minutes=i) for i in tick_positions]

plt.xticks(tick_labels)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# Rotate x-axis labels for better readability (optional)
plt.gcf().autofmt_xdate()
plt.title('EMS Robust Optimization with negative PV flexibility provision')
plt.xlabel('Time')
plt.ylabel('Power Flow in W')
plt.legend()
plt.show()

for i in range(iterations):

 RM = 0.15

 cost_per_cycle = bat_operational_cost # 10 cents is the cost per battery charging/discharging

 n_cycles = (P_bat_discharging_opt_flex[i] + P_bat_charging_opt_flex[i])/P_bat_delta   # to be found

    ### VVVVIIII - CHANGE OPT_FLEX to Variables to club prices in objective function
 def Bat_Pos_Flex_Price():

        C_bat_op_pos_flex = 0.5 * cost_per_cycle / n_cycles * (
                P_bat_charging_opt_flex[i] / P_bat_max)  # battery cycles flex to be replaced

        savings_bat_pos_flex = ( ( P_bat_charging_grid_opt[i] - P_bat_charging_grid_opt_flex[i]) *
                                epex_spot_import_price_flex[i]) * time_res

        price_flex_bat_pos = (savings_bat_pos_flex * (1 - RM) + C_bat_op_pos_flex) / (
            E_bat_positive_flex[i])

        return price_flex_bat_pos


 def Bat_Neg_Flex_Price():

        C_bat_op_neg_flex = 0.5 * cost_per_cycle / n_cycles * (
                abs(P_bat_discharging_opt_flex[i]))/ P_bat_max

        expenses = ((P_grid_import_opt_flex[i] + P_bat_charging_grid_opt_flex[i] - P_grid_import_opt[i] - P_bat_charging_grid_opt[i]) * epex_spot_import_price_flex[i] +
                    (P_bat_charging_PV_opt_flex[i] - P_bat_charging_PV_opt[i]) * epex_spot_export_price_flex[i]) * time_res

        price_flex_bat_neg = (expenses * (1 + RM) + C_bat_op_neg_flex) / (E_bat_negative_flex[i])

        return price_flex_bat_neg


 def PV_Pos_Flex_Price():

        profit = abs(P_bat_charging_PV_opt_flex[i] - P_bat_charging_PV_opt[i]) * epex_spot_import_price_flex[
            i] * time_res  # Battery charging from PV at grid import price

        losses = abs(P_PV_export_grid_opt[i] - P_PV_export_grid_opt_flex[i]) * epex_spot_export_price_flex[i] * time_res

        if losses > profit:
            price_flex_pv_pos = -(losses * (1 + RM) - profit) /abs( E_pv_positive_flex[i])

        elif losses == profit:
            price_flex_pv_pos = -(losses * RM) / abs(E_pv_positive_flex[i])

        else:
            price_flex_pv_pos = (profit - losses * (1 + RM)) / abs(E_pv_positive_flex[i])

        return price_flex_pv_pos


 def PV_Neg_Flex_Price():

        savings_pv_neg_flex = -abs(P_bat_charging_PV_opt[i] - P_bat_charging_PV_opt_flex[i]) * epex_spot_export_price_flex[
            i]*time_res

        price_flex_pv_neg = savings_pv_neg_flex * (1 + RM) / abs(E_pv_negative_flex[i])

        return price_flex_pv_neg


 for k in range(len(pv_neg_flex_iteration_list)):

     if i == pv_neg_flex_iteration_list[k]:
         PV_Negative_Flex_Price[i] = PV_Neg_Flex_Price()

 Price_flex[i] =  PV_Negative_Flex_Price[i] + PV_Positive_Flex_Price[i] + \
                    Bat_Negative_Flex_Price[i] + Bat_Positive_Flex_Price[i] + epex_spot_price[i]

 print(Price_flex[i])

    # for k in range(len(pv_neg_flex_iteration_list)):

    # if pv_pos_flex_iteration_list[k] == i:
    # PV_Negative_Flex_Price[i] = PV_Neg_Flex_Price()


print(pv_neg_flex_iteration_list)

print(P_pv_negative_flex)

print(E_pv_negative_flex)

print(non_optimal_i_values)

# Price Flex Calculation

print("Battery Positive Flex Price is", Bat_Positive_Flex_Price)

print("Battery Negative Flex Price is", Bat_Negative_Flex_Price)

print("PV Positive Flex Price is", PV_Positive_Flex_Price)

print("PV Negative Flex Price is", PV_Negative_Flex_Price)

print("Flex Price is", Price_flex)

print(sum_prosumer_cost)

E_PV_grid_optimum = np.zeros(iterations)
E_PV_grid_optimum[0] = E_bat_initial/1000
E_PV_grid_optimum_flex = np.zeros(iterations)
E_PV_grid_optimum_flex[0] = E_bat_initial/1000

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

           if abs(P_PV_export_grid_opt[j]) >= abs(P_pv_positive_flex[i]):
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

           if abs(P_bat_charging_PV_opt[j] >= abs(P_pv_negative_flex[i])):
               P_pv_negative_flex[j] = P_pv_negative_flex[i]
               E_pv_negative_flex[j] = P_pv_negative_flex[i] * t_pv_negative_flex
               pv_neg_flex_iteration_list.append(j)

       return P_pv_negative_flex[i]

   if i >= 1 and i <= 85:


    E_PV_grid_optimum[i] = E_PV_grid_optimum[i-1] + (P_PV_export_grid_opt[i-1]) * time_res / 1000

    E_PV_grid_optimum_flex[i] = E_PV_grid_optimum_flex[i-1] + (P_PV_export_grid_opt_flex[i-1]) * time_res / 1000

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
    original_time_array_pos_flex = np.arange(i, i + len(pv_pos_flex_iteration_list), 1) * 15  # minutes

    # Convert to datetime objects
    time_pos_flex_vector = [datetime(2019, 7, 25, 0, 0) + timedelta(minutes=int(time)) for time in original_time_array_pos_flex]

    print(time_pos_flex_vector)

    print(E_positive_flex)

    plt.plot(time_pos_flex_vector, E_positive_flex, color='blue')

    E_negative_flex[0] = E_PV_grid_optimum[i]

    for k in (np.arange(1, len(pv_neg_flex_iteration_list), 1)):
        E_negative_flex[k] = E_negative_flex[k - 1] + P_pv_negative_flex[i] * time_res / 1000

    #time_neg_flex_vector = np.arange(i, i + len(bat_neg_flex_iteration_list), 1)

    # Original time array with 15-minute resolution
    original_time_array_neg_flex = np.arange(i, i + len(pv_neg_flex_iteration_list), 1) * 15  # minutes

    # Convert to datetime objects
    time_neg_flex_vector = [datetime(2019, 7, 25, 0, 0) + timedelta(minutes=int(time)) for time in original_time_array_neg_flex]

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

plt.plot(time_objects, E_PV_grid_optimum, color= 'brown', marker='*', label= 'PV-Grid-Exchanged-Energy-Optimum')
plt.plot(time_objects, E_PV_grid_optimum_flex, color= 'orange', marker='.', label='PV-Grid-Exchanged-Energy-Flex-Optimum')

# Set ticks and labels for the x-axis
plt.xticks([time_objects[i] for i in new_time_array], [f'{i/4:.0f}:00' for i in new_time_array], rotation=45)

plt.title('Comparison of Exchanged Energy in Basic and Flexible Optimization')
plt.xlabel('Day Time Scale with 15 min time resolution')
plt.ylabel('Cummulative Exchanged Energy in kWh')

plt.tight_layout()
plt.legend()
plt.show()