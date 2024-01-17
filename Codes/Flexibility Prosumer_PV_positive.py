import gurobipy as gp
import EMPricepointforecast
import matplotlib.pyplot as plt
import numpy as np
import Quantile_Load_forecasts
import Quantile_PV_forecasts
import random as random
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import Flexibility_Prosumer_PV_Battery_Grid_Robust_OPT_Basic

Flex_Basic_Optimization = Flexibility_Prosumer_PV_Battery_Grid_Robust_OPT_Basic.EMS_Basic_Optimization()

C_bat = 4600 # in Wh
Panel_area = 4  # 2msq panel area (1.937 sq m from a article : (https://www.skillstg.co.uk/blog/solar-photovoltaic-panel-sizes/#:~:text=The%20physical%20dimensions%20of%20most,thickness%20of%20around%201.5%20inches)
counter = 0
SOC_min = 0.20
SOC_max = 0.95 # soc limits from the research paper (https://www.researchgate.net/publication/373737720_A_PROGNOSTIC_DECISION-MAKING_APPROACH_UNDER_UNCERTAINTY_FOR_AN_ELECTRIC_VEHICLE_FLEET_ROUTING_PROBLEM)
E_bat_min = SOC_min * C_bat
E_bat_max = SOC_max * C_bat
P_bat_min = 10 # 0.01kW
P_bat_max = 500 # 1,25kW
P_bat_delta = P_bat_max - P_bat_min
E_bat_initial = (E_bat_max + E_bat_min)/2
E_bat_final = E_bat_initial
eta_batt = 0.93
P_grid_min = -500
P_grid_max = 500
P_grid_range = P_grid_max - P_grid_min
P_grid_initial = (P_grid_min + P_grid_max)/2

time_scale = np.arange(0, 288, 1) # 5 days is the simulation time-frame



iterations = len(time_scale)

EM_Price_forecaster = EMPricepointforecast.EM_Price_point_forecast()

PV_forecaster = Quantile_PV_forecasts.PV_Quantiles()

Load_forecaster = Quantile_Load_forecasts.Load_Quantiles()

load_scaling_factor = 109/112.4 # https://www.cleanenergywire.org/factsheets/germanys-energy-consumption-and-power-mix-charts

EM_Price = EM_Price_forecaster.y_pred # for the year 2015 - LINK : https://ember-climate.org/data-catalogue/european-wholesale-electricity-price-data/

Load_hour_forecasts = Load_forecaster.df_predictions_load

Load_ten_percent_quantiles = Load_hour_forecasts['ten_percent_quantile']*load_scaling_factor

Load_ninety_percent_quantiles = Load_hour_forecasts['ninety_percent_quantile']*load_scaling_factor

PV_hour_forecasts = PV_forecaster.df_predictions_PV

PV_ten_percent_quantiles = PV_hour_forecasts['ten_percent_quantile']*Panel_area

PV_ninety_percent_quantiles = PV_hour_forecasts['ninety_percent_quantile']*Panel_area


offset = np.ones(iterations)*10.0 # offset is 10 EUR/MWh


bat_operational_cost = 5
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

P_grid_import_opt_flex = np.zeros(iterations)

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

epex_spot_price = Flex_Basic_Optimization.EM_Price

battery_cycles_flex_iteration_count = np.zeros(iterations)



time_res = float(5/60)

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


for i in range(iterations):

    bat_neg_flex = 0

    pv_pos_flex = 0

    pv_neg_flex = 0

    bat_pos_flex = 0

    counter_bat_positive_flex = 0


    counter_PV_negative_flex = 0

    model_flex = gp.Model("linear_flex_optimization_EMS")

    P_PV_export_grid_flex = model_flex.addVar(lb= 0, vtype=gp.GRB.CONTINUOUS,
                                              name="P_PV_export_grid_flex")

    P_bat_charging_PV_flex = model_flex.addVar(lb=P_bat_min, ub=P_bat_max, vtype=gp.GRB.CONTINUOUS,
                                               name="P_bat_charging_PV_flex")

    P_bat_charging_grid_flex = model_flex.addVar(lb=P_bat_min, ub=P_bat_max, vtype=gp.GRB.CONTINUOUS,
                                                 name="P_bat_charging_grid_flex")

    P_bat_discharging_grid_flex = model_flex.addVar(lb=P_bat_min, ub=P_bat_max, vtype=gp.GRB.CONTINUOUS,
                                                    name="P_bat_discharging_grid_flex")

    P_bat_discharging_load_flex = model_flex.addVar(lb=P_bat_min, ub=P_bat_max, vtype=gp.GRB.CONTINUOUS,
                                                    name="P_bat_discharging_load_flex")

    P_grid_import_flex = model_flex.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="P_grid_import_flex")

    uncertain_param_PV = [PV_ten_percent_quantiles[i], PV_ninety_percent_quantiles[i]]

    uncertain_param_Load = [Load_ten_percent_quantiles[i], Load_ninety_percent_quantiles[i]]

    binary_battery_variable = random.randint(0, 1)

    # Battery max and min power constraints with alternate charging or discharging events


    #sign convention - charging/feed-in to grid curtailed or charging from grid -ve; discharging to grid +ve


    def bat_positive_flex():  # battery charge from grid

        P_bat_positive_flex[i] = -(P_bat_max - abs(P_bat_charging_grid_opt[i])) # calling variables from object of basic optimization class

        t_bat_positive_flex = time_res

        E_bat_positive_flex[i] = P_bat_positive_flex[i] * t_bat_positive_flex

        bat_pos_flex_iteration_list.append(i)

        E_bat_flex_check[i + 1] = E_bat_loop_current_flex[i] + P_bat_positive_flex[i] * t_bat_positive_flex

        if  (E_bat_flex_check[i + 1] - E_bat_final) < sum(P_bat_discharging_load_opt[j]*time_res for j in range(i + 1, iterations-1, 1)):

            for k in range(i + 1, iterations-1, 1):

                 if P_bat_positive_flex[k] >= P_bat_positive_flex[i]:

                     P_bat_positive_flex[k] = -(P_bat_max - abs(P_bat_charging_grid_opt[i]))
                     E_bat_positive_flex[k] = P_bat_positive_flex[i] * t_bat_positive_flex
                     bat_pos_flex_iteration_list.append(k)

        return P_bat_positive_flex[i]


    def bat_negative_flex():  # battery not charging from grid and discharging to grid

        counter_bat_negative_flex = 0

        P_bat_negative_flex[i] = (P_bat_max - abs(P_bat_discharging_opt[i])) + abs(P_bat_charging_grid_opt[i])

        t_bat_negative_flex = time_res

        E_bat_negative_flex[i] = P_bat_negative_flex[i] * t_bat_negative_flex

        bat_neg_flex_iteration_list.append(i)


        E_bat_flex_check[i + 1] = E_bat_loop_current_flex[i] + P_bat_negative_flex[i] * t_bat_negative_flex

        if (E_bat_final - E_bat_flex_check[i + 1]) <= E_bat_max * (len(time_scale) - 1 - (i + 1)):

            for k in range(i + 1, iterations-1, 1):

                if abs(P_bat_charging_grid_opt[k] >= abs(P_bat_charging_grid_opt[i])):

                    P_bat_negative_flex[k] = (P_bat_max - abs(P_bat_discharging_opt[i])) + abs(P_bat_charging_grid_opt[i])
                    E_bat_negative_flex[k] = P_bat_negative_flex[i] * t_bat_negative_flex
                    counter_bat_negative_flex += 1
                    bat_neg_flex_iteration_list.append(k)


        return P_bat_negative_flex[i]


    def PV_positive_flex():  # curtailing pv feed in to grid

        print(P_PV_export_grid_opt[i])

        P_pv_positive_flex[i] = -abs(P_PV_export_grid_opt[i])

        t_pv_positive_flex = time_res

        E_pv_positive_flex[i] = P_pv_positive_flex[i] * t_pv_positive_flex

        counter_PV_positive_flex = 0

        pv_pos_flex_iteration_list.append(i)


        for j in range(i + 1, iterations-1, 1):

            if abs(P_PV_export_grid_opt[j]) >= abs(P_pv_positive_flex[i]):

                P_pv_positive_flex[j] = -abs(P_PV_export_grid_opt[i])
                E_pv_positive_flex[j] = P_pv_positive_flex[i] * t_pv_positive_flex
                counter_PV_positive_flex += 1
                pv_pos_flex_iteration_list.append(j)

            else:
                break




        print(t_pv_positive_flex)

        print(counter_PV_positive_flex)

        return P_pv_positive_flex[i]


    def PV_negative_flex():  # not charging battery from PV

        P_pv_negative_flex[i] = -abs(P_bat_charging_PV_opt[i])

        t_pv_negative_flex = time_res

        E_pv_negative_flex[i] = P_pv_negative_flex[i] * t_pv_negative_flex

        pv_neg_flex_iteration_list.append(i)



        for j in range(i + 1, iterations-1, 1):

            if abs(P_bat_charging_PV_opt[j] >= abs(P_pv_negative_flex[i])):

                P_pv_negative_flex[j] = -abs(P_bat_charging_PV_opt[i])
                E_pv_negative_flex[j] = P_pv_negative_flex[i] * t_pv_negative_flex
                pv_neg_flex_iteration_list.append(j)



        return P_pv_negative_flex[i]


    # bat_pos_flex = bat_positive_flex() # switch to 1 or 0

    #if i == 215: # 6 PM of 8th May :- Battery Flexibility Check

     #bat_neg_flex = bat_negative_flex()  # switch to 1 or 0

     #counter_bat_negative_flex = 1

    if i == 144: # 11 AM around PV POSITIVE Flexibility Check

      pv_pos_flex = PV_positive_flex()

      print(E_pv_positive_flex[i])



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




    # OPTIMIZATION WITH THE FLEX VALUES

    # P_bat_discharging_grid += bat_neg_flex

    # P_bat_charging_grid += bat_pos_flex

    # P_bat_charging_PV += pv_neg_flex

    # P_PV_export_grid += pv_pos_flex

    import_price_flex_costs = -(P_bat_positive_flex[i]) * time_res * 10 * 1e-6
    export_price_flex_savings = (P_bat_negative_flex[i] + P_pv_positive_flex[i] - P_pv_negative_flex[i]) * time_res * 10 * 1e-6

    epex_spot_import_price_flex[i] = epex_spot_import_price[i]
    epex_spot_export_price_flex[i] = epex_spot_export_price[i]   # prices in EUR/MWh

    # P_load_flex = model_flex.addVar(lb=Load_hour_forecasts_lower[i], ub=Load_hour_forecasts_upper[i], vtype=gp.GRB.CONTINUOUS,name="P_PV_gen")# only pv uncertainity # constant load array

    P_bat_charging_flex = (P_bat_charging_PV_flex + P_pv_negative_flex[i]-P_pv_positive_flex[i]) + (P_bat_charging_grid_flex - P_bat_positive_flex[i])

    P_bat_discharging_flex = (P_bat_discharging_load_flex + P_bat_discharging_grid_flex + P_bat_negative_flex[i])

    P_grid_delta_net_flex = P_grid_current_flex[
                                i] + P_PV_export_grid_flex - P_grid_import_flex + P_bat_discharging_grid_flex - P_bat_charging_grid_flex + (
                                        P_bat_negative_flex[i] + P_pv_positive_flex[i] - P_pv_negative_flex[i] + P_bat_positive_flex[i])  # j sum

    E_bat_loop_flex = E_bat_loop_current_flex[i] + (P_bat_charging_flex * eta_batt - P_bat_discharging_flex / eta_batt)*time_res

    if binary_battery_variable == 0:
        model_flex.addConstr(P_bat_discharging_flex == 0)
        model_flex.addConstr(P_bat_charging_flex >= P_bat_min)
        model_flex.addConstr(P_bat_charging_flex <= P_bat_max)
    else:
        model_flex.addConstr(P_bat_charging_flex == 0)
        model_flex.addConstr(P_bat_discharging_flex >= P_bat_min)
        model_flex.addConstr(P_bat_discharging_flex <= P_bat_max)

    model_flex.addConstr((P_grid_delta_net_flex) <= (P_grid_max))  # block i
    model_flex.addConstr((P_grid_delta_net_flex) >= (P_grid_min))
    model_flex.addConstr((E_bat_loop_flex) >= (E_bat_min))
    model_flex.addConstr((E_bat_loop_flex) <= (E_bat_max))
    model_flex.addConstr(P_bat_charging_PV_flex + P_PV_export_grid_flex <= uncertain_param_PV[1])

    model_flex.addConstr(
        ((P_bat_discharging_load_flex + P_bat_discharging_grid_flex + P_grid_import_flex) - (
                    P_bat_charging_grid_flex + P_PV_export_grid_flex)) <= (
                uncertain_param_Load[1] - uncertain_param_PV[0]))  # R6

    model_flex.addConstr(((P_bat_discharging_load_flex + P_bat_discharging_grid_flex + P_grid_import_flex) - (
                P_bat_charging_grid_flex + P_PV_export_grid_flex)) >= (
                                 uncertain_param_Load[0] - uncertain_param_PV[1]))



    P_grid_delta_flex = P_grid_import_flex + P_bat_charging_grid_flex

    battery_cycles_flex = (P_bat_charging_flex + P_bat_discharging_flex) / P_bat_delta



    PV_POS_FLEX_OBJ = P_grid_delta_flex * epex_spot_import_price_flex[i] - (P_PV_export_grid_flex + P_bat_discharging_grid_flex) * epex_spot_export_price_flex[
                   i] + battery_cycles_flex * bat_operational_cost + import_price_flex_costs + export_price_flex_savings

    model_flex.setObjective(PV_POS_FLEX_OBJ, gp.GRB.MINIMIZE)

    model_flex.optimize()

    print(i)

    if model_flex.status == gp.GRB.OPTIMAL:
        print("Optimal solution found")
        print("Optimal objective value =", model_flex.objVal)
        P_grid_current_flex[i + 1] = P_grid_current_flex[
                                         i] - P_grid_import_flex.x + P_PV_export_grid_flex.x + P_bat_discharging_grid_flex.x - P_bat_charging_grid_flex.x + (
                                             P_bat_negative_flex[i] + P_pv_positive_flex[i] - P_pv_negative_flex[i] +
                                             P_bat_positive_flex[i])

        E_bat_loop_current_flex[i + 1] = E_bat_loop_current_flex[i] + ((P_bat_charging_PV_flex.x + P_pv_negative_flex[
            i] + P_bat_charging_grid_flex.x - P_bat_positive_flex[i]) * eta_batt - (
                                                                               P_bat_discharging_load_flex.x + P_bat_discharging_grid_flex.x +
                                                                               P_bat_negative_flex[i]) / eta_batt)

        P_PV_export_grid_opt_flex[i] = -(P_PV_export_grid_flex.x + P_pv_positive_flex[i])
        P_bat_charging_PV_opt_flex[i] = P_bat_charging_PV_flex.x
        P_bat_charging_grid_opt_flex[i] = P_bat_charging_grid_flex.x
        P_bat_discharging_grid_opt_flex[i] = -P_bat_discharging_grid_flex.x
        P_bat_discharging_load_opt_flex[i] = -P_bat_discharging_load_flex.x
        P_bat_charging_opt_flex[i] = P_bat_charging_PV_flex.x + P_pv_negative_flex[i] + P_bat_charging_grid_flex.x - \
                                     P_bat_positive_flex[i] - P_pv_positive_flex[i]
        P_bat_discharging_opt_flex[i] = -P_bat_discharging_grid_flex.x - P_bat_discharging_load_flex.x - \
                                        P_bat_negative_flex[i]
        P_grid_import_opt_flex[i] = P_grid_import_flex.x

        E_bat_discharging_load_flex[i] = P_bat_discharging_load_opt_flex[i] * time_res
        i_count_flex += 1

        battery_cycles_flex_iteration_count[i] = (P_bat_charging_opt_flex[i] + P_bat_discharging_opt_flex[i]) / P_bat_delta

    else:
        print("No solution found")
        counter_non_optimal += 1
        E_bat_loop_current_flex[i + 1] = E_bat_initial  # battery set to initial state in case of non-optimal solution
        non_optimal_i_values[j] = i
        j += 1

    E_bat_profile_flex[i] = E_bat_loop_current_flex[i]

    P_grid_profile_flex[i] = P_grid_current_flex[i]



    print(i_count_flex)



start_time = datetime.strptime('00:00', '%H:%M')

time_points = [start_time + timedelta(minutes=5 * i) for i in range(len(time_scale))]

plt.plot(time_points, P_bat_charging_opt_flex, marker='*', label='Bat_charging_profile')
plt.plot(time_points, P_bat_charging_grid_opt_flex, marker='3', label='Bat_charging_grid_profile')
plt.plot(time_points, P_bat_charging_PV_opt_flex, marker='4', label='Bat_charging_PV_profile' )
plt.plot(time_points, P_bat_discharging_opt_flex, marker='+', label='Bat_discharging_profile')
plt.plot(time_points, P_bat_discharging_load_opt_flex, marker='1', label='Bat_discharging_load_profile')
plt.plot(time_points, P_bat_discharging_grid_opt_flex, marker='2', label='Bat_discharging_grid_profile')
plt.plot(time_points, P_grid_profile_flex, marker='.', label='P_grid_optimum_flow')
plt.plot(time_points, P_grid_import_opt_flex, marker='v', label='P_grid_import_optimum_flow')
plt.plot(time_points, P_PV_export_grid_opt_flex, marker='o', label='PV_export_grid_optimum_flow')
plt.plot(time_points, E_bat_profile_flex, marker='8', label='E_battery_optimum_flow')


hours_3_interval = 3 * 60  # 3 hours in minutes
tick_positions = range(0, (len(time_points)+1) * 5, hours_3_interval)
tick_labels = [start_time + timedelta(minutes=i) for i in tick_positions]

plt.xticks(tick_labels)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# Rotate x-axis labels for better readability (optional)
plt.gcf().autofmt_xdate()

plt.title('EMS Optimization with positive PV flexibility provision')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Price Flex Calculation

for i in range(iterations):

    RM = 0.15

    cost_per_cycle = 0.1  # 10 cents is the cost per battery charging/discharging

    n_cycles = 1  # to be found


    ### VVVVIIII - CHANGE OPT_FLEX to Variables to club prices in objective function
    def Bat_Pos_Flex_Price():

        C_bat_op_pos_flex = 0.5 * cost_per_cycle / n_cycles * (
                P_bat_charging_opt_flex[i] / P_bat_max)  # battery cycles flex to be replaced

        savings_bat_pos_flex = (P_bat_discharging_load_opt_flex[i] * epex_spot_export_price_flex[
            i] + P_PV_export_grid_opt_flex[i] * epex_spot_export_price_flex[i] + (
                                        - P_bat_charging_grid_opt[i] + P_bat_charging_grid_opt_flex[i]) *
                                epex_spot_import_price_flex[i]) * time_res

        price_flex_bat_pos = (savings_bat_pos_flex * (1 - RM) + C_bat_op_pos_flex) / (
            E_bat_positive_flex[i])

        return price_flex_bat_pos


    def Bat_Neg_Flex_Price():

        C_bat_op_neg_flex = 0.5 * cost_per_cycle / n_cycles * (
                P_bat_discharging_opt_flex[i] / P_bat_max)

        expenses = ((P_grid_import_opt_flex[i] + P_bat_charging_grid_opt_flex[i]) * epex_spot_import_price_flex[i] +
                    P_bat_charging_PV_opt_flex[i] * epex_spot_export_price_flex[i]) * time_res

        price_flex_bat_neg = (expenses * (1 + RM) + C_bat_op_neg_flex) / (E_bat_negative_flex[i])

        return price_flex_bat_neg


    def PV_Pos_Flex_Price():

        losses = P_bat_charging_PV_opt_flex[i] * epex_spot_import_price_flex[
            i] * time_res  # Battery charging from PV at grid import price

        profit = P_PV_export_grid_opt_flex[i] * epex_spot_import_price_flex[i] * time_res

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




    for k in range(len(pv_pos_flex_iteration_list)):

        if i == pv_pos_flex_iteration_list[k]:

            PV_Positive_Flex_Price[pv_pos_flex_iteration_list[k]] = PV_Pos_Flex_Price()



    Price_flex[i] = epex_spot_price[i]*1e-6 + PV_Negative_Flex_Price[i] + PV_Positive_Flex_Price[i] + \
                    Bat_Negative_Flex_Price[i] + Bat_Positive_Flex_Price[i]

print(pv_pos_flex_iteration_list)

print(E_pv_positive_flex)

print(PV_Positive_Flex_Price)

print("Flex Price is", Price_flex)

