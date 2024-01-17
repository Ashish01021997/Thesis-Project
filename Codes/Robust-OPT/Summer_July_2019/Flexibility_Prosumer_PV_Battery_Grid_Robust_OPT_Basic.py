import gurobipy as gp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import Quantile_Load_forecasts
import Quantile_PV_forecasts
from numpy import random
import pandas as pd
# performing firstly a robust OPT based on 2 quantiles [0.4, 0.6]
start_time = datetime.strptime('00:00', '%H:%M')

class EMS_Basic_Optimization:

 timesteps = np.arange(0, 96, 1)


 iterations = len(timesteps)

 C_bat = 4600 # in Wh
 Panel_area = 4  # 2msq 2 panels area x 2 panels (1.937 sq m from a article : (https://www.skillstg.co.uk/blog/solar-photovoltaic-panel-sizes/#:~:text=The%20physical%20dimensions%20of%20most,thickness%20of%20around%201.5%20inches)
 counter = 0
 SOC_min = 0.20
 SOC_max = 0.95 # soc limits from the research paper (https://www.researchgate.net/publication/373737720_A_PROGNOSTIC_DECISION-MAKING_APPROACH_UNDER_UNCERTAINTY_FOR_AN_ELECTRIC_VEHICLE_FLEET_ROUTING_PROBLEM)
 E_bat_min = SOC_min * C_bat
 E_bat_max = SOC_max * C_bat
 P_bat_min = 0 # 0.01kW # Assumption to make P_bat_min equal to zero for simplicity reasons
 P_bat_max = 1250 # 1.25kW
 P_bat_delta = P_bat_max - P_bat_min
 E_bat_initial = (E_bat_max + E_bat_min)/2
 E_bat_final = E_bat_initial
 eta_batt = 0.93 # taken 0.93 in research paper
 P_grid_max = 2500
 P_grid_min = -2500 # GRID LIMITS IN 0.1 kW AND 4 kW
 P_grid_delta = P_grid_max - P_grid_min
 P_grid = np.zeros(iterations) # Grid is stabilised at 0 kW power

 #n_steps_hour = 60


 PV_forecaster = Quantile_PV_forecasts.PV_Quantiles()

 Load_forecaster = Quantile_Load_forecasts.Load_Quantiles()

 load_scaling_factor = 1 # https://www.cleanenergywire.org/factsheets/germanys-energy-consumption-and-power-mix-charts

 EM_Price = pd.read_excel('Electricity_Market_Price.xlsx', sheet_name= 'Electricity_Market_Price_July')['Preis (EUR/MWh, EUR/tCO2)'] # for the year 2015 - LINK : https://ember-climate.org/data-catalogue/european-wholesale-electricity-price-data/

 #changing time resolution of energy market price from 1 hour to 5 min

 Load_forecasts = Load_forecaster.df_predictions_load_15min

 Load_ten_percent_quantiles = Load_forecasts['ten_percent_quantile'] * load_scaling_factor

 Load_thirty_percent_quantiles = Load_forecasts['thirty_percent_quantile'] * load_scaling_factor

 Load_forty_percent_quantiles = Load_forecasts['forty_percent_quantile'] * load_scaling_factor

 Load_fifty_percent_quantiles = Load_forecasts['fifty_percent_quantile'] * load_scaling_factor

 Load_sixty_percent_quantiles = Load_forecasts['sixty_percent_quantile'] * load_scaling_factor

 Load_seventy_percent_quantiles = Load_forecasts['seventy_percent_quantile'] * load_scaling_factor

 Load_ninety_percent_quantiles = Load_forecasts['ninety_percent_quantile'] * load_scaling_factor

 PV_forecasts = PV_forecaster.df_predictions_PV

 print(PV_forecasts)

 PV_ten_percent_quantiles = PV_forecasts['ten_percent_quantile'] * Panel_area

 PV_thirty_percent_quantiles = PV_forecasts['thirty_percent_quantile'] * Panel_area

 PV_forty_percent_quantiles = PV_forecasts['forty_percent_quantile'] * Panel_area

 PV_fifty_percent_quantiles = PV_forecasts['fifty_percent_quantile'] * Panel_area

 PV_sixty_percent_quantiles = PV_forecasts['sixty_percent_quantile'] * Panel_area

 PV_seventy_percent_quantiles = PV_forecasts['seventy_percent_quantile'] * Panel_area

 PV_ninety_percent_quantiles = PV_forecasts['ninety_percent_quantile'] * Panel_area


 #offset = np.ones(iterations)*10.0 # offset is 10 EUR/MWh


 bat_operational_cost = 0.01 # 1 cent/cycle


 E_bat_charging_profile = np.zeros(iterations)
 P_grid_profile = np.zeros(iterations)
 E_bat_profile = np.zeros(iterations)
 P_grid_import_profile = np.zeros(iterations)
 P_PV_export_grid_profile = np.zeros(iterations)
 E_bat_discharging_profile = np.zeros(iterations)

 E_bat_profile[0] = E_bat_initial

 E_bat_loop_current = np.zeros(iterations+1)

 E_bat_loop_current[0] = E_bat_initial

 epex_spot_import_price = np.zeros(iterations)

 epex_spot_export_price = np.zeros(iterations)

 P_PV_export_grid_opt = np.zeros(iterations)

 P_bat_charging_PV_opt = np.zeros(iterations)

 P_bat_charging_grid_opt = np.zeros(iterations)

 P_bat_discharging_grid_opt = np.zeros(iterations)

 P_bat_discharging_load_opt = np.zeros(iterations)

 P_bat_charging_opt = np.zeros(iterations)

 P_bat_discharging_opt = np.zeros(iterations)

 P_grid_import_opt = np.zeros(iterations)

 time_res = (15/60) # 5 minutes in hours


 counter_non_optimal = 0

 non_optimal_i_values = np.zeros(iterations)

 j = 0

 switching_operation_battery = [0, 1]

 sum_cost_prosumer = 0.0


 #P_PV_gen = model.addVar(vtype=gp.GRB.CONTINUOUS, name="P_PV_gen")

 #P_load = model.addVar(vtype=gp.GRB.CONTINUOUS, name="P_load")



 # basic_optimization_loop()

 for i in range(iterations):

    model = gp.Model("linear_basic_optimization_EMS")

    P_bat_charging_PV = model.addVar(lb=0 , vtype=gp.GRB.CONTINUOUS, name="P_bat_charging_PV")

    P_bat_charging_grid = model.addVar(lb=0,  vtype=gp.GRB.CONTINUOUS, name="P_bat_charging_grid")

    P_bat_discharging_grid = model.addVar(lb=0,  vtype=gp.GRB.CONTINUOUS, name="P_bat_discharging_grid")

    P_bat_discharging_load = model.addVar(lb=0,   vtype=gp.GRB.CONTINUOUS, name="P_bat_discharging_load")

    P_grid_import = model.addVar(lb= 0,  vtype=gp.GRB.CONTINUOUS, name="P_grid_import")

    P_PV_export_grid = model.addVar(lb=0,  vtype=gp.GRB.CONTINUOUS, name="P_PV_export_grid")

    P_bat_charging = (P_bat_charging_PV + P_bat_charging_grid)

    P_bat_discharging = (P_bat_discharging_load + P_bat_discharging_grid)

    epex_spot_import_price[i] = (EM_Price[i])

    epex_spot_export_price[i] = 90

    if PV_ninety_percent_quantiles[i] < 0:
        PV_ninety_percent_quantiles[i] = 0

    uncertain_param_PV = [PV_ten_percent_quantiles[i], PV_ninety_percent_quantiles[i]]

    uncertain_param_Load = [Load_ten_percent_quantiles[i], Load_ninety_percent_quantiles[i]]

    if i%2 == 0:
        binary_battery_variable = switching_operation_battery[0]
    else:
        binary_battery_variable = switching_operation_battery[1]


    #binary_battery_variable = random.randint(0,1)


    # Battery max and min power constraints with alternate charging or discharging events

    if binary_battery_variable == 1:

        model.addConstr(P_bat_discharging == 0)
        model.addConstr(P_bat_charging >= P_bat_min)
        model.addConstr(P_bat_charging <= P_bat_max)
        model.addConstr(P_bat_charging_PV == 0.25 * (PV_ninety_percent_quantiles[i] + PV_ten_percent_quantiles[i])/2)
    else:

        model.addConstr(P_bat_charging == 0)
        model.addConstr(P_bat_discharging >= P_bat_min)
        model.addConstr(P_bat_discharging <= P_bat_max)

    # 0 for battery charging

    P_grid_delta_net = P_grid[i] + (
            P_PV_export_grid + P_bat_discharging_grid - P_bat_charging_grid - P_grid_import)

    E_bat_loop = E_bat_loop_current[i] + (P_bat_charging * eta_batt - P_bat_discharging / eta_batt) * (time_res) # time_res is 5 min

    model.addConstr(
           ((P_bat_discharging + P_grid_import) - (P_bat_charging + P_PV_export_grid)) <= (uncertain_param_Load[1] - uncertain_param_PV[0])) #4

    model.addConstr(
            ((P_bat_discharging + P_grid_import) - (P_bat_charging + P_PV_export_grid)) >= (uncertain_param_Load[0] - uncertain_param_PV[1])) #5


    model.addConstr(P_grid_delta_net <= P_grid_max)  #6
    model.addConstr(P_grid_delta_net >= P_grid_min) #7
    model.addConstr(E_bat_loop >= E_bat_min) #8
    model.addConstr(E_bat_loop <= E_bat_max) #9
    model.addConstr(P_PV_export_grid + P_bat_charging_PV <= PV_ninety_percent_quantiles[i])



    if i == (iterations-1):
        model.addConstr(E_bat_loop == E_bat_final)


    battery_cycles = (P_bat_charging + P_bat_discharging)/P_bat_delta


    OBJECTIVE_FUNCTION = ((P_grid_import )  * epex_spot_import_price[i]  - (P_PV_export_grid) * epex_spot_export_price[i])* time_res + battery_cycles*bat_operational_cost


    model.setObjective(OBJECTIVE_FUNCTION, gp.GRB.MINIMIZE)

    print(i)

    model.optimize()

    if model.status == gp.GRB.OPTIMAL:
        counter += 1
        print("Optimal solution found")
        print("Optimal objective value =", model.objVal)
        sum_cost_prosumer += model.objVal

        P_grid_profile[i] = P_grid[i] - P_grid_import.x + P_PV_export_grid.x + P_bat_discharging_grid.x - P_bat_charging_grid.x

        E_bat_loop_current[i+1] = E_bat_loop_current[i] + ((P_bat_charging_PV.x + P_bat_charging_grid.x)*eta_batt - (P_bat_discharging_load.x + P_bat_discharging_grid.x)/eta_batt)*time_res

        E_bat_profile[i] = E_bat_loop_current[i]
        P_PV_export_grid_opt[i] = -P_PV_export_grid.x
        P_bat_charging_PV_opt[i] = P_bat_charging_PV.x
        P_bat_charging_grid_opt[i] = P_bat_charging_grid.x
        P_bat_discharging_grid_opt[i] = -P_bat_discharging_grid.x
        P_bat_discharging_load_opt[i] = -P_bat_discharging_load.x
        P_grid_import_opt[i] = P_grid_import.x
        P_bat_charging_opt[i] = P_bat_charging_PV.x + P_bat_charging_grid.x
        P_bat_discharging_opt[i] = -P_bat_discharging_grid.x - P_bat_discharging_load.x

    else:
        print("No solution found")
        model.computeIIS()  # Compute the Infeasible Inconsistency Set (IIS)
        iis_constraints = [constr for constr in model.getConstrs() if constr.IISConstr]
        print("Infeasible constraints:", iis_constraints)
        counter_non_optimal += 1
        E_bat_loop_current[i + 1] = E_bat_initial # battery set to initial state in case of non-optimal solution
        non_optimal_i_values[j] = i
        j += 1




 print(counter)

 print(counter_non_optimal)

 print(P_bat_charging_opt)

 print(P_bat_discharging_opt)

 print(P_grid_import_opt)



 start_time = datetime.strptime('00:00', '%H:%M')

 print(start_time)

 time_scale = [start_time + timedelta(minutes=15 * i) for i in range(len(timesteps))]

 plt.plot(time_scale, P_bat_charging_opt, marker='*', label='Bat_charging_profile')
 plt.plot(time_scale, P_bat_charging_grid_opt, marker='3', label='Bat_charging_grid_profile')
 plt.plot(time_scale, P_bat_charging_PV_opt, marker='4', label='Bat_charging_PV_profile')
 plt.plot(time_scale, P_bat_discharging_opt, marker='+', label='Bat_discharging_profile')
 plt.plot(time_scale, P_bat_discharging_load_opt, marker='1', label='Bat_discharging_load_profile')
 plt.plot(time_scale, P_bat_discharging_grid_opt, marker='2', label='Bat_discharging_grid_profile')
 #plt.plot(time_scale, P_grid_profile, marker='.', label='P_grid_optimum_flow')
 plt.plot(time_scale, P_grid_import_opt, marker='v', label='P_grid_import_optimum_flow')
 plt.plot(time_scale, P_PV_export_grid_opt, marker='o', label='PV_export_grid_optimum_flow')
 #plt.plot(time_scale, E_bat_profile, marker='8', label='E_battery_optimum_flow')

 # Customize the x-axis ticks to show a 3-hour gap
 hours_3_interval = 3 * 60  # 3 hours in minutes
 tick_positions = range(0, (len(time_scale)+1) * 15, hours_3_interval)
 tick_labels = [start_time + timedelta(minutes=i) for i in tick_positions]

 plt.xticks(tick_labels)
 plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

 plt.gcf().autofmt_xdate()
 plt.title('EMS Optimization without flexibility provision')
 plt.xlabel('X')
 plt.ylabel('y')
 plt.legend()
 plt.show()

 print(non_optimal_i_values)

 print(sum_cost_prosumer)























