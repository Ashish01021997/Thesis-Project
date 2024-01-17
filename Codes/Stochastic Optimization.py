import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
import PV_probablistic_forecast
import Load_forecasting
import EMPricepointforecast
import matplotlib.pyplot as plt

# Probabilistic forecasts of PV and Load
PV_forecaster = PV_probablistic_forecast.PV_probablistic_forecasting()
PV_forecasts = PV_forecaster.y_pred
Load_forecaster = Load_forecasting.Load_point_forecast()
Load_forecasts = Load_forecaster.y_pred
EM_Price_forecaster = EMPricepointforecast.EM_Price_point_forecast()

SOC_min = 10 / 100
SOC_max = 90 / 100
No_of_households = 41.6 * 1e6
C_bat = 400.0
Panel_area = 5
counter = 0
E_bat_min = SOC_min * C_bat
E_bat_max = SOC_max * C_bat
E_bat =  E_bat_max
E_bat_loop = E_bat
P_load_opt = []
P_pv_watt_opt = np.zeros(288)
time_scale = np.arange(1, 289, 1)
bat_cycle = 0
grid_operational_cost = 2 # price in euros
bat_operational_cost = 0.01
bat_degradation_cost = 0.75
epex_spot_price = EM_Price_forecaster.y_pred
P_grid_delta = 0
E_bat_delta = 0
E_bat_loop_profile = np.zeros(288)

# Define the objective function to maximize expected utility (e.g., mean-variance utility)
# simulation for 12 days or 12*24 = 288 timesteps
for i in range(287):

 P_load_hour = (Load_forecasts[i]*1e6) / No_of_households
 print(P_load_hour)
 E_bat_loop_profile[0] = E_bat_loop



 def objective(x):



     bat_cycle = 0
     bat_cycle_loop = bat_cycle
     SOC = x[1] / C_bat


     if (x[0] == P_load_hour):
        P_grid_delta = 0
        E_bat_delta = 0
        x[1] += (E_bat_delta)
        #E_bat_loop_profile[i + 1] = E_bat_loop
        print("block 1")
        counter = 1
        SOC = x[1] / C_bat

     if x[0] >= P_load_hour:

        if x[1] == E_bat_max:
            P_grid_delta = (x[0] - P_load_hour)
            E_bat_delta = 0
            #E_bat_loop_profile[i + 1] = E_bat_loop
            x[1] += (E_bat_delta)
            print("block 2")
            counter = 2

            SOC = x[1] / C_bat

        else:
            if  (x[0]) >= (x[1]- P_load_hour - E_bat_max):

                E_bat_delta = (x[0] - P_load_hour)
                x[1] += (E_bat_delta)
                #E_bat_loop_profile[i + 1] = E_bat_loop
                P_grid_delta = 0
                bat_cycle_loop += 1
                print("block 3")
                counter = 3

                SOC = x[1] / C_bat

            else:
                E_bat_delta = (- x[1] + E_bat_max)
                x[1] += (E_bat_delta)
                #E_bat_loop_profile[i + 1] = E_bat_loop
                P_grid_delta =  - E_bat_delta +  (x[0] - P_load_hour)
                bat_cycle_loop += 1
                print("block 4")
                counter = 4

                SOC = x[1] / C_bat


     if x[0] <= P_load_hour:

        if x[1] == E_bat_min:
            P_grid_delta = (x[0] - P_load_hour)

            E_bat_delta = 0

            x[1] += (E_bat_delta)
            #E_bat_loop_profile[i + 1] = E_bat_loop
            print("block 5")
            counter = 5


            SOC = x[1] / C_bat

        else:
            if  (x[0]) >= (P_load_hour + E_bat_min - x[1]):
                E_bat_delta = (x[0] - P_load_hour)
                x[1] += (E_bat_delta)
                #E_bat_loop_profile[i + 1] = E_bat_loop
                P_grid_delta = 0
                bat_cycle_loop += 1
                print("block 6")
                counter = 6
                SOC = x[1] / C_bat
            else:
                E_bat_delta = (-x[1] + E_bat_min)
                x[1] += (E_bat_delta)
                #E_bat_loop_profile[i + 1] = E_bat_loop
                P_grid_delta =  (x[0] - P_load_hour) - E_bat_delta
                bat_cycle_loop += 1
                print("block 7")
                counter = 7
                SOC = x[1] / C_bat

     print(E_bat_delta)
     print(P_grid_delta)

     return (x[0] * epex_spot_price[i] - E_bat_delta*(bat_operational_cost) - P_grid_delta * (grid_operational_cost + epex_spot_price[i]))




 def inequality_constraint1(x):

  return (x[0] - P_load_hour - C_bat - 1000)

 def inequality_constraint2(x):

  return (-x[0] + P_load_hour - C_bat - 2000)

 def inequality_constraint3(x):

  return (E_bat_min - x[1])

 def inequality_constraint4(x):

  return (x[1] - E_bat_max)


# Constraint: SOC should be less than and greater than SOC max and SOC min respectively




 P_pv_watt = (PV_forecasts[i]*Panel_area)

 print(P_pv_watt)


# Bounds: Weights are between 0 and 1
 bounds = [(P_pv_watt*0.5, P_pv_watt*1.5), (E_bat_min, E_bat_max)]

# Initial guess for portfolio weights
 initial_P_pv_watt = float(P_pv_watt)



 constraints = [{'type': 'ineq', 'fun': inequality_constraint1}, {'type': 'ineq', 'fun': inequality_constraint2},{'type': 'ineq', 'fun': inequality_constraint3}, {'type': 'ineq', 'fun': inequality_constraint4} ]



 initial_guess = [initial_P_pv_watt, E_bat_loop_profile[i]]


# Solve the stochastic optimization problem
 result = minimize(objective, initial_guess,  constraints = constraints, bounds = bounds)

 E_bat_loop_profile[i+1] = result.x[1]
 P_pv_watt_opt[i] = result.x[0]



# Optimal portfolio weights
 optimal_values = result.x
 print("Optimal Solution:", optimal_values)
 print(optimal_values[0]/ P_pv_watt)
 print(i)


plt.plot(time_scale, E_bat_loop_profile)
#plt.plot(time_scale, P_pv_watt_opt)
#plt.plot(time_scale, PV_forecasts*Panel_area)
plt.show()
