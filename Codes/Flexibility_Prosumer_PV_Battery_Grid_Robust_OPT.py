import gurobipy as gp
import EMPricepointforecast
import matplotlib.pyplot as plt
import numpy as np
import Quantile_Load_forecasts
import Quantile_PV_forecasts
from numpy import random
import pandas as pd


C_bat = 4600 # in Wh
Panel_area = 2  # 2msq panel area (1.937 sq m from a article : (https://www.skillstg.co.uk/blog/solar-photovoltaic-panel-sizes/#:~:text=The%20physical%20dimensions%20of%20most,thickness%20of%20around%201.5%20inches)
counter = 0
SOC_min = 0.20
SOC_max = 0.95 # soc limits from the research paper (https://www.researchgate.net/publication/373737720_A_PROGNOSTIC_DECISION-MAKING_APPROACH_UNDER_UNCERTAINTY_FOR_AN_ELECTRIC_VEHICLE_FLEET_ROUTING_PROBLEM)
E_bat_min = SOC_min * C_bat
E_bat_max = SOC_max * C_bat
P_bat_min = 500 # 0.5kW
P_bat_max = 2500 # 2.5kW
P_bat_delta = P_bat_max - P_bat_min
E_bat_initial = (E_bat_max + E_bat_min)/2
E_bat_final = E_bat_initial
eta_batt = 0.85
P_grid_min = 1500
P_grid_max = 3000
P_grid_range = P_grid_max - P_grid_min
P_grid_initial = (P_grid_min + P_grid_max)/2

time_scale = np.arange(1, 6, 1) # 5 days is the simulation time-frame

n_steps = 24 # 24 hours in a day

iterations = len(time_scale)*n_steps



EM_Price_forecaster = EMPricepointforecast.EM_Price_point_forecast()

PV_forecaster = Quantile_PV_forecasts.PV_Quantiles()

Load_forecaster = Quantile_Load_forecasts.Load_Quantiles()

load_scaling_factor = 109/112.4 # https://www.cleanenergywire.org/factsheets/germanys-energy-consumption-and-power-mix-charts

EM_Price = EM_Price_forecaster.y_pred # for the year 2015 - LINK : https://ember-climate.org/data-catalogue/european-wholesale-electricity-price-data/

Load_hour_forecasts = Load_forecaster.df_predictions_load

Load_ten_percent_quantiles = Load_hour_forecasts['ten_percent_quantile']*load_scaling_factor

Load_ninety_percent_quantiles = Load_hour_forecasts['ninety_percent_quantile']*load_scaling_factor

PV_hour_forecasts = PV_forecaster.df_predictions_PV

PV_ten_percent_quantiles = PV_hour_forecasts['ten_percent_quantile']

PV_ninety_percent_quantiles = PV_hour_forecasts['ninety_percent_quantile']


offset = np.ones(iterations)*10.0 # offset is 10 EUR/MWh


bat_operational_cost = 5


model = gp.Model("linear_optimization_EMS")

E_bat_charging_profile = np.zeros(iterations)
P_grid_profile = np.zeros(iterations)
E_bat_profile = np.zeros(iterations)
P_grid_import_profile = np.zeros(iterations)
P_PV_export_grid_profile = np.zeros(iterations)
E_bat_discharging_profile = np.zeros(iterations)
E_bat_profile[0] = E_bat_initial

P_grid_current = np.ones(iterations)*P_grid_initial

E_bat_loop_current = np.zeros(iterations)

E_bat_loop_current[0] = E_bat_initial

epex_spot_import_price = np.zeros(iterations)
epex_spot_export_price = np.zeros(iterations)



# basic_optimization_loop():

for i in range(iterations):

    P_PV_export_grid = model.addVar(vtype=gp.GRB.CONTINUOUS,name="P_PV_export_grid")
    P_PV_export_grid_opt = np.zeros(iterations)
    P_bat_charging_PV = model.addVar(lb=P_bat_min, ub=P_bat_max, vtype=gp.GRB.CONTINUOUS, name="P_bat_charging_PV")
    P_bat_charging_PV_opt = np.zeros(iterations)
    P_bat_charging_grid = model.addVar(lb=P_bat_min, ub=P_bat_max, vtype=gp.GRB.CONTINUOUS, name="P_bat_charging_grid")
    P_bat_charging_grid_opt = np.zeros(iterations)
    P_bat_discharging_grid = model.addVar(lb=P_bat_min, ub=P_bat_max, vtype=gp.GRB.CONTINUOUS,
                                          name="P_bat_discharging_grid")
    P_bat_discharging_grid_opt = np.zeros(iterations)
    P_bat_discharging_load = model.addVar(lb=P_bat_min, ub=P_bat_max, vtype=gp.GRB.CONTINUOUS,
                                          name="P_bat_discharging_load")

    P_bat_discharging_load_opt = np.zeros(iterations)
    E_bat_discharging_load = np.zeros(iterations)

    P_bat_charging = model.addVar(lb=P_bat_min, ub=P_bat_max, vtype=gp.GRB.CONTINUOUS, name="P_bat_charging")
    P_bat_charging_opt = np.zeros(iterations)
    P_bat_discharging = model.addVar(lb=P_bat_min, ub=P_bat_max, vtype=gp.GRB.CONTINUOUS, name="P_bat_discharging")
    P_bat_discharging_opt = np.zeros(iterations)
    P_grid_import = model.addVar(lb=0, ub=P_grid_range, vtype=gp.GRB.CONTINUOUS, name="P_grid_import")
    P_grid_import_opt = np.zeros(iterations)

    P_grid_delta_net = model.addVar(vtype=gp.GRB.CONTINUOUS, name="P_grid_delta_net")

    epex_spot_import_price[i] = EM_Price[i] + offset[i]
    epex_spot_export_price[i] = EM_Price[i] - offset[i]

    P_load = model.addVar(vtype=gp.GRB.CONTINUOUS,
                            name="P_load") # tbc

    P_PV_gen = model.addVar( vtype=gp.GRB.CONTINUOUS,
                            name="P_PV_gen") # tbc



    P_bat_charging = (P_bat_charging_PV + P_bat_charging_grid)

    P_bat_discharging = (P_bat_discharging_load + P_bat_discharging_grid)

    P_grid_delta_net = P_grid_current[i]  + (P_PV_export_grid + P_bat_discharging_grid - P_bat_charging_grid - P_grid_import)   # j sum

    E_bat_loop = E_bat_loop_current[i] + (P_bat_charging*eta_batt - P_bat_discharging/eta_batt)


    model.addConstr((P_grid_delta_net) <= (P_grid_max), "constraint1")  # block i
    model.addConstr((P_grid_delta_net) >= (P_grid_min), "constraint2")
    model.addConstr((E_bat_loop) >= (E_bat_min ), "constraint3")
    model.addConstr((E_bat_loop) <= (E_bat_max ), "constraint4")

    model.addConstr(P_PV_export_grid + P_bat_charging_PV <= P_PV_gen, "constraint5")

    model.addConstr(P_PV_gen >= PV_ten_percent_quantiles[i], "constraint6")
    model.addConstr(P_PV_gen <= PV_ninety_percent_quantiles[i], "constraint7")
    model.addConstr(P_load >= Load_ten_percent_quantiles[i], "constraint8")
    model.addConstr(P_load <= Load_ninety_percent_quantiles[i], "constraint9")

    model.addConstr(
          ((P_bat_discharging + P_grid_import) - (P_bat_charging + P_PV_export_grid)) == (P_load - P_PV_gen),
          "constraint10")
       # j sum


    P_grid_delta = P_grid_import - P_bat_discharging_grid + P_bat_charging_grid


    battery_cycles = (P_bat_charging + P_bat_discharging)/P_bat_delta

    OBJ = P_grid_delta * epex_spot_import_price[i] - P_PV_export_grid * epex_spot_export_price[i] + battery_cycles*bat_operational_cost


    model.setObjective(OBJ, gp.GRB.MINIMIZE)

    model.optimize()



    if model.status == gp.GRB.OPTIMAL:
        print("Optimal solution found")
        print("Optimal objective value =", model.objVal)
        print(P_load)
        print(P_PV_gen)

        P_grid_current[i+1] = P_grid_current[i] - P_grid_import.x + P_PV_export_grid.x + P_bat_discharging_grid.x - P_bat_charging_grid.x # j sum

        E_bat_loop_current[i+1] = E_bat_loop_current[i] + (P_bat_charging_PV.x + P_bat_charging_grid.x)*eta_batt - (P_bat_discharging_load.x + P_bat_discharging_grid.x)/eta_batt # j sum
        P_PV_export_grid_opt[i] += P_PV_export_grid.x / n_steps
        P_bat_charging_PV_opt[i] += P_bat_charging_PV.x / n_steps
        P_bat_charging_grid_opt[i] += P_bat_charging_grid.x / n_steps
        P_bat_discharging_grid_opt[i] += P_bat_discharging_grid.x / n_steps
        P_bat_discharging_load_opt[i] += P_bat_discharging_load.x / n_steps
        P_grid_import_opt[i] += P_grid_import.x / n_steps
        P_bat_charging_opt[i] = P_bat_charging_PV_opt[i] + P_bat_charging_grid_opt[i]
        P_bat_discharging_opt[i] = P_bat_discharging_grid_opt[i] + P_bat_discharging_load_opt[i]
    else:
        print("No solution found")



    E_bat_profile[i] = E_bat_loop_current[i]
    P_grid_profile[i] = P_grid_current[i]
    P_grid_import_profile[i] = P_grid_import_opt[i]
    P_PV_export_grid_profile[i] = P_PV_export_grid_opt[i]

    E_bat_discharging_load[i] = P_bat_discharging_load_opt[i]*1 # 1 hour timescale


plt.plot(time_scale, E_bat_profile, marker='*', label='E_bat_optimum_flow')
plt.plot(time_scale, P_grid_profile, marker='.', label='P_grid_optimum_flow')
plt.plot(time_scale, P_grid_import_profile, marker='v', label='P_grid_import_optimum_flow')
plt.plot(time_scale, P_PV_export_grid_profile, marker='o', label='PV_export_grid_optimum_flow')
plt.show()

print("Flex Loop Begins...")

P_bat_positive_flex = np.zeros(len(time_scale)+1)
E_bat_positive_flex = np.zeros(len(time_scale)+1)
t_bat_positive_flex = 0

P_bat_negative_flex = np.zeros(len(time_scale)+1)
E_bat_negative_flex = np.zeros(len(time_scale)+1)
t_bat_negative_flex = 0

P_pv_positive_flex = np.zeros(len(time_scale)+1)
E_pv_positive_flex = np.zeros(len(time_scale)+1)
t_pv_positive_flex = 0

P_pv_negative_flex = np.zeros(len(time_scale)+1)
E_pv_negative_flex = np.zeros(len(time_scale)+1)
t_pv_negative_flex = 0

P_flex = np.zeros(len(time_scale)+1)
E_flex = np.zeros(len(time_scale)+1)
E_bat_flex_check = np.zeros(len(time_scale)+1)
E_bat_flex = np.zeros(len(time_scale)+1)

counter_bat_positive_flex = 0
counter_bat_negative_flex = 0
counter_PV_positive_flex = 0
counter_PV_negative_flex = 0

bat_positive_flex_price = 0
bat_negative_flex_price = 0
pv_positive_flex_price = 0
pv_negative_flex_price = 0

P_grid_current_flex = np.ones(120)*P_grid_initial


E_bat_loop_current_flex = np.zeros(120)

E_bat_loop_current_flex[0] = E_bat_initial
E_bat_profile_flex = np.zeros(120)
P_PV_export_grid_profile_flex = np.zeros(120)
P_grid_import_profile_flex = np.zeros(120)
P_grid_profile_flex = np.zeros(120)

E_bat_flex = np.zeros(120)
E_bat_flex[0] = E_bat_initial
P_grid_flex = np.zeros(120)
P_grid_flex[0] = P_grid_initial

epex_spot_import_price_flex = np.zeros(120)
epex_spot_export_price_flex = np.zeros(120)


battery_cycles_flex_i_count = np.zeros(120)

bat_neg_flex = 0  # switch to 1 or 0

pv_pos_flex = 0  # switch to 1 or 0

    # pv_neg_flex = PV_negative_flex()# switch to 1 or 0

pv_neg_flex = 0

bat_pos_flex = 0

model_flex = gp.Model("linear_flex_optimization_EMS")


for i in range(iterations):



  P_PV_export_grid_flex = model_flex.addVar( vtype=gp.GRB.CONTINUOUS,
                                              name="P_PV_export_grid_flex")
  P_PV_export_grid_opt_flex = np.zeros(120)
  P_bat_charging_PV_flex = model_flex.addVar(lb=P_bat_min, ub=P_bat_max, vtype=gp.GRB.CONTINUOUS,
                                               name="P_bat_charging_PV_flex")
  P_bat_charging_PV_opt_flex = np.zeros(120)
  P_bat_charging_grid_flex = model_flex.addVar(lb=P_bat_min, ub=P_bat_max, vtype=gp.GRB.CONTINUOUS,
                                                 name="P_bat_charging_grid_flex")
  P_bat_charging_grid_opt_flex = np.zeros(120)
  P_bat_discharging_grid_flex = model_flex.addVar(lb=P_bat_min, ub=P_bat_max, vtype=gp.GRB.CONTINUOUS,
                                                    name="P_bat_discharging_grid_flex")
  P_bat_discharging_grid_opt_flex = np.zeros(120)
  P_bat_discharging_load_flex = model_flex.addVar(lb=P_bat_min, ub=P_bat_max, vtype=gp.GRB.CONTINUOUS,
                                                    name="P_bat_discharging_load_flex")

  P_bat_discharging_load_opt_flex = np.zeros(120)
  E_bat_discharging_load_flex = np.zeros(120)

  P_bat_charging_flex = model_flex.addVar(lb=P_bat_min, ub=P_bat_max, vtype=gp.GRB.CONTINUOUS,
                                            name="P_bat_charging_flex")
  P_bat_charging_opt_flex = np.zeros(120)
  P_bat_discharging_flex = model_flex.addVar(lb=P_bat_min, ub=P_bat_max, vtype=gp.GRB.CONTINUOUS,
                                               name="P_bat_discharging_flex")
  P_bat_discharging_opt_flex = np.zeros(120)
  P_grid_import_flex = model_flex.addVar(lb=0, ub=P_grid_range, vtype=gp.GRB.CONTINUOUS, name="P_grid_import_flex")
  P_grid_import_opt_flex = np.zeros(120)

  P_load_flex = model.addVar(vtype=gp.GRB.CONTINUOUS,
                        name="P_load_flex")  # tbc

  P_PV_gen_flex = model.addVar(vtype=gp.GRB.CONTINUOUS,
                          name="P_PV_gen_flex")



  i_count_flex = 0




  def bat_positive_flex():  # battery charge from grid

      P_bat_positive_flex[i] = (P_bat_max - P_bat_charging_grid_opt[i])

      t_bat_positive_flex = 1

      E_bat_flex_check[i + 1] = E_bat_loop_current_flex[i] + P_bat_positive_flex[i] * t_bat_positive_flex

      for k in range(i + 1, 121, 1):

          if P_bat_positive_flex[k] > P_bat_positive_flex[i] and (E_bat_flex_check[i + 1] - E_bat_final) < sum(
                  E_bat_discharging_load[k] for k in range(i + 1, len(time_scale), 1)):

              t_bat_positive_flex += 1

      E_bat_positive_flex[i] = P_bat_positive_flex[i] * t_bat_positive_flex

      counter_bat_positive_flex = 1
      return P_bat_positive_flex[i]


  def bat_negative_flex():  # battery not charging from grid and discharging to grid

      P_bat_negative_flex[i] = - (P_bat_max - P_bat_discharging_opt[i] + P_bat_charging_grid_opt[i])

      t_bat_negative_flex = 1

      E_bat_flex_check[i + 1] = E_bat_loop_current_flex[i] + P_bat_negative_flex[i] * t_bat_negative_flex

      if (E_bat_final - E_bat_flex_check[i + 1]) <= E_bat_max * (len(time_scale) - 1 - (i + 1)):

          for k in range(i + 1, 121, 1):

              if abs(P_bat_charging_grid_opt[k] > abs(P_bat_charging_grid_opt[i])):
                  t_bat_negative_flex += 1

      E_bat_negative_flex[i] = P_bat_negative_flex[i] * t_bat_negative_flex

      return P_bat_negative_flex[i]


  def PV_positive_flex():  # curtailing  feed in from grid

      P_pv_positive_flex[i] = (P_PV_export_grid_opt[i])

      t_pv_positive_flex = 1

      for j in range(i + 1, 121, 1):

          if P_PV_export_grid_opt[j] >= P_pv_positive_flex[i]:
              t_pv_positive_flex += 1

      E_pv_positive_flex[i] = P_pv_positive_flex[i] * t_pv_positive_flex

      return P_pv_positive_flex[i]


  def PV_negative_flex():  # not charging battery from PV

      P_pv_negative_flex[i] = -P_bat_charging_PV_opt[i]

      t_pv_negative_flex = 1

      for j in range(i + 1, 121, 1):

          if abs(P_bat_charging_PV_opt[j] > abs(P_pv_negative_flex[i])):
              t_pv_negative_flex += 1

      E_pv_negative_flex[i] = P_pv_negative_flex[i] * t_pv_negative_flex

      return P_pv_negative_flex[i]


  # if E_bat_flex[i] < E_bat_max and E_bat_flex[i] > E_bat_min and P_grid_flex[i] < P_grid_max and P_grid_flex[i] > P_grid_min :

  # bat_pos_flex = bat_positive_flex() # switch to 1 or 0




  bat_neg_flex = bat_negative_flex()  # switch to 1 or 0

  counter_bat_negative_flex = 1

  pv_pos_flex = PV_positive_flex()

  counter_PV_positive_flex = 1

  pv_neg_flex = PV_negative_flex()

  counter_PV_negative_flex = 1

  bat_pos_flex = bat_positive_flex()

  counter_bat_positive_flex = 1

   # switch to 1 or 0

  # pv_neg_flex = PV_negative_flex()# switch to 1 or 0



  print("Bat_Pos_Flex", bat_pos_flex)

  print("Bat_Neg_Flex", bat_neg_flex)

  print("PV_pos_flex", pv_pos_flex)

  print("PV_neg_flex", pv_neg_flex)

  E_flex[i] = pv_neg_flex + pv_pos_flex + bat_neg_flex + bat_pos_flex

  t_flex = t_pv_negative_flex + t_pv_positive_flex + t_bat_negative_flex + t_bat_positive_flex

  print(t_flex)

  E_bat_flex[i] = E_bat_profile[i] + (
              bat_neg_flex + bat_pos_flex + pv_neg_flex) * t_flex  # deviation from optimal flow in battery

  P_grid_flex[i] = P_grid_profile[i] + (
              pv_pos_flex + bat_pos_flex + bat_neg_flex - pv_neg_flex)  # deviation from optimal flow in grid







  # OPTIMIZATION WITH THE FLEX VALUES

  # P_bat_discharging_grid += bat_neg_flex

  # P_bat_charging_grid += bat_pos_flex

  # P_bat_charging_PV += pv_neg_flex

  # P_PV_export_grid += pv_pos_flex



  import_price_flex = (bat_pos_flex) * 10
  export_price_flex = (bat_neg_flex + pv_pos_flex - pv_neg_flex) * 10

  epex_spot_import_price_flex[i] = epex_spot_import_price[i] + import_price_flex
  epex_spot_export_price_flex[i] = epex_spot_export_price[i] + export_price_flex


    #P_load_flex = model_flex.addVar(lb=Load_hour_forecasts_lower[i], ub=Load_hour_forecasts_upper[i], vtype=gp.GRB.CONTINUOUS,name="P_PV_gen")# only pv uncertainity # constant load array




  P_bat_charging_flex = (P_bat_charging_PV_flex + pv_neg_flex) + (P_bat_charging_grid_flex + bat_pos_flex)

  P_bat_discharging_flex = (P_bat_discharging_load_flex + P_bat_discharging_grid_flex + bat_neg_flex)


  P_grid_delta_net_flex = P_grid_current_flex[i] + P_PV_export_grid_flex - P_grid_import_flex  + P_bat_discharging_grid_flex - P_bat_charging_grid_flex + (bat_neg_flex + pv_pos_flex - pv_neg_flex -  bat_pos_flex)  # j sum

  E_bat_loop_flex = E_bat_loop_current_flex[i] + (P_bat_charging_flex * eta_batt - P_bat_discharging_flex / eta_batt)

  model_flex.addConstr((P_grid_delta_net_flex) <= (P_grid_max))  # block i
  model_flex.addConstr((P_grid_delta_net_flex) >= (P_grid_min))
  model_flex.addConstr((E_bat_loop_flex) >= (E_bat_min))
  model_flex.addConstr((E_bat_loop_flex) <= (E_bat_max))

  model_flex.addConstr(P_PV_gen_flex >= PV_ten_percent_quantiles[i])
  model_flex.addConstr(P_PV_gen_flex <= PV_ninety_percent_quantiles[i])
  model_flex.addConstr(P_load_flex >= Load_ten_percent_quantiles[i])
  model_flex.addConstr(P_load_flex <= Load_ninety_percent_quantiles[i])


  model_flex.addConstr(
            (P_bat_discharging_flex + P_grid_import_flex) - (P_bat_charging_flex + P_PV_export_grid_flex + (pv_pos_flex + pv_neg_flex)) == (P_load_flex - P_PV_gen_flex))
        # j sum

  P_grid_delta_flex = P_grid_import_flex - P_bat_discharging_grid_flex + P_bat_charging_grid_flex + (bat_pos_flex - bat_neg_flex + pv_pos_flex - pv_neg_flex)

  battery_cycles_flex = (P_bat_charging_flex + P_bat_discharging_flex) / P_bat_delta

    # Price Flex Calculation





  FLEX_OBJ = P_grid_delta_flex * epex_spot_import_price_flex[i] - (P_PV_export_grid_flex) * epex_spot_export_price_flex[
            i] + battery_cycles_flex * bat_operational_cost


  model_flex.setObjective(FLEX_OBJ, gp.GRB.MINIMIZE)


  model_flex.optimize()


  if model_flex.status == gp.GRB.OPTIMAL:
            print("Optimal solution found")
            print("Optimal objective value =", model_flex.objVal)
            P_grid_current_flex[i + 1] = P_grid_current_flex[
                                        i] - P_grid_import_flex.x + P_PV_export_grid_flex.x + pv_pos_flex - pv_neg_flex + P_bat_discharging_grid_flex.x + bat_neg_flex - P_bat_charging_grid_flex.x - pv_neg_flex

            E_bat_loop_current_flex[i + 1] = E_bat_loop_current_flex[i] + ((
                        P_bat_charging_PV_flex.x + pv_neg_flex + P_bat_charging_grid_flex.x + bat_pos_flex) * eta_batt - (
                                                    P_bat_discharging_load_flex.x + P_bat_discharging_grid_flex.x + bat_neg_flex) / eta_batt)/n_steps

            P_PV_export_grid_opt_flex[i] += P_PV_export_grid_flex.x / n_steps
            P_bat_charging_PV_opt_flex[i] += P_bat_charging_PV_flex.x / n_steps
            P_bat_charging_grid_opt_flex[i] += P_bat_charging_grid_flex.x / n_steps
            P_bat_discharging_grid_opt_flex[i] += P_bat_discharging_grid_flex.x / n_steps
            P_bat_discharging_load_opt_flex[i] += P_bat_discharging_load_flex.x / n_steps
            P_bat_charging_opt_flex[i] = P_bat_charging_PV_opt_flex[i] + P_bat_charging_grid_opt_flex[i]
            P_bat_discharging_opt_flex[i] = P_bat_discharging_grid_opt_flex[i] + P_bat_discharging_load_opt_flex[i]
            P_grid_import_opt_flex[i] += P_grid_import_flex.x / n_steps

            E_bat_discharging_load_flex[i] += P_bat_discharging_load_opt_flex[i] * 1
            i_count_flex += 1

            battery_cycles_flex_i_count[i] = (P_bat_charging_opt_flex[i] + P_bat_discharging_opt_flex[i]) / P_bat_delta

  else:
            print("No solution found")


  E_bat_profile_flex[i] = E_bat_loop_current_flex[i]

  P_grid_profile_flex[i] = P_grid_current_flex[i]

  RM = 0.2

  cost_per_cycle = 0.1  # 10 cents is the cost per battery charging/discharging


  def Bat_Pos_Flex_Price():

      C_bat_op_pos_flex = 0.5 * cost_per_cycle / battery_cycles_flex_i_count[i] * (P_bat_charging_opt_flex[i] / P_bat_max)

      savings_bat_pos_flex = P_bat_discharging_grid_opt_flex[i] * epex_spot_import_price_flex[
          i] + P_PV_export_grid_opt_flex[i] * epex_spot_export_price_flex[i] + (P_bat_charging_grid_opt[i] - P_bat_charging_grid_opt_flex[i])*epex_spot_import_price_flex[i]

      p_flex_bat_pos = (savings_bat_pos_flex * (1 - RM) + C_bat_op_pos_flex) / (bat_positive_flex() * t_bat_positive_flex)

      return p_flex_bat_pos


  def Bat_Neg_Flex_Price():

      C_bat_op_neg_flex = 0.5 * cost_per_cycle / battery_cycles_flex_i_count[i] * (P_bat_discharging_opt_flex[i] / P_bat_max)

      expenses = (P_grid_import_opt_flex[i] + P_bat_charging_grid_opt_flex[i] ) * epex_spot_import_price_flex[i] + P_bat_charging_PV_opt_flex[i] * epex_spot_export_price_flex[i]

      p_flex_bat_neg = (expenses * (1 + RM) + C_bat_op_neg_flex) / (bat_negative_flex() * t_bat_negative_flex)

      return p_flex_bat_neg


  def PV_Pos_Flex_Price():

      losses = P_bat_charging_PV_opt_flex[i] * epex_spot_import_price_flex[i] # Battery charging from PV at grid import price

      profit = P_PV_export_grid_opt_flex[i] * epex_spot_import_price_flex[i]

      if losses > profit :
          p_flex_pv_pos = losses*(1 + RM) - profit

      elif losses == profit :
          p_flex_pv_pos = losses*RM

      else :
          p_flex_pv_pos = profit - losses*(1 + RM)

      return p_flex_pv_pos


  def PV_Neg_Flex_Price():

      savings_pv_neg_flex = (P_bat_charging_PV_opt[i] - P_bat_charging_PV_opt_flex[i])*epex_spot_export_price_flex[i]

      p_flex_pv_neg = savings_pv_neg_flex * (1 + RM) / (PV_negative_flex() * t_pv_negative_flex)

      return p_flex_pv_neg

  Bat_Positive_Flex_Price = 0

  Bat_Negative_Flex_Price = 0

  PV_Positive_Flex_Price = 0

  PV_Negative_Flex_Price = 0



  if counter_bat_positive_flex == 1:

      Bat_Positive_Flex_Price = Bat_Pos_Flex_Price()

  if counter_bat_negative_flex == 1:

      Bat_Negative_Flex_Price = Bat_Neg_Flex_Price()

  if counter_PV_positive_flex == 1:

      PV_Positive_Flex_Price = PV_Pos_Flex_Price()

  if counter_PV_negative_flex == 1:
      
      PV_Negative_Flex_Price = PV_Neg_Flex_Price()

  print("Battery Positive Flex Price is", Bat_Positive_Flex_Price)

  print("Battery Negative Flex Price is", Bat_Negative_Flex_Price)

  print("PV Positive Flex Price is", PV_Positive_Flex_Price)

  print("PV Negative Flex Price is", PV_Negative_Flex_Price)





  print(i_count_flex)

plt.plot(np.arange(1, 121, 1), E_bat_profile_flex, marker='*', label='E_bat_flex_optimum_flow')
plt.plot(np.arange(1, 121, 1), P_grid_profile_flex, marker='.', label='P_grid_flex_optimum_flow')
plt.plot(np.arange(1, 121, 1), P_grid_import_opt_flex, marker='v', label='P_grid_flex_import_optimum_flow')
plt.plot(np.arange(1, 121, 1), P_PV_export_grid_opt_flex, marker='o', label='PV_export_flex_grid_optimum_flow')
plt.show()





















