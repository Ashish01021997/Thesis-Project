from linopy import Model
import numpy as np
import random as r
import pandas as pd
import RFalgo
import matplotlib.pyplot as plt

PV_forecaster = RFalgo.PV_forecasting()

SOC_min = 10 / 100
SOC_max = 90 / 100

P_pv = PV_forecaster.y_pred
#P_pv = np.array(P_pv)

df = pd.read_csv('Germany Electricity Consumption.csv', delimiter=';')
P_load = df['Total']*1e6
No_of_households = 41.6 * 1e6
print(P_load/No_of_households)
m = Model()
#P_pv_watt = m.add_variables(lower= 50, upper= 1300, name= 'P_pv_watt')
#P_load_hour = m.add_variables(lower=1000, upper= 100000, name='P_load_hour')

C_bat = 200
#C_bat = 6500
x_bat = m.add_variables(lower= 0, upper= 1, name= 'x_bat')
y_grid = m.add_variables(lower= 0, upper= 1, name= 'y_grid')
#E_bat_delta = m.add_variables(lower= 0, upper= 160, name= 'E_bat_delta')
Panel_area = 1.6
counter = 0
# panel area in square metre

E_bat_min = SOC_min * C_bat
E_bat_max = SOC_max * C_bat
#E_bat = m.add_variables(lower= E_bat_min, upper= E_bat_max, name= 'E_bat')
E_bat_init = E_bat_min + (E_bat_max - E_bat_min) * 0.5
E_bat = E_bat_min + (E_bat_max - E_bat_min) * 0.5
P_load_opt = []
P_pv_watt_opt = []
time_scale = np.arange(1, 25, 1)
E_bat_delta_profile = np.zeros(24)
bat_cycle = 0

for i in range(24):

    # Panel_area = 500

    #P_load_hour = float(P_load[i] / No_of_households)

     # for DSM

    cost_pv = 0.5
    cost_bat = 0.7
    cost_grid = 0.8
    grid_op = 1
    bat_op = 0.5
    bat_degradation_cost = 0.75
    cost_panel = 100.0
    cost_battery = 200.0
    a = float(P_pv[i])
    print(a)
    P_pv_watt = (a*Panel_area)
    P_load_hour = float(P_load [i] /No_of_households)
    print(P_load_hour)



    P_grid_delta = 0.0

    if P_pv_watt == P_load_hour:
        P_grid_delta = 0
        E_bat_delta = 0
        E_bat = (E_bat_delta) + E_bat_init

        print("block 1")
        counter = 1
        #m.add_objective(P_pv_watt * cost_pv, overwrite='True')
        #SOC = E_bat / C_bat


        # m.add_constraints(SOC >= SOC_min)
        # m.add_constraints(SOC <= SOC_max)
        # m.add_constraints(E_bat >= E_bat_min)
        # m.add_objective(P_pv_watt*cost_pv)
        # m.solve()
        # print(C_bat)
        # print(Panel_area)

    if P_pv_watt >= P_load_hour:

        if E_bat_init == E_bat_max:
            P_grid_delta = (P_pv_watt - P_load_hour)*y_grid
            E_bat_delta = 0
            E_bat = (E_bat_delta) + E_bat_init

            print("block 2")
            counter = 2

            m.add_constraints((P_grid_delta ) * 0.9 <= (-P_load_hour + P_pv_watt))
            m.add_objective(- P_grid_delta * cost_grid  , overwrite='True')
            #SOC = E_bat / C_bat

        else:
            if (- E_bat_init - P_pv_watt + E_bat_max) >= (- P_load_hour):

                E_bat_delta = (P_pv_watt - P_load_hour)*x_bat
                E_bat = (E_bat_delta) + E_bat_init
                P_grid_delta = 0
                bat_cycle += bat_cycle
                print("block 3")
                counter = 3
                m.add_constraints(E_bat <= E_bat_max)
                m.add_constraints(E_bat >= E_bat_min)
                m.add_constraints(( E_bat_delta) * 0.9 <= (-P_load_hour + P_pv_watt))
                m.add_objective( - E_bat_delta * cost_bat + bat_cycle*bat_degradation_cost*x_bat, overwrite='True')
                #SOC = E_bat / C_bat

            else:
                E_bat_delta = (- E_bat_init + E_bat_max)*x_bat
                E_bat = (E_bat_delta) + E_bat_init

                P_grid_delta =  - E_bat_delta +  (P_pv_watt- P_load_hour)*y_grid
                bat_cycle += bat_cycle
                print("block 4")
                counter = 4
                m.add_constraints(E_bat <= E_bat_max)
                m.add_constraints(E_bat >= E_bat_min)
                m.add_constraints((P_grid_delta + E_bat_delta) * 0.9 <= (-P_load_hour + P_pv_watt))
                m.add_objective( - E_bat_delta * cost_bat - P_grid_delta * cost_grid + bat_cycle*bat_degradation_cost*x_bat ,
                                overwrite='True')
                #SOC = E_bat / C_bat


    if P_pv_watt <= P_load_hour:

        if E_bat_init == E_bat_min:
            P_grid_delta = (P_pv_watt - P_load_hour)*y_grid

            E_bat_delta = 0
            E_bat = (E_bat_delta) + E_bat_init
            print("block 5")
            counter = 5

            m.add_constraints((P_load_hour * 0.9 - P_pv_watt) <= (- P_grid_delta ))
            m.add_objective( - P_grid_delta * cost_grid  , overwrite='True')
            #SOC = E_bat / C_bat

        else:
            if (E_bat_init + P_pv_watt  - E_bat_min) >= (P_load_hour):
                E_bat_delta = (P_pv_watt - P_load_hour)*x_bat
                E_bat = (E_bat_delta) + E_bat_init
                P_grid_delta = 0
                bat_cycle += bat_cycle
                print("block 6")
                counter = 6
                m.add_constraints(E_bat <= E_bat_max)
                m.add_constraints(E_bat >= E_bat_min)
                m.add_constraints((P_load_hour * 0.9 - P_pv_watt) <= (- E_bat_delta))
                m.add_objective( - E_bat_delta * cost_bat + bat_cycle*bat_degradation_cost*x_bat  , overwrite='True')
                #SOC = E_bat / C_bat
            else:
                E_bat_delta = (-E_bat_init + E_bat_min)*x_bat
                E_bat = (E_bat_delta) + E_bat_init

                P_grid_delta =  (P_pv_watt - P_load_hour)*y_grid - E_bat_delta
                bat_cycle += bat_cycle
                print("block 7")
                counter = 7
                m.add_constraints(E_bat <= E_bat_max)
                m.add_constraints(E_bat >= E_bat_min)
                m.add_constraints((P_load_hour * 0.9 - P_pv_watt) <= (- P_grid_delta - E_bat_delta))
                m.add_objective(- E_bat_delta * cost_bat - P_grid_delta * cost_grid + bat_cycle*bat_degradation_cost*x_bat  ,
                                overwrite='True')


                #SOC = E_bat / C_bat



    if counter > 4:
        if counter == 7:
         #m.add_constraints((x_bat*bat_op + y_grid*grid_op ) <= -100) # 500 is the max grid delta +/-
            m.add_constraints((E_bat_delta*bat_op + P_grid_delta*grid_op) >= -1200) # consumer charge limit
        elif counter == 6:
            m.add_constraints((E_bat_delta * bat_op ) >= -70)
        elif counter == 5:
            m.add_constraints(( P_grid_delta * grid_op) >= -1000)
    else:
        if counter == 4:
            m.add_constraints((E_bat_delta*bat_op + P_grid_delta*grid_op) <= 1200)  # 500 is the max grid delta +/-
        elif counter == 3:
            m.add_constraints((E_bat_delta * bat_op) <= 70)
        elif counter == 2:
            m.add_constraints(( P_grid_delta * grid_op) <= 1000)
        #m.add_constraints((x_bat*bat_op + y_grid*grid_op ) >= 100)
    #m.add_constraints(P_grid_delta >= -1000)
    #m.add_constraints(P_grid_delta <= 1000)
    #m.add_constraints(P_pv_watt*cost_pv - E_bat_delta * cost_bat - P_grid_delta * cost_grid + Panel_area*cost_panel + C_bat*cost_battery >= 0)

    print(counter)
    print(E_bat)

    if (counter != 5) and (counter != 2):
     m.add_constraints(E_bat <= E_bat_max)
     m.add_constraints(E_bat >= E_bat_min)

    m.solve()

    #print(E_bat_delta.solution)

    print(y_grid.solution)
    print(x_bat.solution)

    if counter == 3:
        E_bat_delta_sol = (P_pv_watt - P_load_hour) * x_bat.solution
        print(E_bat_delta_sol)
    elif counter == 4:
        E_bat_delta_sol = (- E_bat_init + E_bat_max) * x_bat.solution
        print(E_bat_delta_sol)
    elif counter == 6:
        E_bat_delta_sol = (P_pv_watt - P_load_hour) * x_bat.solution
        print(E_bat_delta_sol)
    elif counter == 7:
        E_bat_delta_sol = (-E_bat_init + E_bat_min) * x_bat.solution
        print(E_bat_delta_sol)
    else:
        E_bat_delta = 0

    if i==0:
     E_bat_init = E_bat_delta_sol + 100
     E_bat_backup = E_bat_init
    else:
     E_bat_init = E_bat_delta_sol + E_bat_backup

    print(E_bat_init)
    E_bat_delta_profile[i] = E_bat_init
    print(i)






    #print(y.solution)
    #print(P_pv_watt.solution)
    #print(P_load_hour.solution)
    #P_load_opt[i] = P_load_hour.solution
    #P_pv_watt_opt[i] = P_pv_watt.solution
    # print(P_pv_watt * cost_pv - E_bat_delta * cost_bat - P_grid_delta * cost_grid)
#plt.plot(time_scale, P_load, 'r+')
#plt.plot(time_scale, P_load_opt, 'bo')
#plt.plot(time_scale, P_pv_predicted, 'r+')
plt.plot(time_scale, E_bat_delta_profile, color= "red")
plt.show()
# print(E_bat.solution)


















