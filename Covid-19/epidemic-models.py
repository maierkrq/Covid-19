# epidemic-models.py
# created by Kristina Maier 
# on November 30, 2022
# edited on Dezember 03, 2022
# changes: p0 in line 304 and plotting style 

import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

def get_initial_values():
    """
    Reads file "Fallzahlen_Gesamtuebersicht.xlsx" and returns content of second sheet as a dataframe 
    grouped by "Meldedatum" and "Bundesland_id" for all states in Germany.
    Input:
        -

    Output:
        pandas.core.frame.DataFrame
    """
    file_loc = "Fallzahlen_Gesamtuebersicht.xlsx" 
    xls = pd.ExcelFile(file_loc)
    df = pd.read_excel(xls, xls.sheet_names[1])
    df.columns = df.iloc[0]
    df.drop(df.index[0], inplace = True)
    df_grouped = df.groupby(["Meldedatum","Bundesland_id"], as_index=False).sum()
    return df_grouped

def SIR(sir, t, alpha, beta, N):
    """
    SIR model: S (susceptible), I (infected), R (removed).
    Input:
        sir:    array of S, I, R for time t
        t:      time point
        alpha:  infection rate
        beta:   rate of removed
        N:      population size

    Output:
        tuple
    """
    (S, I, R) = sir
    dSdt = -alpha * S * I / N
    dIdt = (alpha * S / N - beta) * I
    dRdt = beta * I
    return (dSdt, dIdt, dRdt)

def fit_SIR(t, alpha, beta):
    '''
    Fits the model "SIR". Returns solution of corresponding ODE. 
    Input:
        t:      array of time 
        alpha:  infection rate
        beta:   rate of removed

    Output:
        numpy.ndarray
    '''
    return odeint(SIR, SIR_0, t, args=(alpha, beta, N)).T.flatten()


def fit_SIR_infected(t, alpha, beta):
    '''
    Fits the model "SIR". Returns part of solution of corresponding ODE. 
    Input:
        t:      array of time 
        alpha:  infection rate
        beta:   rate of removed

    Output:
        numpy.ndarray
    '''
    return odeint(SIR, SIR_0, t, args=(alpha, beta, N))[:,1]

def SIR_patches(sir_patches, t, alpha, beta, N_i): 
    '''
    SIR model with three patches. Models spatial heterogeneity.
    Input:
        sir_patches:    array of all S, I, R for time t
        t:              time point
        alpha:          tuple of infection rates
        beta:           tuple of sums of recovery and death rate 
        N_i:            list of population sizes 

    Output:
        tuple
    '''
    (S_1, S_2, S_3, I_1, I_2, I_3, R_1, R_2, R_3) = sir_patches
    (beta_1, beta_2, beta_3) = beta
    (N_1, N_2, N_3) = N_i
    (alpha_11, alpha_12, alpha_13, alpha_21, alpha_22, alpha_23, alpha_31, alpha_32, alpha_33) = alpha

    i_1 = I_1/N_1
    i_2 = I_2/N_2
    i_3 = I_3/N_3

    lamda_1 = alpha_11 * i_1 + alpha_12 * i_2 + alpha_13 * i_3
    dS_1dt = - S_1 * lamda_1 
    dI_1dt = S_1 * lamda_1 - beta_1 * I_1
    dR_1dt = beta_1 * I_1

    lamda_2 = alpha_21 * i_1 + alpha_22 * i_2 + alpha_23 * i_3
    dS_2dt = - S_2 * lamda_2 
    dI_2dt = S_2 * lamda_2 - beta_2 * I_2
    dR_2dt = beta_2 * I_2

    lamda_3 = alpha_31 * i_1 + alpha_32 * i_2 + alpha_33 * i_3
    dS_3dt = - S_3 * lamda_3 
    dI_3dt = S_3 * lamda_3 - beta_3 * I_3
    dR_3dt = beta_3 * I_3
    return (dS_1dt, dS_2dt, dS_3dt, dI_1dt, dI_2dt, dI_3dt, dR_1dt, dR_2dt, dR_3dt)

def fit_SIR_patches(t, alpha_11, alpha_12, alpha_13, alpha_21, alpha_22, alpha_23, alpha_31, alpha_32, alpha_33, beta_1, beta_2, beta_3):
    '''
    Fits the model "SIR_patches". Returns part of solution of corresponding ODE. 
    Input:
        t:          array of time 
        alpha_ij:   infection rate, susceptible of patch P_i comes in contact with infected of patch P_j 
        beta_k:     rate of removed for patch P_k
    Output:
        numpy.ndarray
    '''
    beta = (beta_1, beta_2, beta_3) 
    alpha = (alpha_11, alpha_12, alpha_13, alpha_21, alpha_22, alpha_23, alpha_31, alpha_32, alpha_33) 
    return odeint(SIR_patches, SIR_0, t, args=(alpha, beta, N_i))[:,3:6].T.flatten() 

def SEIRD(seird, t, alpha, beta, gamma, delta, N):
    """
    SEIRD model: S (susceptible), E (exposed), I (infected), R (recovered), D (dead).
    Input:
        seird:  array of S, E, I, R, D for time t
        t:      time point
        alpha:  infection rate
        beta:   incubation rate
        gamma:  recovery rate
        delta:  death rate
        N:      population size

    Output:
        tuple
    """
    (S, E, I, R, D) = seird
    dSdt = -alpha * S * I / N
    dEdt = alpha * S * I / N - beta * E
    dIdt = beta * E - gamma * I - delta * I
    dRdt = gamma * I
    dDdt = delta * I
    return (dSdt, dEdt, dIdt, dRdt, dDdt)

'''
def fit_SEIRD(t, alpha, beta, gamma, delta):
    Fits the model "SEIRD". Returns solution of corresponding ODE. 
    Input:
        t:      array of time 
        alpha:  infection rate
        beta:   incubation rate
        gamma:  recovery rate
        delta:  death rate

    Output:
        numpy.ndarray
    
    # N = S+E+I+R+D
    return odeint(SEIRD, SEIRD_0, t, args=(alpha, beta, gamma, delta, N)).T.flatten()
'''


if __name__ == '__main__':
    num_days_to_model = 365 # how many days to model
    num_states = 3 # how many states to model
    initial_date = "2021-09-11" 
   
    initial_values = get_initial_values()
    dates = initial_values.Meldedatum
    initial_values = initial_values[(initial_values.Bundesland_id <=3) & 
                                    (dates >= initial_date) &
                                    (dates <= "2022-09-10")
    ]

    df_all_cases = pd.DataFrame(initial_values.Faelle_gesamt)
    df_population = pd.DataFrame(initial_values.Bevoelkerung)
    df_new_cases = initial_values.pivot(index='Bundesland_id', columns='Meldedatum', values='Faelle_neu')

    x_data = np.linspace(0, num_days_to_model-1, num_days_to_model) # time space, 1 point for each day
    I_i_0 = [0, 0, 0]
    R_i_0 = [0, 0, 0]
    S_i_0 = [0, 0, 0]
    N_i = [0, 0, 0]
    for state_nr in range(num_states):
        print("\n State Nr.", (state_nr+1))
        population_numbers = initial_values[initial_values.Bundesland_id==1].Bevoelkerung
        if len(population_numbers.unique()) == 1:
            N = df_population.iloc[state_nr,0] # constant population in state
            N_i[state_nr] = N
        else:
            print("Population number changes with time in state with ID ", (state_nr+1), ". Choosing to model the population constant.")
        I_0 = df_new_cases.iloc[state_nr,0] # initial number of infected people
        I_i_0[state_nr] = I_0
        total_infected = df_all_cases.iloc[state_nr,0]
        R_0 = total_infected - I_0 # initial number of removed people
        R_i_0[state_nr] = R_0
        S_0 = N - I_0 - R_0 # initial number of susceptible people
        S_i_0[state_nr] = S_0 # S_0 = N - I_0 - (total_infected - I_0) = N - total_infected

        I = np.array(df_new_cases.iloc[state_nr])
        totoal_infected = np.array(initial_values[initial_values.Bundesland_id == (state_nr+1)].Faelle_gesamt)
        R = totoal_infected - I
        S = N - I - R
        y_data = np.array([S,I,R]).flatten()
        SIR_0 = (S_0, I_0, R_0)

        # fitting all data
        popt, _ = curve_fit(fit_SIR, x_data, y_data, maxfev=1000)
        print("Parameters: ", popt) 
        fitted = fit_SIR(x_data, *popt).reshape((3, num_days_to_model))

        # plotting all data
        fsize = 45 #fontsize
        lwidth = 3
        fig1 = plt.figure(figsize=(20,10))
        #plt.title("SIR model for state ID {0}".format(state_nr+1)) 
        plt.plot(x_data, S, '-o', label="real data, S", linewidth=lwidth)
        plt.plot(x_data, I, '-o', label="real data, I", linewidth=lwidth)
        plt.plot(x_data, R, '-o', label="real data, R", linewidth=lwidth)
        plt.plot(x_data, fitted[0], label="fitted data, S", linewidth=lwidth)
        plt.plot(x_data, fitted[1], label="fitted data, I", linewidth=lwidth)
        plt.plot(x_data, fitted[2], label="fitted data, R", linewidth=lwidth)
        plt.ylabel("number of people", fontsize=fsize)
        plt.xlabel("days", fontsize=fsize)
        plt.legend(fontsize=fsize)
        y_ticks = plt.yticks()[0]
        plt.yticks(ticks=y_ticks, fontsize=fsize)
        x_ticks = plt.xticks()[0]
        plt.xticks(ticks=x_ticks, fontsize=fsize)
        plt.savefig("SIR-model-for-state-id-{0}.png".format(state_nr+1), bbox_inches='tight')
        plt.close(fig1)
        # What do you observe? -> The infection curve is too small in relation to the population size, 
        # but the fitting looks good from a distance.

        # plotting only infection curve
        fig2 = plt.figure(figsize=(20,10))
        #plt.title("SIR model for state ID {0}".format(state_nr+1))
        plt.plot(x_data, I, '-o', label="real data, I", linewidth=lwidth)
        plt.plot(x_data, fitted.reshape((3,365))[1], label="fitted data, I", linewidth=lwidth)
        plt.ylabel("number of infected", fontsize=fsize)
        plt.xlabel("days", fontsize=fsize)
        plt.legend(fontsize=fsize)
        y_ticks = plt.yticks()[0]
        plt.yticks(ticks=y_ticks, fontsize=fsize)
        x_ticks = plt.xticks()[0]
        plt.xticks(ticks=x_ticks, fontsize=fsize)
        plt.savefig("partial-plot-SIR-model-for-state-id-{0}.png".format(state_nr+1), bbox_inches='tight')
        plt.close(fig2)
        ## What do you observe? -> The fitted infection curve is bigger than the real data for infected. The fitting 
        # does not look good anymore up close. Also, the data points jump.
        # We could consider the average of 7 days to reduce the noise, i.e., take values of the last 7 days from data set and divide by 7.

        ## Explore possible ways to improve this result/fit.
        # Fitting and plotting only data for infected, i.e., concentrating only on infected when fitting.
        # Since the rates are constant, the fitting will improve when considering
        # infected only. 
        popt, _ = curve_fit(fit_SIR_infected, x_data, I, maxfev=1000)
        print("Parameters: infection rate ", popt[0]) #, ", removed ", popt[1]) 
        fitted = fit_SIR_infected(x_data, *popt)

        fig3 = plt.figure(figsize=(20,10))
        #plt.title("SIR model for state ID {0}".format(state_nr+1))
        plt.plot(x_data, I, '-o', label="real data, I", linewidth=lwidth)
        plt.plot(x_data, fitted, label="fitted data, I", linewidth=lwidth)
        plt.ylabel("number of infected", fontsize=fsize)
        plt.xlabel("days", fontsize=fsize)
        plt.legend(fontsize=fsize)
        y_ticks = plt.yticks()[0]
        plt.yticks(ticks=y_ticks, fontsize=fsize)
        x_ticks = plt.xticks()[0]
        plt.xticks(ticks=x_ticks, fontsize=fsize)
        plt.savefig("infected-SIR-model-for-state-id-{0}.png".format(state_nr+1), bbox_inches='tight')
        plt.close(fig3)

        # Further, the model is quite simple and the can be extended. Consider, e.g., exposed (E) or dead (D) individuals.
        # The removed individuals are therefore divided into dead and recovered ones.
        # My data set does not contain real data of dead and recovered individuals. Nevertheless, the model is implemented,
        # see function "SEIRD".


    ## What would a model that combines data from 2-3 or more states/places look like? Formulate possible couplings.
    # spatial heterogeneity in epidemical models
    # patches P_i, i=1,2,3
    # individual of P_i can travel to P_j (short visit, same amount of time for every individual)
    # N_i population of patch i
    # alpha_ij transmission coefficient between infected of patch P_j and susceptive of P_i
    # beta_i recovery rate
    # N_i = S_i + I_i + R_i
    # total metapopulation size is N = N_1 + N_2 + N_3

    SIR_0 = (*S_i_0, *I_i_0, *R_i_0)
    y_all_data = []
    for state_nr in range(num_states):
        y_data = np.array(df_new_cases.iloc[state_nr])
        y_all_data.append(y_data)
    y_all_data_new = np.array(y_all_data).flatten()
    popt, _ = curve_fit(fit_SIR_patches, x_data, y_all_data_new, bounds=(0, np.inf), p0=[popt[0]/3]*9 + [popt[1]]*3) # 03.12.22 change: added p0
    print("Parameters: ", popt)
    fitted = fit_SIR_patches(x_data, *popt).reshape((num_states,num_days_to_model))
    y_all_data_new_shaped = y_all_data_new.reshape((num_states,num_days_to_model))

    for state_nr in range(num_states):
        fig_new = plt.figure(figsize=(20,10))
        #plt.title("SIR model with three patches for state ID {0}".format(state_nr+1))
        plt.plot(x_data, y_all_data_new_shaped[state_nr,:], '-o', label="real data, I", linewidth=lwidth)
        plt.plot(x_data, fitted[state_nr,:], label="fitted data, I", linewidth=lwidth)
        plt.ylabel("number of infected", fontsize=fsize)
        plt.xlabel("days", fontsize=fsize)
        plt.legend(fontsize=fsize)
        y_ticks = plt.yticks()[0]
        plt.yticks(ticks=y_ticks, fontsize=fsize)
        x_ticks = plt.xticks()[0]
        plt.xticks(ticks=x_ticks, fontsize=fsize)
        plt.savefig("infected-SIR-model-patches-state-id-{0}.png".format(state_nr+1), bbox_inches='tight')
        plt.close(fig_new)


    # Combine all patches/state by summing population number etc.
    I_sum = np.sum(y_all_data_new.reshape((num_states,num_days_to_model)), axis=0)
    SIR_0 = (np.sum(S_i_0), np.sum(I_i_0), np.sum(R_i_0))

    popt, _ = curve_fit(fit_SIR_infected, x_data, I_sum, bounds=(0, np.inf), maxfev=1000)
    print("Parameters: ", popt)
    fitted = fit_SIR_infected(x_data, *popt)

    fig_4 = plt.figure(figsize=(20,10))
    #plt.title("SIR model with three patches for all states with ID {0}, {1}, {2}".format(*range(num_states)))
    plt.plot(x_data, I_sum, '-o', label="real data, I", linewidth=lwidth)
    plt.plot(x_data, fitted, label="fitted data, I", linewidth=lwidth)
    plt.ylabel("number of infected", fontsize=fsize)
    plt.xlabel("days", fontsize=fsize)
    plt.legend(fontsize=fsize)
    y_ticks = plt.yticks()[0]
    plt.yticks(ticks=y_ticks, fontsize=fsize)
    x_ticks = plt.xticks()[0]
    plt.xticks(ticks=x_ticks, fontsize=fsize)
    plt.savefig("infected-SIR-model-patches-all-states-id-{0}-{1}-{2}.png".format(*range(num_states)), bbox_inches='tight')
    plt.close(fig_4)