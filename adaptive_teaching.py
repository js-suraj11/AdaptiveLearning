import numpy as np
from utils import*


import numpy as np

def compute_h_i(theta_i, n_i):
    """
    Compute the value of h_i (half-life of concept 'i').

    Parameters:
        theta_i (1d-array): The value(s) of theta_i (retention rate of the learner for concept 'i').
        n_i (1d-array): The value(s) of n_i (number of correct and incorrect recalls in history).

    Returns:
        float: The computed value(s) of h_i.

    """
    h_i = 2 ** (np.dot(theta_i, n_i))
    return h_i



def compute_n_i(i, sigma_t, y_t):
    """
    Compute the values of n_i_plus, n_i_minus, and 1 using the given parameters.

    Parameters:
        i (String): The value of i (Concept).
        sigma_t (1d-array): The values of concepts till time 't'.
        y_t (1d-array): The value of recalls till time 't' for concept 'i'.

    Returns:
        np.1darray: An array containing n_i_plus, n_i_minus, and 1.

    """
    n_i_plus = correct_recalls(i, sigma_t, y_t)
    n_i_minus = incorrect_recalls(i, sigma_t, y_t)
    return np.array([n_i_plus, n_i_minus, 1])



def compute_g_i(i, tau, sigma_t, y_t, theta_i):
    l_i = last_time_of_i(i, sigma_t)
    n_i = compute_n_i(i, sigma_t, y_t)
    h_i = compute_h_i(theta_i, n_i)

    if l_i == 0:
        g_i = 0
    else:
        g_i = 1/(2**((tau-l_i)/h_i))
    return g_i


def objective_f(sigma_t, y_t, n, T, theta_i, concepts, t):
    sum_g_i = 0
    for j in concepts:
        for tau in range(1, T+1):
            time = find_min(tau, t+1)
            sum_g_i += compute_g_i(j, tau+1,
                                   sigma_t[:time], y_t[:time], theta_i)
    f_value = (sum_g_i)/(n*T)
    return f_value


def compute_multiple_f(sigma_updated, y_updated, n, T, theta_i, concepts, n_y, t):
    f_list = []
    for j in range(len(n_y)):
        f = objective_f(
            sigma_updated, y_updated[j], n, T, theta_i, concepts, t+1)
        f_list.append(f)
    return f_list


def compute_multiple_g_i(i, tau, sigma_updated, y_updated, theta_i, n_y):
    p_list = []
    for j in range(len(y_updated)):
        p_list.append(compute_g_i(
            i, tau, sigma_updated, y_updated[j], theta_i))
    return p_list


def delta_i(i, sigma_t, y_t, n, T, theta_i, concepts, n_y, t):
    f_t = objective_f(sigma_t, y_t, n, T, theta_i, concepts, t)
    sigma_updated, y_updated = update_history(sigma_t, y_t, i, n_y)
    f_t_n_y = compute_multiple_f(
        sigma_updated, y_updated, n, T, theta_i, concepts, n_y, t)
    diff = compute_difference(f_t, f_t_n_y)
    tau = t+1
    p = compute_multiple_g_i(i, tau, sigma_t, y_updated, theta_i, n_y)
    E = compute_expected_value(diff, p)
    return E


def argmax_i(sigma_t, y_t, n, concepts, T, theta, n_y, t):
    delta_list = []
    for j in range(len(concepts)):
        delta_list.append(
            delta_i(concepts[j], sigma_t, y_t, n, T, theta[j], concepts, n_y, t))
    max_idx = delta_list.index(max(delta_list))
    i_t = concepts[max_idx]
    return i_t

def epsgreeedy_i(sigma_t, y_t, n, concepts, T, theta, n_y, t,epsilon):
    delta_list = np.zeros(n)
    prob =np.zeros(n)
    for j in range(len(concepts)):
        delta_list[j] =delta_i(concepts[j], sigma_t, y_t, n, T, theta[j], concepts, n_y, t)
    max_idx = np.argmax(delta_list)
    
    for j in range(len(concepts)):
        if j == max_idx:
            prob[j] = 1-epsilon
        else:
            prob[j] = epsilon/((len(concepts)-1))
    i_t = concepts[np.random.choice(range(len(concepts)),p=prob)]
    return i_t

