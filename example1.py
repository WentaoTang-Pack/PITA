import numpy as np
import scipy.linalg as la
import cvxpy as cp
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import gurobipy
from response import * 
from autotune import * 

## EXAMPLE I - Ex. 3.8, Skogestad & Postlethwaite
A = np.array([[-6, -2.5, 0, 0], [2, 0, 0, 0], [0, 0, -6, -2.5], [0, 0, 2, 0]])
B = np.array([[4, 0], [0, 0], [0, 2], [0, 0]])
C = np.array([[0, 5/8, 0, 5/4], [5/2, 5/8, 0, 5/2]])
nx, nu, ny = A.shape[1], B.shape[1], C.shape[0]
K = -C @ la.solve(A, B) 
print('\n Static gain G(0) =', K)
x0 = -np.dot(la.solve(A, B), la.solve(K, np.array([1.0, 0.5])))
print('x0 = ', x0)

## DEFINE THE AUTOTUNING 
at = AutoTune(A=A, B=B, C=C, T=60, dt=0.05, diffusion=None)
print('A = ', at.A[0])
print('B = ', at.B[0])

## TEST MODEL PREDICTIVE CONTROLLER
at.set_MPC_weights(Q=1.0, R=1.0)
path_constraints = []
# path_constraints += [('x', 2, '<', 9), ('u', 0, '>', -3)]
at.MPC_set_MPC_constraints(path_constraints=path_constraints)
simulate_scenario = Scenario(nx=nx, T=at.T, randomness=False, x0=x0)
# fig_initial = at.MPC_simulate_response(scenario=simulate_scenario) 

## PHASE I
at.set_reference(x_to_w_initial=1.0, u_to_w_initial=1.0, closed_loop_time=6.0)
at.set_reference_penalty(x_weights=1.0, u_weights=0.0, y_weights=0.0)
proximal_constraints = []
# proximal_constraints.append(('convergence', 'x', 1, 0, 59))
# proximal_constraints.append(('convergence', 'x', 1, 1, 59))
# proximal_constraints.append(('convergence', 'x', 1, 2, 59))
# proximal_constraints.append(('convergence', 'x', 1, 3, 59))
proximal_constraints.append(('smoothness', 'x', 0, 0, 0.1))
proximal_constraints.append(('smoothness', 'x', 0, 1, 0.1))
proximal_constraints.append(('smoothness', 'x', 0, 2, 0.1))
proximal_constraints.append(('smoothness', 'x', 0, 3, 0.1))
# proximal_constraints.append(('smoothness', 'u', 0, 0, 0.1))
# proximal_constraints.append(('smoothness', 'u', 0, 1, 0.1))
# proximal_constraints.append(('smoothness', 'u', 0, 2, 0.1))
# proximal_constraints.append(('smoothness', 'u', 0, 3, 0.1))
# proximal_constraints.append(('smoothness', 'u', 1, 0, 0.1))
# proximal_constraints.append(('smoothness', 'u', 1, 1, 0.1))
# proximal_constraints.append(('smoothness', 'u', 1, 2, 0.1))
# proximal_constraints.append(('smoothness', 'u', 1, 3, 0.1))
at.set_proximal_constraints(proximal_constraints=proximal_constraints)
fig_prox = at.find_proximal(norm_option=2, x0=x0, plotting=True)

## PHASE II
at.set_tuning_penalty(x_weights=1.0, u_weights=10.0, y_weights=10.0)
number_scenarios = 12
tuning_scenarios = []
for i in range(number_scenarios):
    y_initial = np.array([np.cos(2*np.pi*i/12), np.sin(2*np.pi*i/12)]) 
    x_initial = -np.dot(la.solve(A, B), la.solve(K, y_initial)) 
    tuning_scenarios.append(Scenario(nx=nx, T=at.T, randomness=False, x0=x_initial))
simulate_scenario = Scenario(nx=nx, T=at.T, randomness=False, x0=x0)
at.find_tuning(scenarios=tuning_scenarios, norm_option=1, simulate_scenario=simulate_scenario)