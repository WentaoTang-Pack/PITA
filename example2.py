import numpy as np
import scipy.linalg as la
import cvxpy as cp
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import gurobipy
from response import * 
from autotune import * 

## EXAMPLE II - Distillation model (Sec. 12.4, Skogestad & Postlethwaite)
# Inputs: feed rate, feed composition; Outputs: y_D, x_B
A = np.diag([-0.005131, -0.07366, -0.1829, -0.4620, -0.4620])
A[3, 4] = 0.9895
A[4, 3] = -0.9895
B = np.array([[-0.629, 0.624], [0.055, -0.172], [0.030, -0.108], [-0.186, -0.139], [-1.23, -0.056]])
C = np.array([[-0.7223, -0.517, 0.3386, -0.1633, 0.1121], [-0.8913, 0.4728, 0.9876, 0.8425, 0.2186]])
nx, nu, ny = A.shape[1], B.shape[1], C.shape[0]
K = -C @ la.solve(A, B) 
print('Static gain G(0) =', K)
# x0 = np.random.uniform(low=-1.0, high=1.0, size=nx) 
# x0 = x0 / la.norm(x0) * np.sqrt(float(nx)) 
x0 = -np.dot(la.solve(A, B), la.solve(K, np.array([1, -1.5])))
print('x0 = ', x0)

## DEFINE THE AUTOTUNING 
at = AutoTune(A=A, B=B, C=C, T=60, dt=0.5, diffusion=None)
print('A = ', at.A[0])
print('B = ', at.B[0])

## TEST MODEL PREDICTIVE CONTROLLER
at.set_MPC_weights(Q=1e0, R=1e-4)
path_constraints = []
# path_constraints += [('x', 2, '<', 9), ('u', 0, '>', -3)]
at.MPC_set_MPC_constraints(path_constraints=path_constraints)
simulate_scenario = Scenario(nx=nx, T=at.T, x0=x0, randomness=True, disturbance_magnitude=0.075)
# fig_initial = at.MPC_simulate_response(scenario=simulate_scenario) 

## PHASE I
at.set_reference(x_to_w_initial=1.0, u_to_w_initial=0.0, closed_loop_time=5.0/at.dt)
at.set_reference_penalty(x_weights=1.0, u_weights=0.0, y_weights=0.0)
at.set_proximal_constraints(proximal_constraints=[])
fig_prox = at.find_proximal(norm_option=2, x0=x0, plotting=True)

## PHASE II
at.set_tuning_penalty(x_weights=1.0, u_weights=1.0, y_weights=0.0)
number_scenarios = 12
tuning_scenarios = []
for i in range(number_scenarios):
    y_initial = np.array([np.cos(2*np.pi*i/12), np.sin(2*np.pi*i/12)]) 
    x_initial = -np.dot(la.solve(A, B), la.solve(K, y_initial)) 
    tuning_scenarios.append(Scenario(nx=nx, T=at.T, randomness=False, x0=x_initial))
at.find_tuning(scenarios=tuning_scenarios, norm_option=1, simulate_scenario=simulate_scenario)