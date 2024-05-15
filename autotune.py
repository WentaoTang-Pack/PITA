import numpy as np
import scipy.linalg as la
import numbers
from response import *
import cvxpy as cp
import gurobipy

def two_to_one(i, j):
    return int((1 + i) * i / 2 + j)
def one_to_two(k):
    i = np.floor((-1 + np.sqrt(1+8*k))/2)
    j = k - (1 + i) * i / 2
    return (int(i), int(j))


class AutoTune:
    def __init__(self, A, B, C, T=100, dt=1.0, diffusion=100):
        self.nx = A.shape[0]
        self.nu = B.shape[1]
        self.ny = C.shape[0]
        self.T = T
        self.dt = dt
        self.diffusion = diffusion
        self.BslashA = -la.solve(B.T @ B, B.T @ A)
        # Processing the diffusion horizon
        if (diffusion is None) or (diffusion == 0):
            self.diffusion = T 
            self.times = np.arange(T+1) 
        else:
            self.diffusion = diffusion
            self.times = np.zeros(T+1)
            for t in range(T+1):
                self.times[t] = np.maximum(float(t), np.ceil((self.diffusion + 1.0)**(float(t)/self.T)) - 1.0) 
        # Time discretization
        self.A = [np.zeros((self.nx, self.nx))] * self.T
        for t in range(T):
            self.A[t] = la.expm((self.times[t+1] - self.times[t]) * self.dt * A)
        self.B = [np.zeros((self.nx, self.nu))] * self.T
        for t in range(T):
            for i in range(1, 5):
                self.B[t] += (self.times[t+1] - self.times[t]) * self.dt / 5 * la.expm((self.times[t+1] - self.times[t]) * self.dt * i/5 * A) @ B 
            self.B[t] += (self.times[t+1] - self.times[t]) * self.dt / 10 * (B + la.expm((self.times[t+1] - self.times[t]) * self.dt * A) @ B) 
        self.C = C
        # Create the response complexes
        self.response_reference = ResponseComplex(nx=self.nx, nu=self.nu, ny=self.ny, T=self.T, dt=self.dt, times=self.times, C=self.C)
        self.response_proximal = ResponseComplex(nx=self.nx, nu=self.nu, ny=self.ny, T=self.T, dt=self.dt, times=self.times, C=self.C)
        self.response_tuning  = ResponseComplex(nx=self.nx, nu=self.nu, ny=self.ny, T=self.T, dt=self.dt, times=self.times, C=self.C)
        self.response_MPC = ResponseComplex(nx=self.nx, nu=self.nu, ny=self.ny, T=self.T, dt=self.dt, times=self.times, C=self.C)
        # Create tuning and constriants
        self.MPC_Q = None # should be a list of matrices 
        self.MPC_R = None # should be a list of matrices 
        self.MPC_constraints = [] # a list of (time-invariant) path constraints 
        self.tuned = False # Whether tuning has been done
        self.proximal_constraints = [] 

    def set_MPC_weights(self, Q, R):
        if isinstance(Q, list):
            self.MPC_Q = Q
        elif isinstance(Q, numbers.Number):
            self.MPC_Q = [Q * np.eye(self.nx)] * (self.T+1)
        else:
            self.MPC_Q = Q * (self.T+1)
        if isinstance(R, list):
            self.MPC_R = R
        elif isinstance(R, numbers.Number):
            self.MPC_R = [R * np.eye(self.nu)] * self.T
        else:
            self.MPC_R = R * self.T
    def MPC_set_MPC_constraints(self, path_constraints):
        # Valid syntax: [('x', '2', '>', '9.5')]
        self.MPC_constraints = path_constraints
    def MPC_active_constraints(self, response, scenario=None):
        x = []
        u = []
        for t in range(self.T + 1):
            xx = np.zeros(self.nx)
            uu = np.zeros(self.nu)
            for i in range(t+1):
                xx += np.dot(response.x_response[(t, i)], scenario.scenario[i]) 
                uu += np.dot(response.u_response[(t, i)], scenario.scenario[i]) 
            x.append(xx)
            u.append(uu)
        activeness = []
        for t in range(self.T+1):
            trueval = []
            if (self.MPC_constraints is not None) and (len(self.MPC_constraints) > 0):
                for item in self.MPC_constraints:
                    if item[0] == 'x':
                        if item[2] == '<':
                            trueval.append(x[t][item[1]] >= item[3])
                        elif item[2] == '>':
                            trueval.append(x[t][item[1]] <= item[3])
                    elif item[0] == 'u':
                        if item[2] == '<':
                            trueval.append(u[t][item[1]] >= item[3])
                        elif item[2] == '>':
                            trueval.append(u[t][item[1]] <= item[3])
            activeness.append(trueval)
        return activeness, x, u
            
    def MPC_simulate_response(self, scenario=None, initialization_magnitude=1.0, disturbance_magnitude=0.25):
        print('\n\n NOW SIMULATING MPC. \n')
        if scenario is None:
            scenario = Scenario(self.nx, self.T, initialization_magnitude=initialization_magnitude, disturbance_magnitude=disturbance_magnitude)
        scenario = scenario.scenario
        # Variables 
        x = cp.Variable((self.T + 1, self.nx))
        u = cp.Variable((self.T, self.nu))
        y = cp.Variable((self.T + 1, self.ny))
        # Response satisfies the model constraints
        model_constraints = []
        for t in range(self.T):   
            model_constraints.append(x[t+1, :] == self.A[t] @ x[t, :] + self.B[t] @ u[t, :])
        for t in range(self.T+1):
            model_constraints.append(y[t, :] == self.C @ x[t, :])
        # Response satisfies the path constraints
        path_constraints = []
        if self.MPC_constraints is not None:
            for t in range(self.T+1):
                for item in self.MPC_constraints:
                    if item[0] == 'x':
                        if item[2] == '<':
                            path_constraints.append(x[t, item[1]] <= item[3])
                        elif item[2] == '>':
                            path_constraints.append(x[t, item[1]] >= item[3])
                    if item[0] == 'u' and t < self.T:
                        if item[2] == '<':
                            path_constraints.append(u[t, item[1]] <= item[3])
                        elif item[2] == '>':
                            path_constraints.append(u[t, item[1]] >= item[3])
                    if item[0] == 'y':
                        if item[2] == '<':
                            path_constraints.append(y[t, item[1]] <= item[3])
                        elif item[2] == '>':
                            path_constraints.append(y[t, item[1]] >= item[3])
        # initial condition
        objective = cp.sum([(self.times[t+1] - self.times[t]) * (cp.quad_form(x[t+1, :], self.MPC_Q[t+1]) + cp.quad_form(u[t, :], self.MPC_R[t])) for t in range(self.T)])
        # Recursively solve the problem with receding horizon
        x_val = np.zeros((self.T + 1, self.nx))
        u_val = np.zeros((self.T, self.nu))
        y_val = np.zeros((self.T + 1, self.ny))
        x_current = scenario[0]
        for t in range(self.T):
            if int(t) % 10 == 0:
                print('MPC simulation at time %s.' % self.times[t])
            constraint_initial = [(x[0, :] == x_current)]
            prob = cp.Problem(cp.Minimize(objective), model_constraints + path_constraints + constraint_initial)
            prob.solve(solver=cp.GUROBI, verbose=False)
            y_current = self.C @ x_current
            u_current = u.value[0, :]
            x_val[t, :], y_val[t, :], u_val[t, :] = x_current, y_current, u_current
            x_current = self.A[t] @ x_current + self.B[t] @ u_current + scenario[t+1]
        x_val[self.T, :] = x_current
        y_val[self.T, :] = self.C @ x_current
        # Make the trajectory plots 
        fig = plt.figure()
        plt.subplot(3,1,1)
        for ix in range(self.nx):
            plt.plot(np.array(self.times) * self.dt, [x_val[t, ix] for t in range(self.T+1)], marker='o')
        plt.xlim((0, self.times[-1] * self.dt)), plt.ylabel(r'$x_t$')
        plt.legend([r'$x_%s$' % (ix+1) for ix in range(self.nx)], loc='upper right')
        plt.subplot(3,1,2)
        for iy in range(self.ny):
            plt.plot(np.array(self.times) * self.dt, [y_val[t, iy] for t in range(self.T)] + [0.], marker='o')
        plt.xlim((0, self.times[-1] * self.dt)), plt.ylabel(r'$y_t$')
        plt.legend([r'$y_%s$' % (iy+1) for iy in range(self.ny)], loc='lower right')
        plt.subplot(3,1,3)
        for iu in range(self.nu):
            plt.plot(np.array(self.times) * self.dt, [u_val[t, iu] for t in range(self.T)] + [0.], marker='o')
        plt.xlim((0, self.times[-1] * self.dt)), plt.xlabel(r'$t$'), plt.ylabel(r'$u_t$')
        plt.legend([r'$u_%s$' % (iu+1) for iu in range(self.nu)], loc='lower right')
        plt.show()
        return fig
            
    def set_reference(self, x_to_w_initial = 1.0, u_to_w_initial = 1.0, closed_loop_time = 10.0):
        if isinstance(u_to_w_initial, numbers.Number):
            u_to_w_initial = u_to_w_initial * self.BslashA
            print(u_to_w_initial)
        self.response_reference.create_reference(x_to_w_initial, u_to_w_initial, closed_loop_time)
    def set_reference_penalty(self, x_weights = 1.0, u_weights = 1.0, y_weights = 0.0):
        if isinstance(x_weights, numbers.Number):
            x_weights = np.array([x_weights] * self.nx)
        self.response_reference_penalty_x = np.array(x_weights)
        if isinstance(u_weights, numbers.Number):
            u_weights = np.array([u_weights] * self.nu)
        self.response_reference_penalty_u = np.array(u_weights)
        if isinstance(y_weights, numbers.Number):
            y_weights = np.array([y_weights] * self.ny)
        self.response_reference_penalty_y = np.array(y_weights)
    def set_proximal_constraints(self, proximal_constraints):
        self.proximal_constraints = proximal_constraints
    def set_tuning_penalty(self, x_weights = 1.0, u_weights = 1.0, y_weights = 0.0):
        if isinstance(x_weights, numbers.Number):
            x_weights = np.array([x_weights] * self.nx)
        self.response_tuning_penalty_x = np.array(x_weights)
        if isinstance(u_weights, numbers.Number):
            u_weights = np.array([u_weights] * self.nu)
        self.response_tuning_penalty_u = np.array(u_weights)
        if isinstance(y_weights, numbers.Number):
            y_weights = np.array([y_weights] * self.ny)
        self.response_tuning_penalty_y = np.array(y_weights)
        
    def find_proximal(self, norm_option=1, x0=None, plotting=True):
        print('\n\n NOW FINDING A RESPONSE CLOSE TO USER SPECS. \n\n')
        T = self.T
        Phi_x, Phi_u, Phi_y = {}, {}, {}
        for t in range(T+1):
            for i in range(t+1):
                Phi_x[(t, i)] = cp.Variable((self.nx, self.nx))
                Phi_u[(t, i)] = cp.Variable((self.nu, self.nx))
                Phi_y[(t, i)] = cp.Variable((self.ny, self.nx))
        constraints = []
        # Model conformity
        for t in range(T+1):
            constraints.append(Phi_x[(t, t)] == np.eye(self.nx))
            if t < T:
                for i in range(t+1):
                    constraints.append(Phi_x[(t+1, i)] == self.A[t] @ Phi_x[(t, i)] + self.B[t] @ Phi_u[(t, i)])
            for i in range(t+1):
                constraints.append(Phi_y[(t, i)] == self.C @ Phi_x[(t, i)])
        # All user-specified desirable shaping
        for item in self.proximal_constraints:
            # Polarity constraint: ('polarity', 'x', x_index, w_index, '+')
            if item[0] == 'polarity': 
                if item[1] == 'x':
                    if item[4] == '+':
                        for t in range(T+1):
                            for i in range(t+1):
                                constraints.append(Phi_x[(t, i)][item[2], item[3]] >= 0)
                    elif item[4] == '-':
                        for t in range(T+1):
                            for i in range(t+1):
                                constraints.append(Phi_x[(t, i)][item[2], item[3]] <= 0)
                elif item[1] == 'u':
                    if item[4] == '+':
                        for t in range(T+1):
                            for i in range(t+1):
                                constraints.append(Phi_u[(t, i)][item[2], item[3]] >= 0)
                    elif item[4] == '-':
                        for t in range(T+1):
                            for i in range(t+1):
                                constraints.append(Phi_u[(t, i)][item[2], item[3]] <= 0)
                elif item[1] == 'y':
                    if item[4] == '+':
                        for t in range(T+1):
                            for i in range(t+1):
                                constraints.append(Phi_y[(t, i)][item[2], item[3]] >= 0)
                    elif item[4] == '-':
                        for t in range(T+1):
                            for i in range(t+1):
                                constraints.append(Phi_y[(t, i)][item[2], item[3]] <= 0)
            # Convergence constraint: ('convergence', 'x', x_index, w_index, tau)
            if item[0] == 'convergence':
                if item[1] == 'x':
                    for t in range(item[4], T+1):
                        for i in range(0, t-item[4]):
                            constraints.append(cp.abs(Phi_x[(t, i)][item[2], item[3]]) <= 0.05)
                elif item[1] == 'u':
                    for t in range(item[4], T+1):
                        for i in range(0, t-item[4]):
                            constraints.append(cp.abs(Phi_u[(t, i)][item[2], item[3]]) <= 0.05)
                elif item[1] == 'y':
                    for t in range(item[4], T+1):
                        for i in range(0, t-item[4]):
                            constraints.append(cp.abs(Phi_y[(t, i)][item[2], item[3]]) <= 0.05)
            # Overshoot constraint: ('overshoot', 'x', x_index, w_index, ub)
            if item[0] == 'overshoot':
                if item[1] == 'x':
                    for t in range(T+1):
                        for i in range(t+1):
                            constraints.append(cp.abs(Phi_x[(t, i)][item[2], item[3]]) <= item[4])
                elif item[1] == 'u':
                    for t in range(T+1):
                        for i in range(t+1):
                            constraints.append(cp.abs(Phi_u[(t, i)][item[2], item[3]]) <= item[4])
                elif item[1] == 'y':
                    for t in range(T+1):
                        for i in range(t+1):
                            constraints.append(cp.abs(Phi_y[(t, i)][item[2], item[3]]) <= item[4])
            # Smoothness constraint: ('smoothness', 'x', x_index, w_index, delta)
            if item[0] == 'smoothness':
                if item[1] == 'x':
                    for t in range(T):
                        for i in range(t+1):
                            constraints.append(cp.abs(Phi_x[(t+1, i)][item[2], item[3]] - Phi_x[(t, i)][item[2], item[3]]) <= item[4])
                elif item[1] == 'u':
                    for t in range(T):
                        for i in range(t+1):
                            constraints.append(cp.abs(Phi_u[(t+1, i)][item[2], item[3]] - Phi_u[(t, i)][item[2], item[3]]) <= item[4])
                elif item[1] == 'y':
                    for t in range(T):
                        for i in range(t+1):
                            constraints.append(cp.abs(Phi_y[(t+1, i)][item[2], item[3]] - Phi_y[(t, i)][item[2], item[3]]) <= item[4])
            # Locality constraint: ['locality', 'x', x_index, w_index, whatever]
            if item[0] == 'locality':
                if item[1] == 'x':
                    for t in range(T+1):
                        for i in range(t+1):
                            constraints.append(Phi_x[(t, i)][item[2], item[3]] == 0)
                elif item[1] == 'u':
                    for t in range(T+1):
                        for i in range(t+1):
                            constraints.append(Phi_u[(t, i)][item[2], item[3]] == 0)
                elif item[1] == 'y':
                    for t in range(T+1):
                        for i in range(t+1):
                            constraints.append(Phi_y[(t, i)][item[2], item[3]] == 0)
        # Define the objective
        wx = np.array([ [self.response_reference_penalty_x[i]] * self.nx for i in range(self.nx) ])
        wu = np.array([ [self.response_reference_penalty_u[i]] * self.nx for i in range(self.nu) ])
        wy = np.array([ [self.response_reference_penalty_y[i]] * self.nx for i in range(self.ny) ])
        time_double_indices = [(t, i) for t in range(T+1) for i in range(t+1)]
        obj = cp.sum([cp.sum_squares(cp.multiply(Phi_x[ind] - self.response_reference.x_response[ind], wx)) + cp.sum_squares(cp.multiply(Phi_u[ind] - self.response_reference.u_response[ind], wu)) + cp.sum_squares(cp.multiply(Phi_y[ind] - self.response_reference.y_response[ind], wy)) for ind in time_double_indices])
        # Solve and plot
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.GUROBI, verbose=True)
        for t in range(T+1):
            for i in range(t+1):
                self.response_proximal.x_response[(t, i)] = Phi_x[(t, i)].value
                self.response_proximal.u_response[(t, i)] = Phi_u[(t, i)].value
                self.response_proximal.y_response[(t, i)] = Phi_y[(t, i)].value 
        if (x0 is not None) and (plotting): 
            fig = self.response_proximal.plot_response_to_initialization(x0)
            return fig
        else:
            return None
        
    def find_tuning(self, scenarios, norm_option=2, simulate_scenario=None):
        print('\n\n NOW AUTOMATICALLY TUNING CONTROLLER. \n\n')
        activeness = []
        x = []
        u = []
        num_scenario = len(scenarios)
        # Create dual variables. Index: path_dual[(s, t)] has length equal to the number of active constraints
        path_dual = {}
        for s in range(num_scenario):
            activeness_this_scenario, x_this_scenario, u_this_scenario = self.MPC_active_constraints(self.response_proximal, scenario=scenarios[s])
            activeness.append(activeness_this_scenario)
            x.append(x_this_scenario)
            u.append(u_this_scenario)
            path_dual_temp = []
            for t in range(self.T + 1):
                if activeness_this_scenario[t].count(True) > 0:
                    path_dual[(s, t)] = cp.Variable(activeness_this_scenario[t].count(True), nonneg=True)
                else:
                    path_dual[(s, t)] = np.array([0.])
        # Create dual variables for the model constraints. 
        model_dual = {}
        for s in range(num_scenario):
            for t in range(self.T):
                model_dual[(s, t)] = cp.Variable(self.nx)
        # Create tuning variables and their bounds
        Q, R, QT = cp.Variable(self.nx, nonneg=True), cp.Variable(self.nu, nonneg=True), cp.Variable(self.nx, nonneg=True)
        tuning_parameter_constraints = [Q >= np.ones(self.nx), R >= np.ones(self.nu), QT >= np.ones(self.nx)]
        # Gather the information for objective
        w_before_time, w_to_time, mu_matrix_x, mu_matrix_u = {}, {}, {}, {} 
        for s in range(num_scenario):
            activeness_this_scenario, x_this_scenario, u_this_scenario = activeness[s], x[s], u[s]
            for t in range(self.T+1):
                w_before_time[(s, t)] = np.concatenate([scenarios[s].scenario[i] for i in range(t)] + [np.zeros(self.nx)], axis=None) 
                w_to_time[(s, t)] = np.concatenate([scenarios[s].scenario[i] for i in range(t+1)], axis=None) 
                if activeness_this_scenario[t].count(True) > 0:
                    mu_matrix_x[(s, t)] = np.zeros((activeness_this_scenario[t].count(True), self.nx))
                    mu_matrix_u[(s, t)] = np.zeros((activeness_this_scenario[t].count(True), self.nu))
                else:
                    mu_matrix_x[(s, t)] = np.zeros((1, self.nx))
                    mu_matrix_u[(s, t)] = np.zeros((1, self.nu))
                count_path_active = 0
                for count in range(len(activeness_this_scenario[t])):
                    if activeness_this_scenario[t][count] == True:
                        if self.MPC_constraints[count][0] == 'x':
                            mu_matrix_x[(s, t)][count, self.MPC_constraints[count][1]] = (1.0 if self.MPC_constraints[count][2]=='<' else -1.0)
                        elif self.MPC_constraints[count][0] == 'u':
                            mu_matrix_u[(s, t)][count, self.MPC_constraints[count][1]] = (1.0 if self.MPC_constraints[count][2]=='<' else -1.0)
        # Create the objective function based on all scenarios 
        if norm_option == 2:
            obj = cp.sum([(cp.sum([(self.times[t+1] - self.times[t])**2 * (
                cp.sum_squares(cp.outer(cp.multiply(Q, x[s][t]), w_to_time[(s, t)]) 
                        + cp.outer(self.A[t].T @ model_dual[(s, t)], w_to_time[(s, t)]) 
                        - cp.outer(model_dual[(s, t-1)], w_before_time[(s, t)]) 
                        + cp.outer(mu_matrix_x[(s, t)].T @ path_dual[(s, t)], w_to_time[(s, t)])) 
                + cp.sum_squares(cp.outer(cp.multiply(R, u[s][t]), w_to_time[(s, t)]) 
                          + cp.outer(self.B[t].T @ model_dual[(s, t)], w_to_time[(s, t)]) 
                          + cp.outer(mu_matrix_u[(s, t)].T @ path_dual[(s, t)], w_to_time[(s, t)])) ) for t in range(1, self.T)]) 
                           + (self.times[1] - self.times[0])**2 * (
                               cp.sum_squares(cp.outer(self.A[0].T @ model_dual[(s, 0)], w_to_time[(s, 0)])) 
                               + cp.sum_squares(cp.outer(cp.multiply(R, u[s][0]), w_to_time[(s, 0)]) 
                                         + cp.outer(self.B[0].T @ model_dual[(s, 0)], w_to_time[(s, 0)]) 
                                         + cp.outer(mu_matrix_u[(s, 0)].T @ path_dual[(s, 0)], w_to_time[(s, 0)])))
                           + (self.times[-1] - self.times[-2])**2 * cp.sum_squares(
                               cp.outer(cp.multiply(QT, x[s][self.T]), w_to_time[(s, self.T)]) 
                               - cp.outer(model_dual[(s, self.T-1)], w_before_time[(s, self.T)]) 
                               + cp.outer(mu_matrix_x[(s, self.T)].T @ path_dual[(s, self.T)], w_to_time[(s, self.T)]))) for s in range(num_scenario)])
        elif norm_option == 1:
            obj = cp.sum([(cp.sum([(self.times[t+1] - self.times[t])**2 * (
                cp.norm(cp.outer(cp.multiply(Q, x[s][t]), w_to_time[(s, t)]) 
                        + cp.outer(self.A[t].T @ model_dual[(s, t)], w_to_time[(s, t)]) 
                        - cp.outer(model_dual[(s, t-1)], w_before_time[(s, t)]) 
                        + cp.outer(mu_matrix_x[(s, t)].T @ path_dual[(s, t)], w_to_time[(s, t)]), 1) 
                + cp.norm(cp.outer(cp.multiply(R, u[s][t]), w_to_time[(s, t)]) 
                          + cp.outer(self.B[t].T @ model_dual[(s, t)], w_to_time[(s, t)]) 
                          + cp.outer(mu_matrix_u[(s, t)].T @ path_dual[(s, t)], w_to_time[(s, t)]), 1) ) for t in range(1, self.T)]) 
                           + (self.times[1] - self.times[0])**2 * (
                               cp.norm(cp.outer(self.A[0].T @ model_dual[(s, 0)], w_to_time[(s, 0)]), 1) 
                               + cp.norm(cp.outer(cp.multiply(R, u[s][0]), w_to_time[(s, 0)]) 
                                         + cp.outer(self.B[0].T @ model_dual[(s, 0)], w_to_time[(s, 0)]) 
                                         + cp.outer(mu_matrix_u[(s, 0)].T @ path_dual[(s, 0)], w_to_time[(s, 0)]), 1))
                           + (self.times[-1] - self.times[-2])**2 * cp.norm(
                               cp.outer(cp.multiply(QT, x[s][self.T]), w_to_time[(s, self.T)]) 
                               - cp.outer(model_dual[(s, self.T-1)], w_before_time[(s, self.T)]) 
                               + cp.outer(mu_matrix_x[(s, self.T)].T @ path_dual[(s, self.T)], w_to_time[(s, self.T)]), 1)) for s in range(num_scenario)])
        else:
            obj = 0.
        # Solve 
        prob = cp.Problem(cp.Minimize(obj), tuning_parameter_constraints)
        prob.solve(solver=cp.GUROBI, verbose=True)
        print('Q = ', Q.value)
        print('R = ', R.value)
        print('QT = ', QT.value)
        self.MPC_Q, self.MPC_R = [], []
        for t in range(self.T):
            self.MPC_Q.append(np.diag(Q.value))
            self.MPC_R.append(np.diag(R.value))
        self.MPC_Q.append(np.diag(QT.value))
        self.tuned = True
        if simulate_scenario is not None:    
            fig = self.MPC_simulate_response(scenario=simulate_scenario)
            return fig