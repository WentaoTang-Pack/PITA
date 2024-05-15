import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from random import sample
import numbers
import cvxpy as cp
import gurobipy

class ResponseComplex:
    def __init__(self, nx, nu, ny, T, times, dt, C):
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.T = T
        self.dt = dt
        self.C = C
        self.times = times
        # Registering x and u response matrix blocks
        self.x_response = {}
        self.u_response = {}
        self.y_response = {}
        for i in range(T+1):
            for j in range(i+1):
                if (i==0) and (j==0):
                    self.x_response[(i, j)] = np.eye(nx)
                else:
                    self.x_response[(i, j)] = np.zeros((nx, nx))
                self.u_response[(i, j)] = np.zeros((nu, nx))
                self.y_response[(i, j)] = self.C @ self.x_response[(i, j)]
    
    def create_reference(self, x_to_w_initial = 1.0, u_to_w_initial = 0.0, closed_loop_time = 10.0):
        for t in range(self.T+1):
            self.x_response[(t, t)] = x_to_w_initial * np.eye(self.nx)
            for t_after in range(t+1, self.T+1):
                if isinstance(x_to_w_initial, numbers.Number):
                    self.x_response[(t_after, t)] = x_to_w_initial * np.eye(self.nx) * np.exp((-self.times[t_after] + self.times[t])/closed_loop_time)
                else:
                    self.x_response[(t_after, t)] = x_to_w_initial * np.exp((-t_after + t)/closed_loop_time)
            if isinstance(u_to_w_initial, numbers.Number):
                self.u_response[(t, t)] = u_to_w_initial * np.ones((self.nu, self.nx))
            else:
                self.u_response[(t, t)] = u_to_w_initial
            for t_after in range(t+1, self.T+1):
                if isinstance(u_to_w_initial, numbers.Number):
                    self.u_response[(t_after, t)] = u_to_w_initial * np.ones((self.nu, self.nx)) * np.exp((-self.times[t_after] + self.times[t])/closed_loop_time)
                else:
                    self.u_response[(t_after, t)] = u_to_w_initial * np.exp((-t_after + t)/closed_loop_time)
            for t_after in range(t+1, self.T+1):
                self.y_response[(t_after, t)] = self.C @ self.x_response[(t_after, t)]

    def plot_response_to_initialization(self, x0):
        x_traj = np.zeros((self.T + 1, self.nx))
        y_traj = np.zeros((self.T + 1, self.ny))
        u_traj = np.zeros((self.T + 1, self.nu))
        for t in range(self.T + 1):
            x_traj[t, :] = np.dot(self.x_response[(t, 0)], x0)
            y_traj[t, :] = np.dot(self.y_response[(t, 0)], x0)
            u_traj[t, :] = np.dot(self.u_response[(t, 0)], x0)
        fig = plt.figure()
        u_traj_o = u_traj
        u_traj_o[-1, :] = np.nan
        plt.subplot(3, 1, 1), plt.plot(np.array(self.times) * self.dt, x_traj, marker='o'), 
        plt.xlim((0, self.times[-1] * self.dt)), plt.ylabel(r'$x_t$')
        plt.legend([r'$x_%s$' % (i+1) for i in range(self.nx)], loc='upper right')
        plt.subplot(3, 1, 2), plt.plot(np.array(self.times) * self.dt, y_traj, marker='o'),
        plt.xlim((0, self.times[-1] * self.dt)), plt.ylabel(r'$y_t$')
        plt.legend([r'$y_%s$' % (i+1) for i in range(self.ny)], loc='lower right')
        plt.subplot(3, 1, 3), plt.plot(np.array(self.times) * self.dt, u_traj_o, marker='o'),
        plt.xlim((0, self.times[-1] * self.dt)), plt.xlabel(r'$t$'), plt.ylabel(r'$u_t$') 
        plt.legend([r'$u_%s$' % (i+1) for i in range(self.nu)], loc='lower right')
        plt.show()
        return fig
                

                
class Scenario: 
    def __init__(self, nx, T, randomness=True, x0=None, initialization_magnitude=1.0, disturbance_magnitude=0.1):
        self.scenario = []
        self.nx = nx
        self.T = T
        if x0 is not None:
            self.scenario.append(np.array(x0))
        elif randomness:
            self.scenario.append(np.random.uniform(low=-initialization_magnitude, high=initialization_magnitude, size=nx))
        else:
            self.scenario.append(np.zeros(self.nx))
        for t in range(T):
            if randomness:
                self.scenario.append(np.random.uniform(low=-disturbance_magnitude, high=disturbance_magnitude, size=nx))
            else:
                self.scenario.append(np.zeros(self.nx))
    
    def plot_scenario(self):
        plt.figure()
        for ix in range(self.nx):
            plt.plot(range(self.T), [self.scenario[t+1][ix] for t in range(self.T)], marker='o')
        legendlist = [r'$w_%s$' % ix for ix in range(self.nx)]
        plt.xlabel(r'$t$'), plt.ylabel(r'$w_t$'), plt.legend(legendlist)
        plt.show()