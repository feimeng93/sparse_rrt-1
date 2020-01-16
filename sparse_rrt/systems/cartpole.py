import numpy as np
from sparse_rrt.systems.system_interface import BaseSystem

class Cartpole(BaseSystem):
    '''
    Cart Pole implemented by pytorch
    '''
    I = 10
    L = 2.5
    M = 10
    m = 5
    g = 9.8
    #  Height of the cart
    H = 0.5

    STATE_X = 0
    STATE_V = 1
    STATE_THETA = 2
    STATE_W = 3
    CONTROL_A = 0

    MIN_X = -30
    MAX_X = 30
    MIN_V = -40
    MAX_V = 40
    MIN_W = -2
    MAX_W = 2

    def enforce_bounds(self, temp_state):
        if temp_state[0] < self.MIN_X:
               temp_state[0] = self.MIN_X
        elif temp_state[0] > self.MAX_X:
               temp_state[0]=self.MAX_X
        if temp_state[1] < self.MIN_V:
                temp_state[1] = self.MIN_V
        elif temp_state[1] > self.MAX_V:
                temp_state[1] = self.MAX_V
        if temp_state[2] < -np.pi:
                temp_state[2] += 2 * np.pi
        elif temp_state[2] > np.pi:
                temp_state[2] -= 2 * np.pi
        if temp_state[3] < self.MIN_W:
                temp_state[3] = self.MIN_W
        elif temp_state[3] > self.MAX_W:
                temp_state[3] = self.MAX_W
        return temp_state


    def propagate(self, start_state, control, num_steps, integration_step):
        temp_state = start_state.copy()
        for _ in range(num_steps):
            deriv = self.update_derivative(temp_state, control)
            temp_state[0] += integration_step * deriv[0]
            temp_state[1] += integration_step * deriv[1]
            temp_state[2] += integration_step * deriv[2]
            temp_state[3] += integration_step * deriv[3]
            temp_state = self.enforce_bounds(temp_state).copy()
        return temp_state

    def visualize_point(self, state):
        x2 = state[self.STATE_X] + (self.L) * np.sin(state[self.STATE_THETA])
        y2 = -(self.L) * np.cos(state[self.STATE_THETA])
        return state[self.STATE_X], self.H, x2, y2

    def update_derivative(self, state, control):
        '''
        Port of the cpp implementation for computing state space derivatives
        '''
        I = self.I
        L = self.L
        M = self.M
        m = self.m
        g = self.g
        #  Height of the cart
        deriv = state.copy()
        temp_state = state.copy()
        _v = temp_state[self.STATE_V]
        _w = temp_state[self.STATE_W]
        _theta = temp_state[self.STATE_THETA]
        _a = control[self.CONTROL_A]
        mass_term = (self.M + self.m)*(self.I + self.m * self.L * self.L) - self.m * self.m * self.L * self.L * np.cos(_theta) * np.cos(_theta)

        deriv[self.STATE_X] = _v
        deriv[self.STATE_THETA] = _w
        mass_term = (1.0 / mass_term)
        deriv[self.STATE_V] = ((I + m * L * L)*(_a + m * L * _w * _w * np.sin(_theta)) + m * m * L * L * np.cos(_theta) * np.sin(_theta) * g) * mass_term
        deriv[self.STATE_W] = ((-m * L * np.cos(_theta))*(_a + m * L * _w * _w * np.sin(_theta))+(M + m)*(-m * g * L * np.sin(_theta))) * mass_term
        return deriv

    def get_state_bounds(self):
        return [(self.MIN_X, self.MAX_X),
                (self.MIN_V, self.MAX_V),
                (self.MIN_W, self.MAX_W)]

    def get_control_bounds(self):
        return None
