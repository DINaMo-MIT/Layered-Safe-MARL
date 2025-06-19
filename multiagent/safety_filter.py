from typing import List
from enum import Enum
import numpy as np
import jax.numpy as jnp
import hj_reachability as hj
from hj_reachability_utils.common import get_hj_grid_from_meta_data, \
                                         ControlAndDisturbanceAffineDynamics
import cvxpy as cp
from scipy.linalg import solve_continuous_lyapunov
from casadi import *
import pickle

from multiagent.config import DoubleIntegratorConfig, AirTaxiConfig

class Air4dCooperativeDynamics(ControlAndDisturbanceAffineDynamics):
    """ Relative dynamics of two kinematic vehicle (Dubins car + 4th state being velocity)
        Each vehicle dynamics is 4d but the relative dynamics is 5d.
    """
    def __init__(self, config_class: AirTaxiConfig,
                       control_mode="max"):
        self.v_min = config_class.V_MIN
        self.v_max = config_class.V_MAX
        self.accel_min = config_class.ACCEL_MIN
        self.accel_max = config_class.ACCEL_MAX
        self.angular_rate_max = config_class.ANGULAR_RATE_MAX
        # control: angular_rate_a, angular_rate_b, acceleration_a, acceleration_b
        control_space = hj.sets.Box(lo=np.array([-self.angular_rate_max, -self.angular_rate_max, self.accel_min, self.accel_min]),
                                    hi=np.array([self.angular_rate_max, self.angular_rate_max, self.accel_max, self.accel_max]))
        disturbance_space = hj.sets.Box(lo=np.zeros(5), hi=np.zeros(5))
        disturbance_mode= 'max' if control_mode == "min" else "max"

        self.control_space_va_min = hj.sets.Box(lo=np.array([-self.angular_rate_max, -self.angular_rate_max, 0, self.accel_min]),
                                    hi=np.array([self.angular_rate_max, self.angular_rate_max, self.accel_max, self.accel_max]))
        self.control_space_vb_min = hj.sets.Box(lo=np.array([-self.angular_rate_max, -self.angular_rate_max, self.accel_min, 0]),
                                    hi=np.array([self.angular_rate_max, self.angular_rate_max, self.accel_max, self.accel_max]))
        self.control_space_va_max = hj.sets.Box(lo=np.array([-self.angular_rate_max, -self.angular_rate_max, self.accel_min, self.accel_min]),
                                    hi=np.array([self.angular_rate_max, self.angular_rate_max, 0, self.accel_max]))
        self.control_space_vb_max = hj.sets.Box(lo=np.array([-self.angular_rate_max, -self.angular_rate_max, self.accel_min, self.accel_min]),
                                    hi=np.array([self.angular_rate_max, self.angular_rate_max, self.accel_max, 0]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        theta = state[..., 2]
        va = state[..., 3]
        vb = state[..., 4]
        dx1 = -va + vb * jnp.cos(theta)
        dx2 = vb * jnp.sin(theta)
        dx3 = jnp.zeros(theta.shape)
        dx4 = jnp.zeros(va.shape)
        dx5 = jnp.zeros(vb.shape)
        return jnp.array([dx1, dx2, dx3, dx4, dx5])

    def control_jacobian(self, state, time):
        return jnp.array([[state[..., 1], jnp.zeros(state[..., 1].shape), jnp.zeros(state[..., 1].shape), jnp.zeros(state[..., 1].shape)],
                          [-state[..., 0], jnp.zeros(state[..., 0].shape), jnp.zeros(state[..., 0].shape), jnp.zeros(state[..., 0].shape)],
                          [-jnp.ones(state[..., 0].shape), jnp.ones(state[..., 0].shape), jnp.zeros(state[..., 0].shape), jnp.zeros(state[..., 0].shape)],
                          [jnp.zeros(state[..., 0].shape), jnp.zeros(state[..., 0].shape), jnp.ones(state[..., 0].shape), jnp.zeros(state[..., 0].shape)],
                          [jnp.zeros(state[..., 0].shape), jnp.zeros(state[..., 0].shape), jnp.zeros(state[..., 0].shape), jnp.ones(state[..., 0].shape)]
                          ])
    
    def disturbance_jacobian(self, state, time):
        return jnp.eye(5)
    
    def optimal_control_and_disturbance(self, state, time, grad_value):
        """Computes the optimal control and disturbance realized by the HJ PDE Hamiltonian."""
        control_direction = grad_value @ self.control_jacobian(state, time)
        if self.control_mode == "min":
            control_direction = -control_direction

        control_interior = self.control_space.extreme_point(control_direction)
        control_va_min_bound = self.control_space_va_min.extreme_point(control_direction)
        control = jnp.where(state[..., 3] <= self.v_min, control_va_min_bound, control_interior)       
        control_va_max_bound = self.control_space_va_max.extreme_point(control_direction)
        control = jnp.where(state[..., 3] >= self.v_max, control_va_max_bound, control)
        control_vb_min_bound = self.control_space_vb_min.extreme_point(control_direction)
        control = jnp.where(state[..., 4] <= self.v_min, control_vb_min_bound, control)
        control_vb_max_bound = self.control_space_vb_max.extreme_point(control_direction)
        control = jnp.where(state[..., 4] >= self.v_max, control_vb_max_bound, control)

        disturbance_direction = grad_value @ self.disturbance_jacobian(state, time)
        if self.disturbance_mode == "min":
            disturbance_direction = -disturbance_direction
        return (control, self.disturbance_space.extreme_point(disturbance_direction))

class DoubleIntegratorDynamics(ControlAndDisturbanceAffineDynamics):
    """
    Represents the dynamics of a double integrator system.

    Parameters:
    - speed_a (float): Speed of the vehicle, 2X1 vector array [speedx, speedy]?
    - accelx_min (float): The min acceleration in the x-direction.
    - accelx_max (float): The maximum acceleration in the x-direction.
    - accely_min (float): The min acceleration in the y-direction.
    - accely_max (float): The maximum acceleration in the y-direction.

    Methods:
    - open_loop_dynamics(state, time): Computes the open-loop dynamics of the system.
    - control_jacobian(state, time): Computes the control Jacobian matrix.
    - disturbance_jacobian(state, time): Computes the disturbance Jacobian matrix.
    """
    def __init__(self, vx_min=-1.5, vx_max=1.5, vy_min=-1.5, vy_max=1.5, 
                 accelx_min=-1, accelx_max=1, accely_min=-1, accely_max=1, control_mode="max"):
        
        control_space = hj.sets.Box(lo=np.array([accelx_min, accelx_min,accely_min, accely_min]),
                                    hi=np.array([accelx_max, accelx_max, accely_max, accely_max]))
        disturbance_space = hj.sets.Box(lo=np.array([0,0,0,0]), hi=np.array([0,0,0,0]))
        # control_mode = 'min'
        disturbance_mode= 'max' if control_mode == "min" else "max"
        self.vx_min = vx_min
        self.vx_max = vx_max
        self.vy_min = vy_min
        self.vy_max = vy_max
        self.accelx_min = accelx_min
        self.accelx_max = accelx_max   
        self.accely_min = accely_min
        self.accely_max = accely_max  
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        "state array [x, y, vx, vy].T"
        return jnp.array([state[..., 2], state[..., 3], 0, 0])
    
    def control_jacobian(self, state, time):
        "control array [ax1, ay1, ax2, ay2].T"
        return jnp.array([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [1, 0, -1, 0],
                          [0, 1, 0, -1],
                          ])
    
    def disturbance_jacobian(self, state, time):
        return jnp.eye(4)

class SafetyFilterIndividualHandle:
    def __init__(self):
        pass
    
    def apply_safety_filter(self, 
                            ego_state: np.ndarray,
                            ego_action: np.ndarray, 
                            ego_waypoint: np.ndarray,
                            other_state_list: List, 
                            other_action_list: List,
                            other_waypoint_list: List):
        """
            ego_state: state of ego vehicle.
            ego_action: action of ego vehicle.
            other_state_list: list of states of other vehicles.
            other_action_list: list of actions of other vehicles.
        """
        action_filtered = False
        return ego_action, action_filtered

class HjDataHandle:
    def __init__(self, file_name, target_separation_distance):
        with open(file_name, 'rb') as f:
            hj_data_loaded = pickle.load(f)
        original_separation_distance = hj_data_loaded.info['separation_distance']
        shift = target_separation_distance - original_separation_distance
        if shift != 0:
            print(f"Separation distance in data is {original_separation_distance:.3f}, shifting value function by {shift:.3f} for target separation distance {target_separation_distance:.3f}.")
        self.grid_hj_meta_data = hj_data_loaded.grid_meta_data
        self.grid_hj = get_hj_grid_from_meta_data(self.grid_hj_meta_data)
        # saved value function is negative inside safe set so we negate it to use as CBF.
        # calibrate value function based on target separation distance
        self.values_hj = -hj_data_loaded.values - shift
        self.grads_hj = self.grid_hj.grad_values(self.values_hj)
        self.separation_distance = target_separation_distance

    def update_separation_distance(self, target_separation_distance):
        previous_separation_distance = self.separation_distance
        shift = target_separation_distance - previous_separation_distance
        self.values_hj -= shift
        self.separation_distance = target_separation_distance

class KinematicVehicleSafetyHandle(SafetyFilterIndividualHandle):
    def __init__(self, config_class: AirTaxiConfig, hj_data_handle: HjDataHandle):
        super(KinematicVehicleSafetyHandle, self).__init__()
        self.coordination_range = config_class.COORDINATION_RANGE
        self.v_min = config_class.V_MIN
        self.v_max = config_class.V_MAX
        self.accel_min = config_class.ACCEL_MIN
        self.accel_max = config_class.ACCEL_MAX
        self.angular_rate_max = config_class.ANGULAR_RATE_MAX
        self.dt = config_class.DT
        self.num_input = 2        
        self.hj_data_handle = hj_data_handle
        self.hj_dynamics = Air4dCooperativeDynamics(config_class)
        self.cbf_rate = config_class.CBF_RATE
        self.num_relative_state = 5
            
    def get_value_of_relative_state(self, ego_state, other_state):
        relative_state = self.get_relative_state(ego_state, other_state)
        try:
            value_at_relative_state = self.hj_data_handle.grid_hj.interpolate(self.hj_data_handle.values_hj, relative_state)
            state_in_hj_range = (not np.isnan(value_at_relative_state))
        except:
            state_in_hj_range = False
        if not state_in_hj_range:
            value_at_relative_state = np.inf
        return value_at_relative_state, state_in_hj_range
    
    def apply_safety_filter(self, ego_state: np.ndarray, ego_action: np.ndarray,
                            ego_waypoint: np.ndarray,
                            other_state_list: List, other_action_list: List,
                            other_waypoint_list: List):
        """ pseudo code:
            1. evaluate relative distances with other vehicles.
            2. Pick the other vehicle that is the minimum distance.
            3. If that distance is less than coordination range, apply hj safety filter.
        """
        relative_distances = []
        relative_values = []
        relative_states_in_hj_range = []
        for other_state in other_state_list:
            relative_distance = self.get_relative_distance(ego_state, other_state)
            relative_distances.append(relative_distance)
            relative_value, state_in_hj_range = self.get_value_of_relative_state(ego_state, other_state)
            relative_values.append(relative_value)
            relative_states_in_hj_range.append(state_in_hj_range)
        min_agent_index_by_distance = np.argmin(relative_distances)
        # min_agent_index = np.argmin(relative_distances)
        min_agent_index = np.argmin(relative_values)
        # print(f"min_agent_index: by_distance: {min_agent_index_by_distance}, by_value: {min_agent_index}, value at min_dist: {relative_values[min_agent_index_by_distance]:.2f}, at min_value: {relative_values[min_agent_index]:.2f}, dist at min_value: {relative_distances[min_agent_index]:.2f}, dist at min_dist: {relative_distances[min_agent_index_by_distance]:.2f}")
        min_relative_distance = relative_distances[min_agent_index_by_distance]
        if min_relative_distance > self.coordination_range:
            # action is not filtered if agents are beyond coordination range.
            return ego_action, False, min_agent_index
        
        pair_state = np.concatenate([ego_state, other_state_list[min_agent_index]])
        relative_state = self.get_relative_state(ego_state, other_state_list[min_agent_index])
        u_ref = np.zeros(2 * self.num_input)
        u_ref[:self.num_input] = ego_action
        u_ref[self.num_input:] = other_action_list[min_agent_index]
        eps_hj = 0.4
        # print("relative state: ", relative_state)
        value_at_relative_state = relative_values[min_agent_index]
        state_in_hj_range = relative_states_in_hj_range[min_agent_index]


        if not state_in_hj_range:
            # print(f"state out of hj range: {relative_state}")
            return ego_action, False, min_agent_index

        grad_at_relative_state = self.hj_data_handle.grid_hj.interpolate(self.hj_data_handle.grads_hj, relative_state)
          
        # crude least-restrictive reachability control.
        # print(f"value: {value_at_relative_state:.2f}")
        if value_at_relative_state < eps_hj:
            u = np.asarray(self.hj_dynamics.optimal_control(relative_state, None, grad_value=grad_at_relative_state)).copy()
        else:
            # use cbf constraint instead based on reachability value function.
            u = self.cbf_qp(relative_state, u_ref, value_at_relative_state, grad_at_relative_state)

        u = self.clip_ctrl_with_valid_control_bound(relative_state, u)
        # evaluate the norm diff between u and u_ref.
        u_norm_diff = float(np.linalg.norm(u - u_ref))
        filtered = u_norm_diff > 1e-4
        
        return u[:self.num_input], filtered, min_agent_index
        
    def clip_ctrl_with_valid_control_bound(self, relative_state, u_sol):
        dt = self.dt
        if self.num_input > 1:
            accel_max = self.accel_max if relative_state[3] < self.v_max - dt * self.accel_max else 0
            accel_min = self.accel_min if relative_state[3] > self.v_min - dt * self.accel_min else 0
            u_sol[1] = max(min(u_sol[1], accel_max), accel_min)
            accel_max = self.accel_max if relative_state[4] < self.v_max - dt * self.accel_max else 0
            accel_min = self.accel_min if relative_state[4] > self.v_min - dt * self.accel_min else 0
            u_sol[self.num_input+1] = max(min(u_sol[self.num_input+1], accel_max), accel_min)
        return u_sol

    @staticmethod
    def get_relative_distance(ego_state, other_state):
        return np.sqrt((other_state[0] - ego_state[0]) ** 2 + (other_state[1] - ego_state[1]) ** 2)
    
    @staticmethod
    def get_relative_state(ego_state, other_state):
        relative_distance = KinematicVehicleSafetyHandle.get_relative_distance(ego_state, other_state)
        relative_heading = other_state[2] - ego_state[2]
        relative_angle = np.arctan2(other_state[1] - ego_state[1], other_state[0] - ego_state[0])
        x_r = relative_distance * np.cos(relative_angle - ego_state[2])
        y_r = relative_distance * np.sin(relative_angle - ego_state[2])
        return np.array([x_r, y_r, relative_heading, ego_state[3], other_state[3]])
    
    def cbf_qp(self, x, u_ref, value, grad):
        u = cp.Variable(2 * self.num_input)
        if self.num_input == 1:
            if x[0] < 0:
                objective = cp.Minimize(cp.quad_form(u-u_ref, np.diag([10.0, 1.0])))
            else:
                objective = cp.Minimize(cp.quad_form(u-u_ref, np.diag([1.0, 10.0])))
        else:
            if x[0] < 0:
                objective = cp.Minimize(cp.quad_form(u-u_ref, np.diag([100.0, 10.0, 10.0, 1.0])))
            else:
                objective = cp.Minimize(cp.quad_form(u-u_ref, np.diag([10.0, 1.0, 100.0, 10.0])))
        constraints = [grad @ self.hj_dynamics(x, u, np.zeros(self.num_relative_state), None) + self.cbf_rate * value >= 0]
        problem = cp.Problem(objective, constraints)
        
        solve_status = problem.solve()
        # print(solve_status)
        u_sol = u.value
        if u_sol is None:
            return  u_ref
        u_sol[0] = max(min(u_sol[0], self.angular_rate_max), -self.angular_rate_max)
        u_sol[self.num_input] = max(min(u_sol[self.num_input], self.angular_rate_max), -self.angular_rate_max)
        return u_sol    

class DoubleIntegratorSafetyHandle(SafetyFilterIndividualHandle):
    def __init__(self, hj_data_handle: HjDataHandle):
        super(DoubleIntegratorSafetyHandle, self).__init__()
        self.coordination_range = DoubleIntegratorConfig.COORDINATION_RANGE
        self.vx_min = DoubleIntegratorConfig.VX_MIN
        self.vx_max = DoubleIntegratorConfig.VX_MAX
        self.vy_min = DoubleIntegratorConfig.VY_MIN
        self.vy_max = DoubleIntegratorConfig.VY_MAX
        self.accelx_min = DoubleIntegratorConfig.ACCELX_MIN
        self.accelx_max = DoubleIntegratorConfig.ACCELX_MAX
        self.accely_min = DoubleIntegratorConfig.ACCELY_MIN
        self.accely_max = DoubleIntegratorConfig.ACCELY_MAX
        self.num_input = 2        
        self.hj_data_handle = hj_data_handle
        self.hj_dynamics = DoubleIntegratorDynamics(vx_min=self.vx_min, vx_max=self.vx_max, vy_min=self.vy_min, vy_max=self.vy_max, accelx_min=self.accelx_min, accelx_max=self.accelx_max, accely_min=self.accely_min, accely_max=self.accely_max)
        self.cbf_rate = DoubleIntegratorConfig.CBF_RATE # function to generaliz
        self.num_relative_state = 4

    def clip_ctrl_with_valid_control_bound(self, state, u_sol):
        """ note that clipping is applied to each vehicle state and control input
        """
        dt = DoubleIntegratorConfig.DT

        accelx_max = self.accelx_max if state[2] < self.vx_max - dt * self.accelx_max else 0
        accelx_min = self.accelx_min if state[2] > self.vx_min - dt * self.accelx_min else 0
        u_sol[0] = max(min(u_sol[0], accelx_max), accelx_min)
        accely_max = self.accely_max if state[3] < self.vy_max - dt * self.accely_max else 0
        accely_min = self.accely_min if state[3] > self.vy_min - dt * self.accely_min else 0
        u_sol[1] = max(min(u_sol[1], accely_max), accely_min)
            
        return u_sol
        
    def get_relative_distance(self, ego_state, other_state):
        return np.sqrt((other_state[0] - ego_state[0]) ** 2 + (other_state[1] - ego_state[1]) ** 2)

    def get_value_of_relative_state(self, ego_state, other_state):
        relative_state = self.get_relative_state(ego_state, other_state)
        try:
            value_at_relative_state = self.hj_data_handle.grid_hj.interpolate(self.hj_data_handle.values_hj, relative_state)
            state_in_hj_range = (not np.isnan(value_at_relative_state))
        except:
            state_in_hj_range = False
        if not state_in_hj_range:
            value_at_relative_state = np.inf
        return value_at_relative_state, state_in_hj_range
    
    @staticmethod
    def get_relative_state(ego_state, other_state):
        x_r =  ego_state[0] - other_state[0]  # x2 - x1
        y_r = ego_state[1] - other_state[1]   # y2 - y1
        delta_vx = ego_state[2] - other_state[2]   # vx2 - vx1
        delta_vy = ego_state[3] - other_state[3]  # vy2 - vy1
        return np.array([x_r, y_r, delta_vx, delta_vy])
    
    def cbf_qp(self, x, u_ref, value, grad):
        u = cp.Variable(2 * self.num_input)
        objective = cp.Minimize(cp.quad_form(u-u_ref, np.eye(2 * self.num_input)))
        constraints = [grad @ self.hj_dynamics(x, u, np.zeros(self.num_relative_state), None) + self.cbf_rate * value >= 0]
        problem = cp.Problem(objective, constraints)
        
        solve_status = problem.solve()
        # print(solve_status)
        u_sol = u.value
        if u_sol is None:
            # print("Infeasible.")
            return  u_ref
        return u_sol    
            
    def apply_safety_filter(self, ego_state: np.ndarray, ego_action: np.ndarray,
                            ego_waypoint: np.ndarray,
                            other_state_list: List, other_action_list: List,
                            other_waypoint_list: List):
        """ pseudo code:
            1. evaluate relative distances with other vehicles.
            2. Pick the other vehicle that is the minimum distance.
            3. If that distance is less than coordination range, apply hj safety filter.
        """
        relative_distances = []
        relative_values = []
        relative_states_in_hj_range = []
        for other_state in other_state_list:
            relative_distance = self.get_relative_distance(ego_state, other_state)
            relative_distances.append(relative_distance)
            relative_value, state_in_hj_range = self.get_value_of_relative_state(ego_state, other_state)
            relative_values.append(relative_value)
            relative_states_in_hj_range.append(state_in_hj_range)
        min_agent_index_by_distance = np.argmin(relative_distances)
        # min_agent_index = np.argmin(relative_distances)
        min_agent_index = np.argmin(relative_values)
        # print(f"min_agent_index: by_distance: {min_agent_index_by_distance}, by_value: {min_agent_index}, value at min_dist: {relative_values[min_agent_index_by_distance]:.2f}, at min_value: {relative_values[min_agent_index]:.2f}, dist at min_value: {relative_distances[min_agent_index]:.2f}, dist at min_dist: {relative_distances[min_agent_index_by_distance]:.2f}")
        min_relative_distance = relative_distances[min_agent_index_by_distance]
        if min_relative_distance > self.coordination_range:
            # action is not filtered if agents are beyond coordination range.
            return ego_action, False, min_agent_index

        # pair_state = np.concatenate([ego_state, other_state_list[min_agent_index]])
        relative_state = self.get_relative_state(ego_state, other_state_list[min_agent_index])
        u_ref = np.zeros(2 * self.num_input)
        u_ref[:self.num_input] = ego_action
        u_ref[self.num_input:] = other_action_list[min_agent_index]
        eps_hj = 0.4
        # print("relative state: ", relative_state)
        value_at_relative_state = relative_values[min_agent_index]
        state_in_hj_range = relative_states_in_hj_range[min_agent_index]
        if not state_in_hj_range:
            # print(f"state out of hj range: {relative_state}")
            return ego_action, False, min_agent_index

        grad_at_relative_state = self.hj_data_handle.grid_hj.interpolate(self.hj_data_handle.grads_hj, relative_state)
          
        # crude least-restrictive reachability control.
        # print(f"value: {value_at_relative_state:.2f}")
        if value_at_relative_state < eps_hj:
            u = np.asarray(self.hj_dynamics.optimal_control(relative_state, None, grad_value=grad_at_relative_state)).copy()
        else:
            # use cbf constraint instead based on reachability value function.
            u = self.cbf_qp(relative_state, u_ref, value_at_relative_state, grad_at_relative_state)

        u = self.clip_ctrl_with_valid_control_bound(relative_state, u)
        # evaluate the norm diff between u and u_ref.
        u_norm_diff = float(np.linalg.norm(u - u_ref))
        filtered = u_norm_diff > 1e-4
        
        return u[:self.num_input], filtered, min_agent_index

class DoubleIntegratorSafetyHandleWithExponentialCBF(SafetyFilterIndividualHandle):
    def __init__(self):
        super(DoubleIntegratorSafetyHandleWithExponentialCBF, self).__init__()
        self.coordination_range = DoubleIntegratorConfig.COORDINATION_RANGE
        self.vx_min = DoubleIntegratorConfig.VX_MIN
        self.vx_max = DoubleIntegratorConfig.VX_MAX
        self.vy_min = DoubleIntegratorConfig.VY_MIN
        self.vy_max = DoubleIntegratorConfig.VY_MAX
        self.accelx_min = DoubleIntegratorConfig.ACCELX_MIN
        self.accelx_max = DoubleIntegratorConfig.ACCELX_MAX
        self.accely_min = DoubleIntegratorConfig.ACCELY_MIN
        self.accely_max = DoubleIntegratorConfig.ACCELY_MAX
        self.num_input = 2        
        self.cbf_rate = DoubleIntegratorConfig.CBF_RATE # function to generaliz
        self.num_relative_state = 4
        self.separation_distance = DoubleIntegratorConfig.SEPARATION_DISTANCE

    def clip_ctrl_with_valid_control_bound(self, state, u_sol):
        """ note that clipping is applied to each vehicle state and control input
        """
        dt = DoubleIntegratorConfig.DT

        accelx_max = self.accelx_max if state[2] < self.vx_max - dt * self.accelx_max else 0
        accelx_min = self.accelx_min if state[2] > self.vx_min - dt * self.accelx_min else 0
        u_sol[0] = max(min(u_sol[0], accelx_max), accelx_min)
        accely_max = self.accely_max if state[3] < self.vy_max - dt * self.accely_max else 0
        accely_min = self.accely_min if state[3] > self.vy_min - dt * self.accely_min else 0
        u_sol[1] = max(min(u_sol[1], accely_max), accely_min)
            
        return u_sol
        
    def get_relative_distance(self, ego_state, other_state):
        return np.sqrt((other_state[0] - ego_state[0]) ** 2 + (other_state[1] - ego_state[1]) ** 2)

    def get_value_and_lie_derivative_of_cbf_at_relative_state(self, ego_state, other_state):
        relative_state = self.get_relative_state(ego_state, other_state)
        relative_distance = np.sqrt(relative_state[0] ** 2 + relative_state[1] ** 2)
        distance_shifted = relative_distance - self.separation_distance
        ddistance = (relative_state[0] * relative_state[2] + relative_state[1] * relative_state[3]) / relative_distance
        value_cbf = ddistance + self.cbf_rate * distance_shifted
        
        Lf_cbf = self.cbf_rate * ddistance + (relative_state[0] * relative_state[3] - relative_state[1] * relative_state[2]) ** 2 / relative_distance ** 3
        Lg_cbf = np.zeros(4)
        Lg_cbf[0] = relative_state[0] / relative_distance
        Lg_cbf[1] = relative_state[1] / relative_distance
        Lg_cbf[2] = -relative_state[0] / relative_distance
        Lg_cbf[3] = -relative_state[1] / relative_distance
        return value_cbf, Lf_cbf, Lg_cbf
    
    @staticmethod
    def get_relative_state(ego_state, other_state):
        x_r =  ego_state[0] - other_state[0]  # x2 - x1
        y_r = ego_state[1] - other_state[1]   # y2 - y1
        delta_vx = ego_state[2] - other_state[2]   # vx2 - vx1
        delta_vy = ego_state[3] - other_state[3]  # vy2 - vy1
        return np.array([x_r, y_r, delta_vx, delta_vy])
    
    def cbf_qp(self, x, u_ref, value, Lf_cbf, Lg_cbf):
        u = cp.Variable(2 * self.num_input)
        objective = cp.Minimize(cp.quad_form(u-u_ref, np.eye(2 * self.num_input)))
        constraints = [Lg_cbf @ u + Lf_cbf + self.cbf_rate * value >= 0]
        problem = cp.Problem(objective, constraints)
        
        solve_status = problem.solve()
        # print(solve_status)
        u_sol = u.value
        if u_sol is None:
            # print("Infeasible.")
            return  u_ref
        return u_sol    
            
    def apply_safety_filter(self, ego_state: np.ndarray, ego_action: np.ndarray,
                            ego_waypoint: np.ndarray,
                            other_state_list: List, other_action_list: List,
                            other_waypoint_list: List):
        """ pseudo code:
            1. evaluate relative distances with other vehicles.
            2. Pick the other vehicle that is the minimum distance.
            3. If that distance is less than coordination range, apply hj safety filter.
        """
        relative_distances = []
        relative_values = []
        relative_states_in_hj_range = []
        for other_state in other_state_list:
            relative_distance = self.get_relative_distance(ego_state, other_state)
            relative_distances.append(relative_distance)
            relative_value, _, _ = self.get_value_and_lie_derivative_of_cbf_at_relative_state(ego_state, other_state)
            relative_values.append(relative_value)
            relative_states_in_hj_range.append(True)
        min_agent_index_by_distance = np.argmin(relative_distances)
        # min_agent_index = np.argmin(relative_distances)
        min_agent_index = np.argmin(relative_values)
        # print(f"min_agent_index: by_distance: {min_agent_index_by_distance}, by_value: {min_agent_index}, value at min_dist: {relative_values[min_agent_index_by_distance]:.2f}, at min_value: {relative_values[min_agent_index]:.2f}, dist at min_value: {relative_distances[min_agent_index]:.2f}, dist at min_dist: {relative_distances[min_agent_index_by_distance]:.2f}")
        min_relative_distance = relative_distances[min_agent_index_by_distance]
        if min_relative_distance > self.coordination_range:
            # action is not filtered if agents are beyond coordination range.
            return ego_action, False, min_agent_index

        # pair_state = np.concatenate([ego_state, other_state_list[min_agent_index]])
        relative_state = self.get_relative_state(ego_state, other_state_list[min_agent_index])
        u_ref = np.zeros(2 * self.num_input)
        u_ref[:self.num_input] = ego_action
        u_ref[self.num_input:] = other_action_list[min_agent_index]

        value_cbf, Lf_cbf, Lg_cbf = self.get_value_and_lie_derivative_of_cbf_at_relative_state(ego_state, other_state_list[min_agent_index])
        u = self.cbf_qp(relative_state, u_ref, value_cbf, Lf_cbf, Lg_cbf)

        u = self.clip_ctrl_with_valid_control_bound(relative_state, u)
        # evaluate the norm diff between u and u_ref.
        u_norm_diff = float(np.linalg.norm(u - u_ref))
        filtered = u_norm_diff > 1e-4
        
        return u[:self.num_input], filtered, min_agent_index