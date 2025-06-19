from typing import List
from enum import Enum

import numpy as np
import csv
from scipy.integrate import solve_ivp

from multiagent.safety_filter import KinematicVehicleSafetyHandle, DoubleIntegratorSafetyHandle, HjDataHandle
from multiagent.config import DoubleIntegratorConfig, AirTaxiConfig
from copy import deepcopy
# function to check for team or single agent scenarios
def is_list_of_lists(lst):
    if isinstance(lst, list) and lst:  # Check if it's a non-empty list
        return all(isinstance(item, list) for item in lst)
    return False

class EntityDynamicsType(Enum):
    DoubleIntegratorXY = 0
    KinematicVehicleXY = 1

# Base Class for entity state.
# physical/external base state of all entites
class BaseEntityState(object):
    def __init__(self, state_dim):
        self.state_dim = state_dim
        # state values
        self.values = np.zeros(4)
        self.max_speed = None        
        # travel distance
        self.p_dist = 0.0
        # travel time
        self.time = 0.0
        # communication state (only used when entity is agent)
        self.c = None
                    
    @property
    def p_pos(self):
        pass

    @p_pos.setter
    def p_pos(self, val):
        pass
    
    @property
    def p_vel(self):
        pass

    @p_vel.setter
    def p_vel(self, val):
        pass
    
    @property
    def speed(self):
        pass
    
    @staticmethod
    def dstate(state, action):
        pass
    
    def update_state(self, action, dt):
        # state space equation shoul be specified here in child class.
        pass
    
    def stop(self):
        # stop the vehicle (set vehicle speed states to 0)
        pass
    
    def reset_velocity(self, theta=None):
        # reset vehicle velocity. the default value can be zero or random values
        pass 


class KinematicVehicleXYState(BaseEntityState):
    def __init__(self, v_min, v_max):
        # state_dim = 4
        # p_x, p_y, theta, v
        super(KinematicVehicleXYState, self).__init__(4)
        self.min_speed = v_min
        self.max_speed = v_max

    @property
    def p_pos(self):
        return self.values[:2]

    @p_pos.setter
    def p_pos(self, val):
        self.values[:2] = val

    @property
    def speed(self):
        return self.values[3]
    
    @speed.setter
    def speed(self, val):
        self.values[3] = val

    @property
    def theta(self):
        return self.values[2]

    @theta.setter
    def theta(self, val):
        self.values[2] = val

    @property
    def p_vel(self):
        return np.array([self.speed * np.cos(self.theta),
                         self.speed * np.sin(self.theta)])
    
    @staticmethod
    def dstate(state, action):
        dp_x = state[3] * np.cos(state[2])
        dp_y = state[3] * np.sin(state[2])
        dtheta = action[0]
        dv = action[1]
        return np.array([dp_x, dp_y, dtheta, dv])

    def update_state(self, action, dt):
        def ode(t, y):
            return self.dstate(y, action)
        y0 = self.values
        sol = solve_ivp(ode, [0, dt], y0, method='RK45')
        self.values = sol.y[:, -1]
        
        if self.speed > self.max_speed:
            self.speed = self.max_speed
        if self.speed < self.min_speed:
            self.speed = self.min_speed
        # update traveled time and distance.
        self.p_dist += self.speed * dt
        self.time += dt
    
    def stop(self):
        self.theta = 0
        self.speed = 0
    
    def reset_velocity(self, theta=None, speed=None):
        if theta is not None:
            self.theta = theta
        else:
            self.theta = np.random.uniform(0, 2 * np.pi)
        if speed is not None:
            self.speed = speed
        else:
            self.speed = self.min_speed
    
    def __getitem__(self, idx):
        return self.values[idx]
    
class DoubleIntegratorXYState(BaseEntityState):
    def __init__(self):
        # state_dim = 4
        # p_x, p_y, v_x, v_y
        super(DoubleIntegratorXYState, self).__init__(4)
        # double integrator can stop.
        self.min_speed = 0.0
        self.max_speed = DoubleIntegratorConfig().VX_MAX
        self.min_vel_x = DoubleIntegratorConfig().VX_MIN
        self.max_vel_x = DoubleIntegratorConfig().VX_MAX
        self.min_accel_x = DoubleIntegratorConfig.ACCELX_MIN
        self.max_accel_x = DoubleIntegratorConfig.ACCELX_MAX
        self.min_accel_y = DoubleIntegratorConfig.ACCELY_MIN
        self.max_accel_y = DoubleIntegratorConfig.ACCELY_MAX

    @property
    def p_pos(self):
        return self.values[:2]

    @p_pos.setter
    def p_pos(self, val):
        self.values[:2] = val


    @property
    def speed(self):
        return np.sqrt(self.values[2] ** 2 + self.values[3] ** 2)
    

    @property
    def theta(self):
        return np.arctan2(self.values[3], self.values[2])

    @property
    def p_vel(self):
        return self.values[2:]

    @p_vel.setter
    def p_vel(self, val):
        self.values[2:] = val

    @staticmethod
    def dstate(state, action):
        dp_x = state[2]
        dp_y = state[3]
        dv_x = action[0]
        dv_y = action[1]
        return np.array([dp_x, dp_y, dv_x, dv_y])
        
    def update_state(self, action, dt): # check
        def ode(t, y):
            return self.dstate(y, action)
        y0 = self.values
        sol = solve_ivp(ode, [0, dt], y0, method='RK45')
        self.values = sol.y[:, -1]
        if self.speed > self.max_speed:
            # adjust magnitude to self.max_speed
            self.p_vel = self.max_speed * self.p_vel / self.speed
        # update traveled time and distance.
        self.p_dist += self.speed * dt
        self.time += dt
        
    def stop(self):
        self.p_vel = np.zeros(2)        

    def reset_velocity(self, theta=None):
        """ theta is unused but needed to match the interface of KinematicVehicleXYState """
        self.p_vel = np.zeros(2)        
    
    def __getitem__(self, idx):
        return self.values[idx]

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties of wall entities
class Wall(object):
    def __init__(self, orient='H', axis_pos=0.0, endpoints=(-1, 1), width=0.1,
                hard=True):
        # orientation: 'H'orizontal or 'V'ertical
        self.orient = orient
        # position along axis which wall lays on (y-axis for H, x-axis for V)
        self.axis_pos = axis_pos
        # endpoints of wall (x-coords for H, y-coords for V)
        self.endpoints = np.array(endpoints)
        # width of wall
        self.width = width
        self.size = self.width
        # whether wall is impassable to all agents
        self.hard = hard
        # color of wall
        self.color = np.array([0.0, 0.0, 0.0])
        self.state = DoubleIntegratorXYState()
        # commu channel
        self.channel = None


# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # id
        self.id = None
        self.global_id = None
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # entity can pass through non-hard walls
        self.ghost = False
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = np.array([0.20, 0.20, 0.20])
                
        self.accel = None
        # state
        self.state = DoubleIntegratorXYState() #why?
        # mass
        self.initial_mass = 1.0
        # commu channel
        self.channel = None

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()
        # if heading and speed are None, goal-reaching reward is only based on position.
        # if they have specific values, we evaluate reward based on heading and speed as well.
        self.heading = None # rad
        self.speed = None # m/s
        
# properties of agent entities
class Agent(Entity):
    def __init__(self, dynamics_type: EntityDynamicsType):
        super(Agent, self).__init__()
        # agent are adversary
        self.adversary = False
        # agent are dummy
        self.dummy = False
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        self.min_speed = None
        # state & dynamics
        self.dynamics_type = dynamics_type
        if dynamics_type == EntityDynamicsType.DoubleIntegratorXY:
            self.config_class = DoubleIntegratorConfig
            self.state = DoubleIntegratorXYState()
            self.min_speed = self.state.min_speed
        elif dynamics_type == EntityDynamicsType.KinematicVehicleXY:
            self.config_class = AirTaxiConfig
            self.state = KinematicVehicleXYState(v_min=AirTaxiConfig.V_MIN, v_max=AirTaxiConfig.V_MAX)
            self.min_speed = self.state.min_speed
        else:
            raise NotImplementedError("Dynamics type not implemented")
        self.max_speed = self.state.max_speed
        # action
        self.action = Action()
        # flag of whether the action is filtered or not.
        self.action_safety_filtered = False
        
        # script behavior to execute
        self.action_callback = None
        # min time required to get to its allocated goal
        self.goal_min_time = np.inf
        # time passed for each agent
        self.t = 0.0
        # done is True only when agent went through all the waypoints.
        # it is updated in navigation_graph_safe.py/SafeAamScenario/reward()
        self.done = False
        # departed is True only when agent starts moving towards the goal.
        self.departed = True

        # rendering the agent with rotations needs to store initial theta
        self.initial_theta = None
        
        self.target_waypoint = None
        
        # record other agent's index that is deconflicting with this agent in safety filter (that results in the primary active safety constraint)
        self.deconflicting_agent_index = -1
        self.safety_filtered = False
        self.min_relative_distance = np.inf
        # diff between safety-filtered action and raw action.
        self.action_diff = 0.0

# multi-agent world
class World(object):
    def __init__(self, dynamics_type: EntityDynamicsType, use_safety_filter: bool=False,  
                 use_hj_handle: bool=False,
                 num_internal_step=1, separation_distance=None, separation_distance_target=None):

        assert dynamics_type in EntityDynamicsType, "Invalid dynamics type"
        self.dynamics_type = dynamics_type
        self.use_hj_handle = use_hj_handle
        # if we want to construct graphs with the entities 
        self.graph_mode = False
        self.edge_list = None
        self.graph_feat_type = None
        self.edge_weight = None
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.scripted_agents = []
        self.scripted_agents_goals = []
        self.obstacles, self.walls = [], []
        self.wall_obstacles = []
        self.belief_targets = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        if dynamics_type == EntityDynamicsType.DoubleIntegratorXY:
            self.config_class = DoubleIntegratorConfig
        elif dynamics_type == EntityDynamicsType.KinematicVehicleXY:
            self.config_class = AirTaxiConfig
        else:
            raise NotImplementedError("Dynamics type not implemented")
        self.dt = self.config_class.DT
        self.simulation_time = 0.0
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1.3e+2
        self.wall_contact_force = 2.2e+2

        self.contact_margin = 1.9e-3
        self.wall_contact_margin =2.4e-2
        # self.contact_force = 1.1e+2
        # self.wall_contact_force = 2.9e+2

        # self.contact_margin = 1.7e-3
        # self.wall_contact_margin =2.9e-2
        #         # contact response parameters
        # self.contact_force = 2e+2
        # self.wall_contact_force = 3.0e+2

        # self.contact_margin = 3e-2
        # self.wall_contact_margin = 5e-2
        # cache distances between all agents (not calculated by default)
        self.cache_dists = True
        self.cached_dist_vect = None
        self.cached_dist_mag = None
        
        self.num_internal_step = num_internal_step

        self.use_safety_filter = use_safety_filter
        self.coordination_range = self.config_class.COORDINATION_RANGE
        self.min_dist_thresh = self.config_class.DISTANCE_TO_GOAL_THRESHOLD
        # Used for rendering even if use_safety_filter is False
        self.separation_distance = separation_distance
        self.separation_distance_target = separation_distance_target
        self.engagement_distance = self.config_class.ENGAGEMENT_DISTANCE
            
        self.agent_safety_handle_list = []
        self.hj_data_handle = self._init_hj_handle(self.dynamics_type, self.separation_distance) if use_hj_handle else None
        if self.dynamics_type == EntityDynamicsType.DoubleIntegratorXY:
            self.get_relative_state_func = DoubleIntegratorSafetyHandle.get_relative_state
        elif self.dynamics_type == EntityDynamicsType.KinematicVehicleXY:
            self.get_relative_state_func = KinematicVehicleSafetyHandle.get_relative_state
        elif self.dynamics_type == EntityDynamicsType.KinematicVehicleXY:
            self.get_relative_state_func = KinematicVehicleSafetyHandle.get_relative_state
            
    @property
    def num_departed_agents(self):
        return sum([1 for agent in self.agents if agent.departed])
    
    @property
    def num_done_agents(self):
        return sum([1 for agent in self.agents if agent.done])

    @staticmethod
    def _init_hj_handle(dynamics_type, separation_distance):
        # Choose the file based on dynamics type
        if dynamics_type == EntityDynamicsType.DoubleIntegratorXY:
            hj_file_name = DoubleIntegratorConfig.VALUE_FUNCTION_FILE_NAME
        elif dynamics_type == EntityDynamicsType.KinematicVehicleXY:
            hj_file_name = AirTaxiConfig.VALUE_FUNCTION_FILE_NAME
        else:
            raise NotImplementedError("Dynamics type not implemented for HJ handle initialization")
        
        # Initialize the HJ data handle with the selected file
        hj_data_handle = HjDataHandle(hj_file_name, separation_distance)
        return hj_data_handle
    
    def get_hj_value_between_two_agents(self, agent1, agent2):
        assert self.hj_data_handle is not None, "HJ data handle is not initialized."
        relative_state = self.get_relative_state_func(agent1.state.values, agent2.state.values)
        try:
            value_at_relative_state = self.hj_data_handle.grid_hj.interpolate(self.hj_data_handle.values_hj, relative_state)
            if np.isnan(value_at_relative_state):
                return np.inf
            return value_at_relative_state
        except:
            return np.inf
    
    def init_safety_filter(self):
        if self.use_safety_filter:
            if self.hj_data_handle is None:
                raise ValueError("HJ data handle is not initialized. since use_hj_handle is False.")
            # Initialize safety handles for agents
            for _ in self.agents:
                if self.dynamics_type == EntityDynamicsType.DoubleIntegratorXY:
                    self.agent_safety_handle_list.append(DoubleIntegratorSafetyHandle(self.hj_data_handle))
                elif self.dynamics_type == EntityDynamicsType.KinematicVehicleXY:
                    self.agent_safety_handle_list.append(KinematicVehicleSafetyHandle(AirTaxiConfig, self.hj_data_handle))
                else:
                    raise NotImplementedError("Dynamics type not implemented for safety handle initialization")

    def update_safety_filter_separation_distance(self, separation_distance):
        self.separation_distance = separation_distance
        if self.hj_data_handle is not None:
            self.hj_data_handle.update_separation_distance(separation_distance)

    # return all entities in the world
    @property
    def entities(self):
        if is_list_of_lists(self.agents):
            ## flatten the list of lists into a single list
            flattened_agents = [agent for team in self.agents for agent in team]
            return flattened_agents + self.landmarks + self.obstacles + self.wall_obstacles + self.walls
        if not is_list_of_lists(self.agents):
            return self.agents + self.landmarks + self.obstacles + self.wall_obstacles + self.walls

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        if is_list_of_lists(self.agents):
            ## flatten the list of lists into a single list
            flattened_agents = [agent for team in self.agents for agent in team]
            return flattened_agents
        if not is_list_of_lists(self.agents):
            return self.agents
        # return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def get_scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def calculate_distances(self):
        if self.cached_dist_vect is None:
            # initialize distance data structure

            # remove landmarks from the list of entities
            # self.new_entities = [a for a in self.entities if not isinstance(a, Landmark)] ## used to remove landmarks from goals
            self.cached_dist_vect = np.zeros((len(self.entities),
                                            len(self.entities),
                                            self.dim_p))
            # calculate minimum distance for a collision between all entities
            self.min_dists = np.zeros((len(self.entities), len(self.entities)))
            for ia, entity_a in enumerate(self.entities):
                for ib in range(ia + 1, len(self.entities)):
                    entity_b = self.entities[ib]
                    min_dist = entity_a.size + entity_b.size
                    self.min_dists[ia, ib] = min_dist
                    self.min_dists[ib, ia] = min_dist

        for ia, entity_a in enumerate(self.entities):
            for ib in range(ia + 1, len(self.entities)):
                entity_b = self.entities[ib]
                # print("ia",ia,"ib",ib, "entity_a",entity_a.name, "entity_b",entity_b.name)
                # print("entity_a",entity_a.state.p_pos, "entity_b",entity_b.state.p_pos)
                delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
                self.cached_dist_vect[ia, ib, :] = delta_pos
                self.cached_dist_vect[ib, ia, :] = -delta_pos

        self.cached_dist_mag = np.linalg.norm(self.cached_dist_vect, axis=2)

        self.cached_collisions = (self.cached_dist_mag <= self.min_dists)

    # get the entity given the id and type
    def get_entity(self, entity_type: str, id:int) -> Entity:
        if entity_type == 'agent':
            for agent in self.agents:
                if agent.name == f'agent {id}':
                    return agent
            raise ValueError(f"Agent with id: {id} doesn't exist in the world")
        if entity_type == 'landmark':
            for landmark in self.landmarks:
                if landmark.name == f'landmark {id}':
                    return landmark
            raise ValueError(f"Landmark with id: {id} doesn't exist in the world")
        if entity_type == 'obstacle':
            for obstacle in self.obstacles:
                if obstacle.name == f'obstacle {id}':
                    return obstacle
            raise ValueError(f"Obstacle with id: {id} doesn't exist in the world")

    def get_waypoint_for_safety_filter(self, action_list, num_internal_step):
        waypoint_list = []
        for i, agent in enumerate(self.agents):
            if not agent.movable: continue
            waypoint_list.append(agent.target_waypoint)
        return waypoint_list
        # waypoint_list = []
        # for i, agent in enumerate(self.agents):
        #     if not agent.movable: continue                
        #     action_i = action_list[i]
        #     evolved_state_i = np.copy(agent.state.values)
        #     for _ in range(num_internal_step):
        #         dp_x = evolved_state_i[3] * np.cos(evolved_state_i[2])
        #         dp_y = evolved_state_i[3] * np.sin(evolved_state_i[2])
        #         dtheta = action_i[0]
        #         dv = action_i[1]
        #         dstate = np.array([dp_x, dp_y, dtheta, dv])
        #         # simple euler integration.
        #         evolved_state_i += dstate * self.dt
        #     if not agent.status:           
        #         if agent.max_speed is not None and evolved_state_i[3] > agent.max_speed:
        #             evolved_state_i[3] = agent.max_speed
        #         if agent.min_speed is not None and agent.state.speed < agent.min_speed:
        #             evolved_state_i[3] = agent.min_speed
        #     waypoint_list.append(evolved_state_i[:-1])
        # print("waypoint_list (original)")
        # print(waypoint_list)
        # return waypoint_list

    # update state of the world
    def step(self):
        # set actions for scripted agents (moving obstacle)
        # for agent in self.scripted_agents: 
        #     agent.action = agent.action_callback(agent, self)
        #     agent.t += self.dt

        # print("self.horizon",horizon)
        ## repeat step for 5 or n times each time step is called
        raw_action_list = self.get_action()
        if self.use_safety_filter:
            waypoint_list = self.get_waypoint_for_safety_filter(raw_action_list, self.num_internal_step)
        else:
            waypoint_list = None

        for _ in range(self.num_internal_step):
            if self.use_safety_filter:
                # add filter
                safe_action_list, filtered_flag_list, deconflicting_agent_index_list = self.apply_safety_filter(raw_action_list, waypoint_list)
                self.update_agent_filtered_flag(filtered_flag_list)
                for agent in self.agents:
                    agent.deconflicting_agent_index = deconflicting_agent_index_list[agent.id]
                    agent.safety_filtered = filtered_flag_list[agent.id]
            else:
                safe_action_list = raw_action_list
            for agent in self.agents:
                # evaluate norm diff between raw and safe actions.
                agent.action_diff = np.linalg.norm(np.array(raw_action_list[agent.id]) - np.array(safe_action_list[agent.id]))
            # integrate physical state
            self.update_agent_state(safe_action_list)
            # update agent state
            for agent in self.agents:
                agent.t += self.dt
                self.update_agent_communication_state(agent)
            if self.cache_dists:
                self.calculate_distances()
                
            self.update_agent_min_relative_distance()
            
            self.simulation_time += self.dt

    # gather agent action forces
    def get_action(self):
        # set applied forces
        ## agent action has an linear acceleration term and an angular acceleration term
        action_list = []
        for i,agent in enumerate(self.agents):
            if agent.u_noise:
                # Jason's temporary fix
                raise NotImplementedError

            action_i = np.array([agent.action.u[0], agent.action.u[1]])
            action_list.append(action_i)
       
        return action_list

    def apply_safety_filter(self, action_list: List, waypoint_list: List):
        """ return filtered_action_list and flag_list of whether the action is filtered or not.
        """
        safe_action_list = []
        filtered_flag_list = []
        deconflicting_agent_index_list = []
        for i, agent in enumerate(self.agents):
            if agent.done or not agent.departed:
                safe_action_list.append(action_list[i])
                filtered_flag_list.append(False)
                # -1 indicates that there is no deconflicting agent.
                deconflicting_agent_index_list.append(-1)
                continue
            ego_state = agent.state.values
            other_state_list, other_state_agent_index_list = self.get_other_agent_state_list(agent)
            if not other_state_list:
                safe_action_list.append(action_list[i])
                filtered_flag_list.append(False)
                deconflicting_agent_index_list.append(-1)
                continue
            ego_action = action_list[i]
            other_action_list = [action_list[j] for j in range(len(action_list)) if j != i and (not self.agents[j].done and self.agents[j].departed)]
            ego_waypoint = waypoint_list[i]
            other_waypoint_list = [waypoint_list[j] for j in range(len(waypoint_list)) if j != i and (not self.agents[j].done and self.agents[j].departed)]
            safe_ego_action, filtered_flag, deconflicting_agent_index = self.agent_safety_handle_list[i].apply_safety_filter(ego_state, ego_action, ego_waypoint, other_state_list, other_action_list, other_waypoint_list)
            filtered_flag_list.append(filtered_flag)
            safe_action_list.append(safe_ego_action)
            deconflicting_agent_index_list.append(other_state_agent_index_list[deconflicting_agent_index])

        return safe_action_list, filtered_flag_list, deconflicting_agent_index_list

    # integrate physical state
    def update_agent_state(self, action_list: List):
        # TODO: Change entities to agents
        for i, agent in enumerate(self.agents):
            action_i = action_list[i]
            if not agent.movable: continue
            if agent.done or not agent.departed:
                continue
            agent.state.update_state(action_i, self.dt)
            
    def update_agent_filtered_flag(self, filtered_flag_list: List):
        for i, agent in enumerate(self.agents):
            if not agent.movable: continue
            if agent.done or not agent.departed:
                continue
            agent.action_safety_filtered = filtered_flag_list[i]
    
    def update_agent_min_relative_distance(self):
        agents_positions = [agent.state.p_pos for agent in self.agents]
        agent_relative_distance_matrix = np.inf * np.ones((len(self.agents), len(self.agents)))
        for i, agent in enumerate(self.agents):
            if agent.done or not agent.departed:
                continue
            for j in range(len(self.agents)):
                if i == j:
                    continue
                if not self.agents[j].departed or self.agents[j].done:
                    continue
                agent_relative_distance_matrix[i, j] = np.linalg.norm(agents_positions[i] - agents_positions[j])
        for i, agent in enumerate(self.agents):
            agent.min_relative_distance = np.min(agent_relative_distance_matrix[i, :])
    
    def update_waypoints_for_safety_filter_from_predicted_trajectories(self, state_prediction_sequence_list: List):
        for i, agent in enumerate(self.agents):
            if agent.done or not agent.departed:
                continue
            state_prediction_sequence = state_prediction_sequence_list[i]
            # use the first three state (x, y, theta) of the terminal time state as the waypoint
            agent.target_waypoint = state_prediction_sequence[-1, :-1]

    def update_agent_communication_state(self, agent:Agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * \
                    agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    def get_other_agent_state_list(self, agent:Agent):
        state_list = []
        index_list = []
        for (i_other, other_agent) in enumerate(self.agents):
            if other_agent == agent or other_agent.done or not other_agent.departed:
                continue
            state_list.append(other_agent.state.values)
            index_list.append(i_other)
        return state_list, index_list

    # get collision forces for any contact between two entities
    # NOTE: this is better than using get_collision_force() since 
    # it takes into account if the entity is movable or not
    def get_entity_collision_force(self, ia, ib):
        entity_a = self.entities[ia]
        entity_b = self.entities[ib]
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if (not entity_a.movable) and (not entity_b.movable):
            return [None, None]  # neither entity moves
        if (entity_a is entity_b):
            return [None, None]  # don't collide against itself
        if (self.cache_dists) and (self.cached_dist_vect is not None):
            delta_pos = self.cached_dist_vect[ia, ib]
            dist = self.cached_dist_mag[ia, ib]
            dist_min = self.min_dists[ia, ib]
        else:
            # compute actual distance between entities
            delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            # minimum allowable distance
            dist_min = 1.1*entity_a.size + 1.1*entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        # print("force",force)
        if entity_a.movable and entity_b.movable:
            # consider mass in collisions
            force_ratio = entity_b.mass / entity_a.mass
            force_a = force_ratio * force if entity_a.done != True else None
            force_b = -(1 / force_ratio) * force if entity_b.done != True else None
        else:
            force_a = +force if entity_a.movable else None
            force_b = -force if entity_b.movable else None
        # print("Entity collision forces: ", force_a, force_b)
        return [force_a, force_b]

    # get collision forces for contact between an entity and a wall
    def get_wall_collision_force(self, entity, wall):
        if entity.ghost and not wall.hard:
            return None  # ghost passes through soft walls
        if wall.orient == 'H':
            prll_dim = 0
            perp_dim = 1
        else:
            prll_dim = 1
            perp_dim = 0
        ent_pos = entity.state.p_pos
        if (ent_pos[prll_dim] < wall.endpoints[0] - entity.size or
                ent_pos[prll_dim] > wall.endpoints[1] + entity.size):
            return None  # entity is beyond endpoints of wall
        elif (ent_pos[prll_dim] < wall.endpoints[0] or
                ent_pos[prll_dim] > wall.endpoints[1]):
            # part of entity is beyond wall
            if ent_pos[prll_dim] < wall.endpoints[0]:
                dist_past_end = ent_pos[prll_dim] - wall.endpoints[0]
            else:
                dist_past_end = ent_pos[prll_dim] - wall.endpoints[1]
            theta = np.arcsin(dist_past_end / entity.size)
            dist_min = np.cos(theta) * entity.size + 0.5 * wall.width
        else:  # entire entity lies within bounds of wall
            theta = 0
            dist_past_end = 0
            dist_min = entity.size + 0.5 * wall.width

        # only need to calculate distance in relevant dim
        delta_pos = ent_pos[perp_dim] - wall.axis_pos
        dist = np.abs(delta_pos)
        # softmax penetration
        k = self.wall_contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force_mag = self.wall_contact_force * delta_pos / dist * penetration
        # force_mag = self.wall_contact_force * delta_pos / dist

        force = np.zeros(2)
        force[perp_dim] = np.cos(theta) * force_mag
        force[prll_dim] = np.sin(theta) * np.abs(force_mag)
        return force

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        # print("Collision forces: ", force_a, force_b)
        return [force_a, force_b]
    
    def assign_agent_colors(self):
        n_dummies = 0
        if hasattr(self.agents[0], 'dummy'):
            n_dummies = len([a for a in self.agents if a.dummy])
        n_adversaries = 0
        if hasattr(self.agents[0], 'adversary'):
            n_adversaries = len([a for a in self.agents if a.adversary])
        n_good_agents = len(self.agents) - n_adversaries - n_dummies
        # r g b
        dummy_colors = [(0.25, 0.75, 0.25)] * n_dummies
        adv_colors = [(0.75, 0.25, 0.25)] * n_adversaries
        good_colors = [(0.25, 0.25, 0.75)] * n_good_agents
        colors = dummy_colors + adv_colors + good_colors
        for color, agent in zip(colors, self.agents):
            agent.color = color

    # landmark color
    def assign_landmark_colors(self):
        for landmark in self.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])


## implement 3d dubins dynamics for the agents
