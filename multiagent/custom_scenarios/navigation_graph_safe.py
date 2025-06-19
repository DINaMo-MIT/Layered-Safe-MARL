"""
	Navigation for `n` agents to `n` goals from random initial positions
	With random obstacles added in the environment
	Each agent is destined to get to its own goal unlike
	`simple_spread.py` where any agent can get to any goal (check `reward()`)
"""
from typing import Optional, Tuple, List
import argparse
import numpy as np
from numpy import ndarray as arr
from scipy import sparse
from copy import deepcopy
from PIL import Image
import pickle

import os,sys
sys.path.append(os.path.abspath(os.getcwd()))
from scipy.optimize import linear_sum_assignment
import math
from multiagent.core import EntityDynamicsType, World, Agent, Landmark, Entity, Wall
from multiagent.scenario import BaseScenario
from multiagent.config import RewardWeightConfig, RewardBinaryConfig, DoubleIntegratorConfig, AirTaxiConfig
from multiagent.custom_scenarios.utils import *
from hj_reachability_utils.common import get_hj_grid_from_meta_data, TtrData

class SafeAamScenario(BaseScenario):
	def get_default_landmark_num_for_scenario(self) -> int:
		# default landmark num per agent
		# Specific implementation is required for child classes.
		raise NotImplementedError

	def get_aspect_ratio_for_scenario(self) -> float:
		# width / height ratio
		# Specific implementation is required for child classes.
		raise NotImplementedError

	def init_landmarks(self, 
					world:World, 
					num_landmark_per_agent:int,
					global_id:int) -> None:
		if num_landmark_per_agent == 0:
			num_landmark_per_agent = self.get_default_landmark_num_for_scenario()
		assert num_landmark_per_agent > 0, "Number of landmarks should be positive."
		self.num_landmark_per_agent = num_landmark_per_agent
		# number of total landmarks
		self.num_landmarks = num_landmark_per_agent * self.num_agents
		world.landmarks = [Landmark() for i in range(self.num_landmarks)]
		
		for i, landmark in enumerate(world.landmarks):
			landmark.id = i
			landmark.name = f'landmark {i}'
			landmark.collide = False
			landmark.movable = False
			landmark.global_id = global_id
			global_id += 1
		return global_id

	def make_world(self, args:argparse.Namespace) -> World:
		"""
			Parameters in args
			––––––––––––––––––
			• num_agents: int
				Number of agents in the environment
				NOTE: this is equal to the number of goal positions
			• num_obstacles: int
				Number of num_obstacles obstacles
			• collaborative: bool
				If True then reward for all agents is sum(reward_i)
				If False then reward for each agent is what it gets individually
			• max_speed: Optional[float]
				Maximum speed for agents
				NOTE: Even if this is None, the max speed achieved in discrete 
				action space is 2, so might as well put it as 2 in experiments
				TODO: make list for this and add this in the state
			• collision_rew: float
				The reward to be negated for collisions with other agents and 
				obstacles
			• goal_rew: float
				The reward to be added if agent reaches the goal
			• min_dist_thresh: float
				The minimum distance threshold to classify whether agent has 
				reached the goal or not
			• use_dones: bool
				Whether we want to use the 'done=True' when agent has reached 
				the goal or just return False like the `simple.py` or 
				`simple_spread.py`
			• episode_length: int
				Episode length after which environment is technically reset()
				This determines when `done=True` for done_callback
			• graph_feat_type: str
				The method in which the node/edge features are encoded
				Choices: ['global', 'relative']
					If 'global': 
						• node features are global [pos, vel, goal, entity-type]
						• edge features are relative distances (just magnitude)
						• 
					If 'relative':
						• TODO decide how to encode stuff
		"""
		# pull params from args
		if not hasattr(self, 'world_size'):
			self.world_size = args.world_size
		self.world_aspect_ratio = self.get_aspect_ratio_for_scenario()
		self.num_agents = args.num_agents
		self.num_scripted_agents = args.num_scripted_agents
		self.num_obstacles = args.num_obstacles
		self.collaborative = args.collaborative
		self.use_dones = args.use_dones
		self.episode_length = args.episode_length
		# used for curriculum learning (in training)
		self.num_total_episode = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads
		# print("num_total_episode",self.num_total_episode)

		# create heatmap matrix to determine the goal agent pairs
		self.goal_reached = np.zeros(self.num_agents)
		self.wrong_goal_reached = np.zeros(self.num_agents)
		self.goal_matched = np.zeros(self.num_agents)
		self.agent_dist_traveled = np.zeros(self.num_agents)
		self.agent_time_taken = np.zeros(self.num_agents)
		if args.dynamics_type == 'double_integrator':
			self.dynamics_type = EntityDynamicsType.DoubleIntegratorXY
			self.config_class = DoubleIntegratorConfig
			self.min_turn_radius = 0.0
			self.grid_ttr = None
			self.ttr_values = None
			self.ttr_max = None

		elif args.dynamics_type == 'airtaxi':
			self.dynamics_type = EntityDynamicsType.KinematicVehicleXY
			self.config_class = AirTaxiConfig
			self.min_turn_radius = 0.5 * (AirTaxiConfig.V_MAX + AirTaxiConfig.V_MIN) / AirTaxiConfig.ANGULAR_RATE_MAX
			ttr_file_name = AirTaxiConfig.TTR_FILE_NAME
			with open(ttr_file_name, 'rb') as f:
				ttr_data_loaded = pickle.load(f)
			grid_hj_meta_data = ttr_data_loaded.grid_meta_data
			self.grid_ttr = get_hj_grid_from_meta_data(grid_hj_meta_data)
			self.ttr_values = ttr_data_loaded.values
			self.ttr_max = ttr_data_loaded.ttr_max
		else:
			raise NotImplementedError
		self.engagement_distance_ref = self.config_class.ENGAGEMENT_DISTANCE
		self.engagement_distance_ref_separation_distance = self.config_class.ENGAGEMENT_DISTANCE_REFERENCE_SEPARATION_DISTANCE
		self.coordination_range = self.config_class.COORDINATION_RANGE
		# self.min_dist_thresh_init = 2 * self.config_class.DISTANCE_TO_GOAL_THRESHOLD
		self.min_dist_thresh_init = self.config_class.DISTANCE_TO_GOAL_THRESHOLD
		self.min_dist_thresh_target = self.config_class.DISTANCE_TO_GOAL_THRESHOLD
		self.min_dist_thresh = self.min_dist_thresh_init

		self.goal_heading_error = self.config_class.GOAL_HEADING_THRESHOLD
		self.goal_heading_error_thresh_init = 0.5 - 0.5 * np.cos(self.goal_heading_error)
		self.goal_heading_error_thresh_target = 0.5 - 0.5 * np.cos(self.goal_heading_error)
		self.goal_heading_error_thresh = self.goal_heading_error_thresh_init
		# self.goal_speed_error_thresh_init = 2 * self.config_class.GOAL_SPEED_THRESHOLD
		self.goal_speed_error_thresh_init = self.config_class.GOAL_SPEED_THRESHOLD
		self.goal_speed_error_thresh_target = self.config_class.GOAL_SPEED_THRESHOLD
		self.goal_speed_error_thresh = self.goal_speed_error_thresh_init
		self.goal_speed_min = self.config_class.V_MIN
		self.goal_speed_max = self.config_class.V_NOMINAL
		# scripted agent dynamics are fixed to double integrator for now (as it was originally)
		scripted_agent_dynamics_type = EntityDynamicsType.DoubleIntegratorXY

		self.goal_rew = RewardWeightConfig.GOAL_REACH
		self.safety_violation_rew = RewardWeightConfig.SAFETY_VIOLATION # negative
		self.hj_value_rew = RewardWeightConfig.HJ_VALUE # negative
		self.potential_conflict_rew = RewardWeightConfig.POTENTIAL_CONFLICT # negative
		self.diff_from_filtered_action_rew = RewardWeightConfig.DIFF_FROM_FILTERED_ACTION # negative
		# will be set up by curriculum learning
		self.multiple_engagement_rew_scaled = 0
		self.conflict_rew_scaled = 0
		self.diff_from_filtered_action_rew_scaled = 0
		self.conflict_value_rew_scaled = 0

		# needed when scenario is used in evaluation.
		self.curriculum_ratio = 1.0

		self.min_reward = RewardWeightConfig.MIN_REWARD
		self.max_reward = RewardWeightConfig.MAX_REWARD

		self.optimal_match_index = np.arange(self.num_agents)

		self.max_edge_dist = self.coordination_range
		self.use_safety_filter = args.use_safety_filter
		self.separation_distance_curriculum = RewardBinaryConfig.SEPARATION_DISTANCE_CURRICULUM
		# If this is false, in the initial phase of the training, safety filter is not used.
		self.initial_phase_safety_filter = self.use_safety_filter and RewardBinaryConfig.INITIAL_PHASE_USE_SAFETY_FILTER
		self.separation_distance_target = self.config_class.SEPARATION_DISTANCE
		if self.separation_distance_curriculum:
			self.separation_distance_init = 0
		else:
			self.separation_distance_init = self.separation_distance_target
		self.separation_distance = self.separation_distance_init
		self.update_engagement_distance_based_on_separation_distance(self.separation_distance)


		use_hj_handle = self.use_safety_filter or RewardBinaryConfig.HJ_VALUE
		world = World(dynamics_type=self.dynamics_type, use_safety_filter=self.use_safety_filter, use_hj_handle=use_hj_handle, num_internal_step=args.num_internal_step, separation_distance=self.separation_distance, separation_distance_target=self.separation_distance_target)
		# graph related attributes
		world.graph_mode = True
		world.graph_feat_type = args.graph_feat_type
		world.world_length = args.episode_length
		# metrics to keep track of
		world.current_time_step = 0
		# to track time required to reach goal
		world.times_required = -1 * np.ones(self.num_agents)
		world.dists_to_goal = -1 * np.ones(self.num_agents)
		# set any world properties
		world.dim_c = 2

		num_scripted_agents_goals = self.num_scripted_agents
		world.collaborative = args.collaborative
		self.use_masking = args.use_masking
		# add agents
		global_id = 0
		world.agents = [Agent(self.dynamics_type) for i in range(self.num_agents)]
		world.init_safety_filter()
		world.scripted_agents = [Agent(scripted_agent_dynamics_type) for _ in range(self.num_scripted_agents)]
		for i, agent in enumerate(world.agents + world.scripted_agents):
			agent.id = i
			agent.name = f'agent {i}'
			agent.collide = True
			agent.silent = True
			agent.global_id = global_id
			global_id += 1

		global_id = self.init_landmarks(world, 
									args.num_landmarks, 
									global_id)
		world.scripted_agents_goals = [Landmark() for i in range(num_scripted_agents_goals)]
		
  		# add obstacles
		world.obstacles = [Landmark() for i in range(self.num_obstacles)]
		for i, obstacle in enumerate(world.obstacles):
			obstacle.name = f'obstacle {i}'
			obstacle.collide = True
			obstacle.movable = False
			obstacle.global_id = global_id
			global_id += 1
		## add wall
		# num obstacles per wall is twice the length of the wall
		wall_length = np.random.uniform(0.2, 0.8)
		# wall_length = 0.25
		self.wall_length = wall_length * self.world_size/4
		self.num_obstacles_per_wall = np.int32(1*self.wall_length/world.agents[0].size)
		self.num_walls = args.num_walls
		world.walls = [Wall() for i in range(self.num_walls)]
		for i, wall in enumerate(world.walls):
			wall.name = f'wall {i}'
			wall.collide = True
			wall.movable = False
			wall.global_id = global_id
			global_id += 1
		# num_walls = 1

		self.zeroshift = args.zeroshift
		self.reset_world(world)
		world.world_size = self.world_size
		world.world_aspect_ratio = self.world_aspect_ratio
		if hasattr(self, 'with_background'):
			world.with_background = self.with_background
		else:
			world.with_background = False
		return world

	def reset_world(self, world:World, num_current_episode: int = 0) -> None:
		# print("RESET WORLD")
		# metrics to keep track of
		world.current_time_step = 0
		world.simulation_time = 0.0
		# to track time required to reach goal
		world.times_required = -1 * np.ones(self.num_agents)
		world.dists_to_goal = -1 * np.ones(self.num_agents)
		# track distance left to the goal
		world.dist_left_to_goal = -1 * np.ones(self.num_agents)
		# number of times agents collide with stuff
		world.num_obstacle_collisions = np.zeros(self.num_agents)
		world.num_goal_collisions = np.zeros(self.num_agents)
		world.num_agent_collisions = np.zeros(self.num_agents)
		world.agent_dist_traveled = np.zeros(self.num_agents)
		# print("resetting world")
		#################### set colours ####################
		# set colours for agents
		for i, agent in enumerate(world.agents):
			# print("agent",i)
			if i%3 == 0:
				agent.color = np.array([0.85, 0.35, 0.35])
			elif i%3 == 1:
				agent.color = np.array([0.35, 0.85, 0.35])
			else:
				agent.color = np.array([0.35, 0.35, 0.85])
			agent.state.p_dist = 0.0
			agent.state.time = 0.0
			agent.initial_theta = None
		# set colours for scripted agents
		for i, agent in enumerate(world.scripted_agents):
			agent.color = np.array([0.15, 0.15, 0.15])
		# set colours for landmarks
		for i, landmark in enumerate(world.landmarks):
			landmark_agent = i % self.num_agents
			if landmark_agent%3 == 0:
				landmark.color = np.array([0.85, 0.35, 0.35])
			elif landmark_agent%3 == 1:
				landmark.color = np.array([0.35, 0.85, 0.35])
			else:
				landmark.color = np.array([0.35, 0.35, 0.85])
			# landmark.color = np.array([0.85, 0.85, 0.35])
		# set colours for scripted agents goals
		for i, landmark in enumerate(world.scripted_agents_goals):
			landmark.color = np.array([0.15, 0.95, 0.15])
		# set colours for obstacles
		for i, obstacle in enumerate(world.obstacles):
			obstacle.color = np.array([0.25, 0.25, 0.25])

		#####################################################
		self.update_curriculum(world, num_current_episode)
		self.random_scenario(world)
		self.initialize_min_time_distance_graph(world)
		self.initialize_landmarks_group_reached_goal(world)

	def update_engagement_distance_based_on_separation_distance(self, separation_distance:float) -> float:
		shift = separation_distance - self.engagement_distance_ref_separation_distance
		# print("Engagment distance set to ", self.engagement_distance_ref + shift)
		self.engagement_distance = self.engagement_distance_ref + shift

	def update_curriculum(self, world:World, num_current_episode:int) -> None:
		""" Update the curriculum learning parameters if necessary."""
		self.curriculum_ratio = np.clip(num_current_episode / self.num_total_episode, 0.0, 1.0)

		curriculum_ratio_sloped = self.get_effective_curriculum_ratio_sloped()
		curriculum_ratio_stair = self.get_effective_curriculum_ratio_stair()

		# Curriculum for performance.

		# print("multiple_engagement_rew_scaled",self.multiple_engagement_rew_scaled)
		# adjust heading threshold based on curriculum ratio
		self.goal_heading_error_thresh = self.goal_heading_error_thresh_init * (1.0 - curriculum_ratio_sloped) + self.goal_heading_error_thresh_target * curriculum_ratio_sloped
		self.goal_speed_error_thresh = self.goal_speed_error_thresh_init * (1.0 - curriculum_ratio_stair) + self.goal_speed_error_thresh_target * curriculum_ratio_stair
		self.min_dist_thresh = self.min_dist_thresh_init * (1.0 - curriculum_ratio_stair) + self.min_dist_thresh_target * curriculum_ratio_stair

		# Curriculum for safety reward.
		self.multiple_engagement_rew_scaled = self.potential_conflict_rew * curriculum_ratio_stair
		self.conflict_rew_scaled = self.safety_violation_rew * curriculum_ratio_stair

		self.diff_from_filtered_action_rew_scaled = self.diff_from_filtered_action_rew * curriculum_ratio_stair

		self.conflict_value_rew_scaled = self.hj_value_rew * curriculum_ratio_stair

		# Curriculum for safety filtering.
		# curriculum_ratio_separation_distance = self.get_effective_curriculum_ratio_sloped(start=0.2, end=0.75)
		curriculum_ratio_phase = self.get_effective_curriculum_ratio_stair(start=0.2, end=0.75, num_steps=4) * 0.5 * np.pi
		curriculum_ratio_separation_distance = 1 - np.cos(curriculum_ratio_phase)
		if (not self.initial_phase_safety_filter) and self.use_safety_filter:
			# initial phase is determined based on default curriculum_ratio_sloped:
			if curriculum_ratio_sloped > 0:
				world.use_safety_filter = True
			else:
				# initial phase
				world.use_safety_filter = False
			# separation increase has to start from 0 when curriculum starts.
			# curriculum_ratio_separation_distance = self.get_effective_curriculum_ratio_stair(start=0.36, end=0.8, num_steps=4)
			curriculum_ratio_stair = self.get_effective_curriculum_ratio_stair(start=0.36, end=0.8, num_steps=4)

		# self.separation_distance = self.separation_distance_init * (1.0 - curriculum_ratio_stair) + self.separation_distance_target * curriculum_ratio_stair
		self.separation_distance = self.separation_distance_init * (1.0 - curriculum_ratio_separation_distance) + self.separation_distance_target * curriculum_ratio_separation_distance
		world.update_safety_filter_separation_distance(self.separation_distance)
		self.update_engagement_distance_based_on_separation_distance(self.separation_distance)
		# print("Updating separation distance to", self.separation_distance)

	def initialize_min_time_distance_graph(self, world):
		for agent in world.agents:
			self.min_time(agent, world)
		world.calculate_distances()
		self.update_graph(world)

	def initialize_landmarks_group_reached_goal(self, world):
		# create a self.landmarks_group that has different sets of landmarks for each agent
		self.landmarks_group = np.array(np.split(np.array(world.landmarks), self.num_agents)).T
		# print("landmarks_group",self.landmarks_group)
		# create a flag that tells if the agent has reached the goal in a particular set of landmarks
		self.reached_goal = np.zeros(self.num_agents)

	def random_scenario(self, world):
		""" The specific implementation goes under Scenario class below.
  		"""
		raise NotImplementedError

	def info_callback(self, agent:Agent, world:World) -> Tuple:
		goal = self.get_agent_current_goal(agent, world)
		dist = np.sqrt(np.sum(np.square(agent.state.p_pos - 
										goal.state.p_pos)))
		
		# only update times_required for the first time it reaches the goal
		if self.evaluate_agent_goal_reached(agent, world) and (world.times_required[agent.id] == -1):
			# print("CALLBACKagent", agent.id, "reached goal",world.dist_left_to_goal[agent.id])
			world.times_required[agent.id] = world.current_time_step * world.dt
			world.dists_to_goal[agent.id] = agent.state.p_dist
			world.dist_left_to_goal[agent.id] = dist
			# print("dist to goal",world.dists_to_goal[agent.id],world.dists_to_goal)
		if world.times_required[agent.id] == -1:
			# print("agent", agent.id, "not reached goal yet",world.dist_left_to_goal[agent.id])
			world.dists_to_goal[agent.id] = agent.state.p_dist
			world.dist_left_to_goal[agent.id] = dist
		if agent.collide:
			if self.is_obstacle_collision(agent.state.p_pos, agent.size, world):
				world.num_obstacle_collisions[agent.id] += 1
			for a in world.agents:

				if a is agent:
					self.agent_dist_traveled[a.id] = a.state.p_dist
					self.agent_time_taken[a.id] = a.state.time
					# print("agent", a.id, "dist", a.state.p_dist, "time", a.state.time)
					continue
				if self.is_collision(agent, a):
					world.num_agent_collisions[agent.id] += 1

			# print("dist is",world.dists_to_goal[agent.id])
		world.dist_traveled_mean = np.mean(world.dists_to_goal)
		world.dist_traveled_stddev = np.std(world.dists_to_goal)
		# create the mean and stddev for time taken
		world.time_taken_mean = np.mean(world.times_required)
		world.time_taken_stddev = np.std(world.times_required)
		# world.dist_traveled_mean = np.mean(self.agent_dist_traveled)
		# world.dist_traveled_stddev = np.std(self.agent_dist_traveled)
		# world.agent_dist_traveled = self.agent_dist_traveled

		agent_info = {
			'id': agent.id,
			'position': agent.state.p_pos,
			'min_relative_distance': agent.min_relative_distance,
			'Dist_to_goal': world.dist_left_to_goal[agent.id],
			'Time_req_to_goal': world.times_required[agent.id],
			# NOTE: total agent collisions is half since we are double counting
			'Num_agent_collisions': world.num_agent_collisions[agent.id], 
			'Num_obst_collisions': world.num_obstacle_collisions[agent.id],
			'Distance_mean': world.dist_traveled_mean, 
			'Distance_variance': world.dist_traveled_stddev,
			'Mean_by_variance': world.dist_traveled_mean/(world.dist_traveled_stddev+0.0001),
			'Dists_traveled': world.dists_to_goal[agent.id],
			# 'Time_taken': self.agent_time_taken[agent.id], ## this may not be correct
			'Time_taken': world.times_required[agent.id],
			# Time mean and stddev
			'Time_mean': world.time_taken_mean,
			'Time_stddev': world.time_taken_stddev,
			'Time_mean_by_stddev': world.time_taken_mean/(world.time_taken_stddev+0.0001),
		}
		agent_info['Min_time_to_goal'] = agent.goal_min_time
		agent_info['Departed'] = agent.departed
		agent_info['Safety filtered'] = agent.safety_filtered
		agent_info['Safety violated'] = agent.min_relative_distance < self.separation_distance 
		# print("dist_left_to_goal", world.dist_left_to_goal[agent.id])
		return agent_info

	# check collision of entity with obstacles
	def is_obstacle_collision(self, pos, entity_size:float, world:World) -> bool:
		# pos is entity position "entity.state.p_pos"
		collision = False
		for obstacle in world.obstacles:
			delta_pos = obstacle.state.p_pos - pos
			dist = np.linalg.norm(delta_pos)
			dist_min = 1.05*(obstacle.size + entity_size)
			# print("1Dist", dist,"dist_min",dist_min, collision)

			if dist < dist_min:
				collision = True
				break	
		
		# check collision with walls
		for wall in world.walls:
			if wall.orient == 'H':
				# Horizontal wall, check for collision along the y-axis
				if 1.05*(wall.axis_pos - entity_size / 2) <= pos[1] <= 1.05*(wall.axis_pos + entity_size / 2):
					if 1.05*(wall.endpoints[0] - entity_size / 2) <= pos[0] <= 1.05*(wall.endpoints[1] + entity_size / 2):
						collision = True
						break
			elif wall.orient == 'V':
				# Vertical wall, check for collision along the x-axis
				if 1.05*(wall.axis_pos - entity_size / 2) <= pos[0] <= 1.05*(wall.axis_pos + entity_size / 2):
					if 1.05*(wall.endpoints[0] - entity_size / 2) <= pos[1] <= 1.05*(wall.endpoints[1] + entity_size / 2):
						collision = True
						# print("wall collision")
						break
		return collision


	# check collision of agent with other agents
	def check_agent_collision(self, pos, agent_size, agent_added) -> bool:
		collision = False
		if len(agent_added):
			for agent in agent_added:
				delta_pos = agent.state.p_pos - pos
				dist = np.linalg.norm(delta_pos)
				if dist < 1.05*(agent.size + agent_size):
					collision = True
					break
		return collision

	# check collision of agent with another agent
	def is_collision(self, agent1:Agent, agent2:Agent) -> bool:
		delta_pos = agent1.state.p_pos - agent2.state.p_pos
		dist = np.linalg.norm(delta_pos)
		dist_min = 1.05*(agent1.size + agent2.size)
		return True if dist < dist_min else False

	def is_confliction(self, agent1:Agent, agent2:Agent) -> bool:
		delta_pos = agent1.state.p_pos - agent2.state.p_pos
		dist = np.linalg.norm(delta_pos)
		return True if dist < self.separation_distance else False

	def is_in_engagement(self, agent1:Agent, agent2:Agent) -> bool:
		delta_pos = agent1.state.p_pos - agent2.state.p_pos
		dist = np.linalg.norm(delta_pos)
		return True if dist < self.engagement_distance else False

	def is_landmark_collision(self, pos, size:float, landmark_list:List) -> bool:
		collision = False
		for landmark in landmark_list:
			delta_pos = landmark.state.p_pos - pos
			dist = np.sqrt(np.sum(np.square(delta_pos)))
			dist_min = 1.05*(size + landmark.size)
			if dist < dist_min:
				collision = True
				break
		return collision

	# get min time required to reach to goal without obstacles
	def min_time(self, agent:Agent, world:World) -> float:
		assert agent.max_speed is not None, "Agent needs to have a max_speed."
		assert agent.max_speed > 0, "Agent max_speed should be positive."
		agent_id = agent.id
		# get the goal associated to this agent
		landmark = world.get_entity(entity_type='landmark', id=self.optimal_match_index[agent_id])
		dist = np.sqrt(np.sum(np.square(agent.state.p_pos - 
										landmark.state.p_pos)))
		min_time = dist / agent.max_speed
		agent.goal_min_time = min_time
		return min_time

	# done condition for each agent
	def done(self, agent:Agent, world:World) -> bool:
		# if we are using dones then return appropriate done
		# print("using dones")
		if self.use_dones:
			if world.current_time_step >= world.world_length:
				return True
			else:
				landmark = world.get_entity('landmark', self.optimal_match_index[agent.id])
				dist = np.sqrt(np.sum(np.square(agent.state.p_pos - 
												landmark.state.p_pos)))
				# # bipartite matching
				# world.dists = np.array([[np.linalg.norm(a.state.p_pos - l.state.p_pos) for l in world.landmarks]
				# 				   for a in world.agents])
				# # optimal 1:1 agent-landmark pairing (bipartite matching algorithm)
				# self.min_dists = self._bipartite_min_dists(world.dists)
				# if self.min_dists[agent.id] < self.min_dist_thresh:
				if dist < self.min_dist_thresh:
					return True
				else:
					return False
		# it not using dones then return done 
		# only when episode_length is reached
		else:
			if world.current_time_step >= world.world_length:
				return True
			else:
				return False


	def _bipartite_min_dists(self, dists):
		ri, ci = linear_sum_assignment(dists)
		# print("ri",ri,"ci",ci)
		min_dists = dists[ri, ci]
		return min_dists

	def agent_reached_all_goals(self, agent:Agent, world:World) -> bool:
		return self.reached_goal[agent.id] >= self.landmarks_group.shape[0]

	def get_agent_current_goal(self, agent:Agent, world:World) -> Landmark:
		goal_order = self.reached_goal[agent.id]*self.num_agents + agent.id
		# This means agent is done already. Return the last goal.
		if goal_order >= self.num_landmarks:
			goal_order = (self.reached_goal[agent.id] - 1) * self.num_agents + agent.id
		agent_goal = world.get_entity(entity_type='landmark', id=np.int8(goal_order))
		return agent_goal

	def get_agent_next_goal(self, agent:Agent, world:World) -> Landmark:
		goal_order = (self.reached_goal[agent.id] + 1)*self.num_agents + agent.id
		try:
			agent_goal = world.get_entity(entity_type='landmark', id=np.int8(goal_order))
		except:
			agent_goal = None
		return agent_goal

	def get_agent_reached_goal(self, agent:Agent) -> int:
		return self.reached_goal[agent.id]
 
	def agent_distance_to_current_goal(self, agent:Agent, world:World) -> float:
		agent_goal = self.get_agent_current_goal(agent, world)
		dist_to_goal = np.sqrt(np.sum(np.square(agent.state.p_pos - 
												agent_goal.state.p_pos)))
		return dist_to_goal

	def agent_heading_error_to_current_goal(self, agent:Agent, world:World) -> float:
		agent_goal = self.get_agent_current_goal(agent, world)
		heading_error = direction_alignment_error(agent.state.theta, agent_goal.heading)
		return heading_error

	@staticmethod
	def evaluate_goal_heading_condition_for_double_integrator(agent_pos, 
															agent_heading, 
															goal_pos, 
															goal_heading,
															goal_speed,
															min_dist_thresh, 
															goal_heading_error_thresh, 
															speed_advantage_thresh=0.2) -> bool:
		# custom condition for double integrator
		# the main purpose of this condition is to give bigger tolerance for heading error when goal speed is low
		assert speed_advantage_thresh > 0, "Speed advantage threshold should be positive."
		dist_to_goal = np.linalg.norm(agent_pos - goal_pos)
		heading_error = direction_alignment_error(agent_heading, goal_heading)
		if dist_to_goal > min_dist_thresh:
			return heading_error < goal_heading_error_thresh
		elif goal_speed > speed_advantage_thresh:
			return heading_error < goal_heading_error_thresh
		else:
			# if agent is close to goal, then give more tolerance for heading error when it is low speed.
			# reference: if goal speed = 0, we set the threshold to 0.5 (pi/2)
			# 0 if goal_speed = speed_advantage_thresh, 1 if goal_speed = 0
			speed_advantage = np.clip(1 - goal_speed / speed_advantage_thresh, 0, 1)
			threshold_speed_advantage = 0.5
			threshold_center = threshold_speed_advantage * speed_advantage + goal_heading_error_thresh * (1 - speed_advantage)

			# 0 if distance to goal = min_dist_thresh, 1 if distance to goal = 0
			distance_advantage = np.clip(1- dist_to_goal / min_dist_thresh, 0, 1)
			threshold_center_at_agent = threshold_center * distance_advantage + goal_heading_error_thresh * (1 - distance_advantage)

			return heading_error < threshold_center_at_agent

	def evaluate_agent_goal_reached(self, agent:Agent, world:World) -> bool:
		agent_goal = self.get_agent_current_goal(agent, world)
		dist_to_goal = self.agent_distance_to_current_goal(agent, world)
		heading_error = direction_alignment_error(agent.state.theta, agent_goal.heading)
		velocity_error = np.abs(agent.state.speed - agent_goal.speed)
		# print(f"agent {agent.id} dist to goal {dist_to_goal} / {self.min_dist_thresh} heading error {np.rad2deg(agent.state.theta-agent_goal.heading)}, values: {heading_error} / {self.goal_heading_error_thresh} speed error {velocity_error} / {self.goal_speed_error_thresh}")
		if self.dynamics_type == EntityDynamicsType.DoubleIntegratorXY:
			goal_heading_condition = self.evaluate_goal_heading_condition_for_double_integrator(agent.state.p_pos, 
																							agent.state.theta, 
																							agent_goal.state.p_pos, 
																							agent_goal.heading,
																							agent_goal.speed,
																							self.min_dist_thresh, 
																							self.goal_heading_error_thresh)
		else:
			goal_heading_condition = heading_error < self.goal_heading_error_thresh
		if dist_to_goal < self.min_dist_thresh and goal_heading_condition and velocity_error < self.goal_speed_error_thresh:
			return True
		return False

	def update_reached_goal_and_done(self, agent:Agent, world:World) -> None:
		if self.evaluate_agent_goal_reached(agent, world):
			agent_goal = self.get_agent_current_goal(agent, world)
			agent_goal.color = np.array([0.2, 0.2, 0.2])
			if self.use_masking:
				if not agent.done:
					self.reached_goal[agent.id] +=1
			else:
				self.reached_goal[agent.id] +=1
			

		if self.agent_reached_all_goals(agent, world):
			# print("All goals reached", agent.id)
			# if reached all goals, then set done to True and freeze agents.
			agent.done = True
			self.freeze_agent(agent)
			# agent.state.p_vel[1]=0.0
			# print("Agent velocity", agent.state.p_vel)

	def is_dubins_penalty_area(self, agent_pos, goal_pos, goal_heading) -> float:
		""" check if the agent is in the dubins penalty area.
			Penalty area is if agent is within the adjacent dubins circles of the goal,
			or if the agent is in front of the goal.
		"""
		circle_left, circle_right = get_adjacent_dubins_circles(goal_pos[0], goal_pos[1], goal_heading, self.min_turn_radius + self.min_dist_thresh)
		# check if agent is in the penalty area
		if np.linalg.norm(agent_pos - circle_left) < self.min_turn_radius:
			return True
		if np.linalg.norm(agent_pos - circle_right) < self.min_turn_radius:
			return True
		# check if agent is in front of the goal
		return is_in_front_of_ref_state(agent_pos[0], agent_pos[1], goal_pos[0], goal_pos[1], goal_heading)

	def reward_reach_goal(self, agent:Agent, world:World) -> float:
		rew = 0
		curriculum_ratio_sloped = self.get_effective_curriculum_ratio_sloped()
		curriculum_ratio_stair = self.get_effective_curriculum_ratio_stair()
		agent_goal = self.get_agent_current_goal(agent, world)
		if agent_goal.heading is None:
			raise ValueError("Goal heading is not set.")
		if agent_goal.speed is None:
			raise ValueError("Goal speed is not set.")
		goal_pos = agent_goal.state.p_pos
		goal_heading = agent_goal.heading
		goal_speed = agent_goal.speed
  
		agent_pos = agent.state.p_pos
		agent_heading = agent.state.theta
		agent_speed = agent.state.speed

		heading_error = direction_alignment_error(agent_heading, goal_heading)
		heading_performance_ratio = 1 - np.clip(heading_error / self.goal_heading_error_thresh, 0, 1)

		speed_error = np.abs(agent_speed - goal_speed)
		speed_error_max = self.goal_speed_error_thresh
		speed_error_normalized = np.clip(speed_error / speed_error_max, 0, 1)

		dist_to_goal = self.agent_distance_to_current_goal(agent, world)
		curriculum_ratio_airtaxi = self.get_effective_curriculum_ratio_sloped(start=0.25, end=0.75)
		if self.use_safety_filter:
			curriculum_ratio_airtaxi = 1
		if self.evaluate_agent_goal_reached(agent, world):
			speed_performance_ratio = 1 - speed_error_normalized
			agent_cross_track_error = cross_track_error(agent_pos, agent_heading, goal_pos)
			cross_track_performance_ratio = 1 - agent_cross_track_error
			# print(f"performance ratios: heading {heading_performance_ratio}, speed {speed_performance_ratio}, cross track {cross_track_performance_ratio}")
			performance_ratio = heading_performance_ratio * speed_performance_ratio * cross_track_performance_ratio
			if self.dynamics_type == EntityDynamicsType.DoubleIntegratorXY:
				goal_rew = self.goal_rew * performance_ratio
			else:
				preformance_ratio_curriculum = performance_ratio * curriculum_ratio_airtaxi + (1 - curriculum_ratio_airtaxi)
				goal_rew = self.goal_rew * preformance_ratio_curriculum
				# goal_rew = self.goal_rew
			# print(f"agent {agent.id} reached goal, reward: {goal_rew}")
			if self.use_masking:
				if not agent.done:
					rew += goal_rew
			else:
				rew += goal_rew
		# if not reached goal, penalize with distance to goal
		if not agent.done:
			relative_position = get_relative_position_from_reference(agent_pos, goal_pos, goal_heading)
			relative_heading = agent_heading - goal_heading
			if self.grid_ttr is None:
				if not self.use_safety_filter:
					heading_aware_penalty = 3 * double_integrator_velocity_error_from_magnetic_field_reference(agent.state, agent_goal, 2 * self.min_dist_thresh)
					# print(f"distance to goal: {dist_to_goal}, heading aware penalty: {heading_aware_penalty}")
					heading_aware_penalty = np.clip(1 - curriculum_ratio_sloped, 0, 1) * heading_aware_penalty
					rew -= heading_aware_penalty
			else:
				# Here, TTR is defined relative to goal, so goal is the reference.
				relative_state = np.array([relative_position[0], relative_position[1], relative_heading, agent.state.speed])
				try:
					ttr_goal = self.grid_ttr.interpolate(self.ttr_values, relative_state)
				except:
					ttr_goal = self.ttr_max
				if math.isnan(ttr_goal):
					ttr_goal = self.ttr_max
				# if ttr_goal >= self.ttr_max:
				# 	print(f"{dist_to_goal:.1f}")
				# range_distance_penalty = 6
				# range_ttr_penalty = 5
				# ttr_ratio = range_distance_penalty / self.ttr_max
				# ttr_penalty = ttr_ratio * ttr_goal
				# # if ttr_penalty >= 20:
				# # 	# print(f"{ttr_penalty:.1f}")
				# # 	print(f"{dist_to_goal:.1f}")
				# 	# ttr_penalty += dist_to_goal
				# # if distance is smaller than range_ttr_penalty, then we use ttr (1), if larger than range_distance_penalty, we use distance (0)
				# ttr_distance_mix_ratio = 1 - np.clip((dist_to_goal - range_ttr_penalty) / (range_distance_penalty - range_ttr_penalty), 0, 1)
				
				# rew -= ttr_distance_mix_ratio * ttr_penalty + (1 - ttr_distance_mix_ratio) * dist_to_goal
				rew -= 0.04 * ttr_goal

				# heading_aware_penalty = get_heading_aware_distance_penalty(relative_position)
				# heading_aware_penalty += 0.5 * self.curriculum_ratio * dist_to_goal
				# # print(f"distance to goal: {dist_to_goal}, heading aware penalty: {heading_aware_penalty}")
				# # rew -= dist_to_goal
				# rew -= heading_aware_penalty
			if self.dynamics_type == EntityDynamicsType.DoubleIntegratorXY:
				time_penalty_weight = 1.0
				if self.use_safety_filter:
					rew -= time_penalty_weight
				else:
				# rew -= speed_error_normalized * (1 - self.curriculum_ratio)
				# introduce time penalty
					rew -= time_penalty_weight * curriculum_ratio_sloped
			else:
				# time_penalty_weight = 1.0
				rew -= speed_error_normalized * curriculum_ratio_airtaxi
			# 	# rew -= time_penalty_weight * curriculum_ratio_sloped
			# rew -= dist_to_goal * (np.clip(curriculum_ratio_sloped, 0, 1) * 0.75 + 0.25)
		# print(f"agent {agent.id} reward: {rew}")
		return rew

	def reward_safety_violation(self, agent:Agent, world:World) -> float:
		rew = 0
		for a in world.agents:
			if a.id != agent.id and self.is_confliction(a, agent) and not a.done:
				rew += self.conflict_rew_scaled
		return rew

	def reward_multiple_engagement(self, agent:Agent, world:World) -> float:
		""" penalize when more than two agents are in close proximity.
		"""
		engagement_count = 0
		engagement_penalty = 0
		for a in world.agents:
			if a.id != agent.id and self.is_in_engagement(a, agent) and not a.done:
				relative_distance_vector = a.state.p_pos - agent.state.p_pos
				relative_distance = np.linalg.norm(relative_distance_vector)
				# 1 if smaller or equal to separation distance, 0 if larger or equal to engagement distance
				closeness_to_separation_distance = 1 - np.clip((relative_distance - self.separation_distance) / (self.engagement_distance - self.separation_distance), 0, 1)
				relative_distance_direction = np.arctan2(relative_distance_vector[1], relative_distance_vector[0])
				relative_distance_direction = np.array([np.cos(relative_distance_direction), np.sin(relative_distance_direction)])
				relative_velocity_vector = a.state.p_vel - agent.state.p_vel
				relative_distance_change = np.inner(relative_distance_direction, relative_velocity_vector)
				# penalize if distance is shrinking
				relative_distance_change = np.abs(min(0, relative_distance_change))
				engagement_penalty += relative_distance_change * closeness_to_separation_distance
				# engagement_penalty += relative_distance_change
				engagement_count += 1
		if engagement_count > 1:
			return self.multiple_engagement_rew_scaled * engagement_penalty
			# return self.multiple_engagement_rew_scaled * (engagement_count - 1)
		return 0

	def reward_diff_from_filtered_action(self, agent:Agent) -> float:
		if not agent.done:
			return self.diff_from_filtered_action_rew_scaled * agent.action_diff
		return 0

	def reward_hj_value(self, agent:Agent, world:World, eps_hj=0.4) -> float:
		rew = 0
		for a in world.agents:
			if a.id != agent.id and not a.done:
				value_at_relative_state = world.get_hj_value_between_two_agents(agent, a)
				conflict_value_penalty = np.abs(min(value_at_relative_state - eps_hj, 0))
				rew += self.conflict_value_rew_scaled * conflict_value_penalty
		return rew
				
	def reward(self, agent:Agent, world:World) -> float:
		# print(f"agent {agent.id} reward called")

		rew = self.reward_reach_goal(agent, world)
		if RewardBinaryConfig.SAFETY_VIOLATION:
			rew += self.reward_safety_violation(agent, world)
		if RewardBinaryConfig.POTENTIAL_CONFLICT:
			rew += self.reward_multiple_engagement(agent, world)
		if RewardBinaryConfig.DIFF_FROM_FILTERED_ACTION and self.use_safety_filter:
			rew += self.reward_diff_from_filtered_action(agent)
		if RewardBinaryConfig.HJ_VALUE:
			rew += self.reward_hj_value(agent, world)

		self.update_reached_goal_and_done(agent, world)
		return np.clip(rew, self.min_reward, self.max_reward)

	def observation(self, agent:Agent, world:World) -> arr:
		"""
			Return:
				[agent_vel, agent_pos, goal_pos]
		"""
		agent_position = agent.state.p_pos
		agent_heading = agent.state.theta
		agent_speed = agent.state.speed
		agent_velocity = agent.state.p_vel

		agents_goal = self.get_agent_current_goal(agent, world)
		goal_position = agents_goal.state.p_pos
		goal_heading = agents_goal.heading
		goal_speed = agents_goal.speed
		
		if agent.dynamics_type == EntityDynamicsType.KinematicVehicleXY or agent.dynamics_type == EntityDynamicsType.KinematicVehicleXY:
			return get_agent_observation_relative_with_heading(agent_position, agent_heading, agent_speed, goal_position, goal_heading, goal_speed)
		elif agent.dynamics_type == EntityDynamicsType.DoubleIntegratorXY:
			return get_agent_observation_relative_without_heading(agent_position, agent_velocity, goal_position, goal_heading, goal_speed)
		else:
			raise NotImplementedError(f"Unknown dynamics type")
	
	def get_id(self, agent:Agent) -> arr:
		return np.array([agent.global_id])

	def collect_rewards(self,world):
		"""
		This function collects the rewards of all agents at once to reduce further computations of the reward
		input: world and agent information
		output: list of agent names and array of corresponding reward values [agent_names, agent_rewards]
		"""
		agent_names = []
		agent_rewards = np.zeros(self.num_agents)
		count = 0
		for agent in world.agents:
			# print(agent.name)
			agent_names.append(agent.name)
			agent_rewards[count] = self.reward(agent, world)
			# print("Rew", agent_rewards[count])
			count +=1


		return agent_names, agent_rewards
	
	def collect_dist(self, world):
		"""
		This function collects the distances of all agents at once to reduce further computations of the reward
		input: world and agent information
		output: list of agent names and array of corresponding distance values [agent_names, agent_rewards]
		"""
		agent_names = []
		agent_dist = np.zeros(self.num_agents)
		agent_pos =  np.zeros((self.num_agents, 2)) # create a zero vector with the size of the number of agents and positions of dim 2
		count = 0
		for agent in world.agents:
			# print(agent.name)
			agent_names.append(agent.name)
			agent_dist[count] = agent.state.p_dist
			agent_pos[count]= agent.state.p_pos
			# print("Rew", agent_rewards[count])
			count +=1
		# mean_dist = np.mean(agent_dist)
		# std_dev_dist = np.std(agent_dist)
		return np.mean(agent_dist), np.std(agent_dist), agent_pos
	
	def sigmoid(x):
		return 1 / (1 + np.exp(-x))
	
	def collect_goal_info(self, world):
		goal_pos =  np.zeros((self.num_agents, 2)) # create a zero vector with the size of the number of goal and positions of dim 2
		count = 0
		for goal in world.landmarks:
			# print("goal" , goal.name, "at", goal.state.p_pos)
			goal_pos[count]= goal.state.p_pos
			count +=1
		return goal_pos

	def graph_observation(self, agent:Agent, world:World) -> Tuple[arr, arr]:
		"""
			FIXME: Take care of the case where edge_list is empty
			Returns: [node features, adjacency matrix]
			• Node features (num_entities, num_node_feats):
				If `global`: 
					• node features are global [pos, vel, goal, entity-type]
					• edge features are relative distances (just magnitude)
					NOTE: for `landmarks` and `obstacles` the `goal` is 
							the same as its position
				If `relative`:
					• node features are relative [pos, vel, goal, entity-type] to ego agents
					• edge features are relative distances (just magnitude)
					NOTE: for `landmarks` and `obstacles` the `goal` is 
							the same as its position
			• Adjacency Matrix (num_entities, num_entities)
				NOTE: using the distance matrix, need to do some post-processing
				If `global`:
					• All close-by entities are connectd together
				If `relative`:
					• Only entities close to the ego-agent are connected
			
		"""
		# num_entities = len(world.entities)
		# self.new_entities = [a for a in world.entities if not isinstance(a, Landmark)] ## used to remove landmarks from goals
		# node observations
		node_obs = []

		if world.graph_feat_type == 'global':			
			for i, entity in enumerate(world.entities):
				node_obs_i = self._get_entity_feat_global(entity, world)
				node_obs.append(node_obs_i)

		elif world.graph_feat_type == 'relative':
			for i, entity in enumerate(world.entities):
				# if 'agent' in entity.name and entity.done and entity.name != agent.name:
					# print("For Agent",agent.id, " Neighbor entity",entity.name, "with id", entity.id)
				node_obs_i = self._get_entity_feat_relative(agent, entity, world)
				node_obs.append(node_obs_i)


		node_obs = np.array(node_obs)
		adj = world.cached_dist_mag

		disconnected_mask = []
		# for agent entity, disconnect if it is done or not departed
		for entity in world.agents:
			disconnected = entity.done or not entity.departed
			disconnected_mask.append(disconnected)
		# for landmark agent, disconnect if it is reached by the agent.
		for (i_landmark, landmark) in enumerate(world.landmarks):
			landmark_agent_id = i_landmark % self.num_agents
			landmark_order = i_landmark // self.num_agents
			landmark_done = self.reached_goal[landmark_agent_id] > landmark_order
			disconnected_mask.append(landmark_done)

		adj[disconnected_mask, :] = 0   # Mask rows for done agents
		adj[:, disconnected_mask] = 0   # Mask columns for done agents

		connect_mask = ((adj < self.max_edge_dist) & (adj > 0)).astype(np.float32)
		adj = adj * connect_mask
		
		return node_obs, adj

	def update_graph(self, world:World):
		"""
			Construct a graph from the cached distances.
			Nodes are entities in the environment
			Edges are constructed by thresholding distances
		"""
		dists = world.cached_dist_mag
		# just connect the ones which are within connection 
		# distance and do not connect to itself
		connect = np.array((dists <= self.max_edge_dist) * \
							(dists > 0)).astype(int)
		sparse_connect = sparse.csr_matrix(connect)
		sparse_connect = sparse_connect.tocoo()
		row, col = sparse_connect.row, sparse_connect.col
		edge_list = np.stack([row, col])
		world.edge_list = edge_list
		if world.graph_feat_type == 'global':
			world.edge_weight = dists[row, col]
		elif world.graph_feat_type == 'relative':
			world.edge_weight = dists[row, col]
	
	def _get_entity_feat_global(self, entity:Entity, world:World) -> arr:
		"""
			Returns: ([velocity, position, goal_pos, entity_type])
			in global coords for the given entity
		"""
		pos = entity.state.p_pos
		vel = entity.state.p_vel
		if 'agent' in entity.name:
			goal_pos = world.get_entity('landmark', self.optimal_match_index[entity.id]).state.p_pos
			entity_type = entity_mapping['agent']
		elif 'landmark' in entity.name:
			goal_pos = pos
			entity_type = entity_mapping['landmark']
		elif 'obstacle' in entity.name:
			goal_pos = pos
			entity_type = entity_mapping['obstacle']
		else:
			raise ValueError(f'{entity.name} not supported')

		return np.hstack([vel, pos, goal_pos, entity_type])

	def _get_entity_feat_relative(self, agent:Agent, entity:Entity, world:World) -> arr:
		"""
			Returns: ([velocity, position, goal_pos, entity_type])
			in coords relative to the `agent` for the given entity
		"""
		reference_agent_state = agent.state
		if agent.dynamics_type == EntityDynamicsType.KinematicVehicleXY or agent.dynamics_type == EntityDynamicsType.KinematicVehicleXY:
			if 'agent' in entity.name:
				agent_state = entity.state
				agent_goal = self.get_agent_current_goal(entity, world)
				agent_goal_position = agent_goal.state.p_pos
				agent_goal_heading = agent_goal.heading
				agent_goal_speed = agent_goal.speed
				return get_agent_node_observation_relative_with_heading(agent_state, 
																agent_goal_position,
																agent_goal_heading,
																agent_goal_speed,
																reference_agent_state)
			elif 'landmark' in entity.name:
				landmark_position = entity.state.p_pos
				landmark_heading = entity.heading
				landmark_speed = entity.speed
				return get_landmark_node_observation_relative_with_heading(landmark_position,
																	landmark_heading,
																	landmark_speed,
																	reference_agent_state)
			else:
				raise ValueError(f'{entity.name} not supported')
		elif agent.dynamics_type == EntityDynamicsType.DoubleIntegratorXY:
			if 'agent' in entity.name:
				agent_state = entity.state
				agent_goal = self.get_agent_current_goal(entity, world)
				agent_goal_position = agent_goal.state.p_pos
				agent_goal_heading = agent_goal.heading
				agent_goal_speed = agent_goal.speed
				return get_agent_node_observation_relative_without_heading(agent_state, 
																agent_goal_position,
																agent_goal_heading,
																agent_goal_speed,
																reference_agent_state)
			elif 'landmark' in entity.name:
				landmark_position = entity.state.p_pos
				landmark_heading = entity.heading
				landmark_speed = entity.speed
				return get_landmark_node_observation_relative_without_heading(landmark_position,
																	landmark_heading,
																	landmark_speed,
																	reference_agent_state)
			else:
				raise ValueError(f'{entity.name} not supported')
		else:
			raise NotImplementedError(f"Unknown dynamics type")

	def freeze_agent(self, agent: Agent):
		if self.dynamics_type == EntityDynamicsType.KinematicVehicleXY:
			agent.state.speed=0.0
		elif self.dynamics_type == EntityDynamicsType.DoubleIntegratorXY:
			agent.state.p_vel = np.array([0.0, 0.0])
		elif self.dynamics_type == EntityDynamicsType.KinematicVehicleXY:
			agent.state.speed=0.0
		else:
			raise NotImplementedError

	def get_effective_curriculum_ratio_sloped(self, start=0.25, end=0.75):
		"""
			Linearly increase the curriculum ratio based on the slope.
			- curriculum_start: start ratio
			- curriculum_end: end ratio
		"""
		return np.clip(self.curriculum_ratio - start, 0, end - start) / (end - start)

	def get_effective_curriculum_ratio_stair(self, num_steps=4, start=0.2, end=0.75):
		"""
			Staircase increase the curriculum ratio based on the number of steps.
			- curriculum_start: start ratio
			- curriculum_end: end ratio
			- num_steps: number of steps
		"""
		if self.curriculum_ratio < start:
			return 0
		if self.curriculum_ratio > end:
			return 1
		# from 0 to (num_steps-1)
		continuous_val = (num_steps-1) * np.clip(self.curriculum_ratio - start, 0, end - start) / (end - start)
		return (1 + np.floor(continuous_val)) / num_steps

class RealisticScenario(SafeAamScenario):
	def __init__(self, scenario_image_file_name: str, km_in_pixel: float):
		super().__init__()
		self.scenario_image_file_name = scenario_image_file_name
		self.km_in_pixel = km_in_pixel
		image_path = 'multiagent/custom_scenarios/data/' + scenario_image_file_name
		try:
			# Open the image file
			with Image.open(image_path) as img:
				width, height = img.size
				self.world_width_pixel = width
				self.world_height_pixel = height
				img.close()
		except FileNotFoundError:
			raise FileNotFoundError(f"Image file {image_path} not found.")
		self.world_size = 0.5 * self.world_height_pixel / self.km_in_pixel
		self.with_background = True

	def convert_pixel_to_world_coordinates(self, pixel_pos: Tuple[int, int]):
		"""
			Convert pixel coordinates to world coordinates
			pixel coorindate: (0, 0) is the top-left corner
			world coordinate: (0, 0) is the center of the world
		"""
		x, y = pixel_pos
		world_x = (x - 0.5 * self.world_width_pixel) / self.km_in_pixel
		world_y = (0.5 * self.world_height_pixel - y) / self.km_in_pixel
		return np.array([world_x, world_y])

	def update_reached_goal_and_done(self, agent: Agent, world: World) -> None:
		if agent.departure_timer <= 0 and not agent.departed:
			# Depart only when the agent is far enough from other agents.
			if agent.min_relative_distance > self.separation_distance_target:
				# print(f"Agent {agent.id} min distance to other agents: {agent.min_relative_distance:.2f}")
				agent.state.reset_velocity(theta=agent.state.init_theta, speed=self.goal_speed_max)
				# agent.state.reset_velocity(theta=agent.state.init_theta)
				agent.state.c = np.zeros(world.dim_c)
				agent.departed = True
				print(f"agent {agent.id} is departing with speed {agent.state.speed}")
			else:
				print(f"Agent {agent.id} is not departing due to close proximity.")
		elif not agent.departed:
			# print(f"agent {agent.id} is departing in {agent.departure_timer} steps.")
			agent.departure_timer -= 1
			self.freeze_agent(agent)

		if self.evaluate_agent_goal_reached(agent, world):
			agent_goal = self.get_agent_current_goal(agent, world)
			agent_goal.color = np.array([0.2, 0.2, 0.2])
			if self.use_masking:
				if not agent.done:
					self.reached_goal[agent.id] += 1
			else:
				self.reached_goal[agent.id] += 1

			if self.agent_reached_all_goals(agent, world):
				print(f"agent {agent.id} reached all goals.")
				# if reached all goals, then set done to True and freeze agents.
				agent.done = True
				self.freeze_agent(agent)
			else:
				# call recursively to check if agent has reached all goals.
				self.update_reached_goal_and_done(agent, world)

class Scenario(SafeAamScenario):
	"""
		Scenario mainly used for Training.
	"""
	# in the training scenario
	def get_default_landmark_num_for_scenario(self):
		return 0

	def get_aspect_ratio_for_scenario(self) -> float:
		return 1.0

	def random_scenario(self, world) -> None:
		"""
			Randomly place agents and landmarks
		"""
		####### set random positions for entities ###########
		# set random static obstacles first
		for obstacle in world.obstacles:
			obstacle.state.p_pos = 0.8 * np.random.uniform(-self.world_size/2, 
															self.world_size/2, 
															world.dim_p)
			obstacle.state.stop()
		#####################################################

		# set agents at random positions not colliding with obstacles
		num_agents_added = 0
		agents_added = []
		boundary_thresh = 0.99
		curriculum_ratio_airtaxi = self.get_effective_curriculum_ratio_sloped(start=0.25, end=0.75)
		if self.use_safety_filter:
			curriculum_ratio_airtaxi = 1
		while True:
			if num_agents_added == self.num_agents:
				break
			### for random pos
			if self.dynamics_type == EntityDynamicsType.DoubleIntegratorXY:
				random_pos = np.random.uniform(-0.8 * self.world_size, 
												0.8 * self.world_size, 
												world.dim_p)
			else:
				random_pos_x_min = -0.5 * self.world_size
				random_pos_x_max = 0.25 * self.world_size * curriculum_ratio_airtaxi + 0.0 * (1 - curriculum_ratio_airtaxi) * self.world_size
				random_pos_y = np.random.uniform(-0.5 * self.world_size,
												0.5 * self.world_size)
				random_pos = np.array([np.random.uniform(random_pos_x_min, random_pos_x_max), random_pos_y])

			agent_size = world.agents[num_agents_added].size
			obs_collision = self.is_obstacle_collision(random_pos, agent_size, world)
			# goal_collision = self.is_goal_collision(uniform_pos, agent_size, world)

			# agent_collision = self.check_agent_collision(random_pos, agent_size, agents_added)
			# if not obs_collision and not agent_collision:
			if not obs_collision:
				world.agents[num_agents_added].state.p_pos = random_pos
				if self.dynamics_type == EntityDynamicsType.DoubleIntegratorXY:
					world.agents[num_agents_added].state.reset_velocity()
				else:
					random_speed = np.random.uniform(self.goal_speed_min, self.goal_speed_max)
					world.agents[num_agents_added].state.reset_velocity(speed=random_speed)
				world.agents[num_agents_added].state.c = np.zeros(world.dim_c)
				world.agents[num_agents_added].done = False
				agents_added.append(world.agents[num_agents_added])
				num_agents_added += 1
			# print(num_agents_added)
		agent_pos = [agent.state.p_pos for agent in world.agents]
		# print("agent pos:", agent_pos)

		#####################################################
		
		# set landmarks (goals) at random positions not colliding with obstacles 
		# and also check collisions with already placed goals
		landmark_domain_witdh = self.world_size
		list_of_agent_landmarks = []
		list_of_agent_landmark_headings = []
		list_of_agent_landmark_speeds = []
		previous_agent_goal_positions = None
		for i in range(self.num_agents):
			if self.dynamics_type == EntityDynamicsType.DoubleIntegratorXY:
				goal_positions = randomly_generate_separated_positions(
					self.num_landmark_per_agent,
					(-0.5 * landmark_domain_witdh, 0.5 * landmark_domain_witdh),
					(-0.5 * landmark_domain_witdh, 0.5 * landmark_domain_witdh),
					min_distance=0.25 * self.coordination_range,
					max_distance=0.75 * self.coordination_range
				)
				overlap_probability = 0.5
				if previous_agent_goal_positions is not None:
					# for each goal in goal_positions with the overlap probability, set the position to the previous agent's goal.
					for i in range(len(goal_positions)):
						if np.random.uniform(0, 1) < overlap_probability:
							goal_positions[i] = previous_agent_goal_positions[i]
			else:
				y_width = 0.1 * (1 - curriculum_ratio_airtaxi) + 0.5 * curriculum_ratio_airtaxi
				goal_positions = randomly_generate_separated_positions(
					self.num_landmark_per_agent,
					(0, 0.75 * landmark_domain_witdh),
					(-y_width * landmark_domain_witdh, y_width * landmark_domain_witdh),
					min_distance=0.5 * self.coordination_range,
					max_distance=self.coordination_range
				)
				overlap_probability = 0.5
				if previous_agent_goal_positions is not None:
					# for each goal in goal_positions with the overlap probability, set the position to the previous agent's goal.
					for i in range(len(goal_positions)):
						if np.random.uniform(0, 1) < overlap_probability:
							goal_positions[i] = previous_agent_goal_positions[i]
				# make first goal to be always at the left most.
				if goal_positions[0][0] > goal_positions[1][0]:
					goal_positions[0], goal_positions[1] = goal_positions[1], goal_positions[0]
			goal_headings = creat_relative_heading_list_from_goal_position_list(goal_positions)
			last_heading = deepcopy(goal_headings[-1])
			if self.use_safety_filter:
				curriculum_ratio = 1
			else:
				curriculum_ratio = self.get_effective_curriculum_ratio_sloped()
			if self.dynamics_type == EntityDynamicsType.KinematicVehicleXY:
				goal_speeds_fixed = self.goal_speed_max * np.ones(self.num_landmark_per_agent)
				# set last landmark speed to min.
				# goal_speeds_fixed[-1] = self.goal_speed_min
				# if self.num_landmark_per_agent > 2:
				# 	goal_speeds_fixed[-2] = 0.5 * (self.goal_speed_min + self.goal_speed_max)

				# # This randomization set random speed for each landmark.
				# goal_speeds_random = np.random.uniform(self.goal_speed_min, self.goal_speed_max, self.num_landmark_per_agent)

				# # curriculum_ratio = 1
				# # for the probability of curriculum_ratio, use random, else use fixed.
				# var_random = np.random.uniform(0, 1)
				# fixed_leftover_probability = 0.2
				# if var_random < min(curriculum_ratio, 1 - fixed_leftover_probability):
				# 	goal_speeds = goal_speeds_random
				# else:
				# 	goal_speeds = goal_speeds_fixed
				goal_speeds = goal_speeds_fixed
			else:
				goal_speeds_fixed = self.goal_speed_max * np.ones(self.num_landmark_per_agent)
				# set last landmark speed to min.
				goal_speeds_fixed[-1] = self.goal_speed_min

				# This randomization only mix the order of landmrk speeds among predefined candidates.
				# predefined_speed_candidates = np.linspace(self.goal_speed_min, self.goal_speed_max, 3)
				# goal_speeds_random = get_random_landmark_speeds_from_fixed_speed_candidates(self.num_landmark_per_agent, predefined_speed_candidates)

				# This randomization set random speed for each landmark.
				goal_speeds_random = np.random.uniform(self.goal_speed_min, self.goal_speed_max, self.num_landmark_per_agent)

				# curriculum_ratio = 1
				# for the probability of curriculum_ratio, use random, else use fixed.
				var_random = np.random.uniform(0, 1)
				fixed_leftover_probability = 0.2
				if var_random < min(curriculum_ratio, 1 - fixed_leftover_probability):
					goal_speeds = goal_speeds_random
				else:
					goal_speeds = goal_speeds_fixed

			
			# curriculum: perturb headings.
			if self.dynamics_type == EntityDynamicsType.DoubleIntegratorXY:
				for i in range(len(goal_headings)):
					perturbation_range = curriculum_ratio * 0.25 * np.pi
					goal_headings[i] += np.random.uniform(-perturbation_range, perturbation_range)
			else:
				for i in range(len(goal_headings)):
					perturbation_range = curriculum_ratio_airtaxi * 0.1 * np.pi
					goal_headings[i] += np.random.uniform(-perturbation_range, perturbation_range)				
			# last goal heading is specified as same as the one before perturbation.
			goal_headings.append(last_heading)
			list_of_agent_landmarks.append(goal_positions)
			list_of_agent_landmark_headings.append(goal_headings)
			list_of_agent_landmark_speeds.append(goal_speeds)
			previous_agent_goal_positions = goal_positions
		list_of_all_landmarks = map_each_agent_landmarks_to_entire_landmarks(list_of_agent_landmarks)
		list_of_all_landmark_headings = map_each_agent_landmarks_to_entire_landmarks(list_of_agent_landmark_headings)
		list_of_all_landmark_speeds = map_each_agent_landmarks_to_entire_landmarks(list_of_agent_landmark_speeds)
		for i, landmark in enumerate(world.landmarks):
			# print(f"landmark id {i}, p_pos {list_of_all_landmarks[i][0]}, {list_of_all_landmarks[i][1]}, heading: {list_of_all_landmark_headings[i]}")
			landmark.state.p_pos = list_of_all_landmarks[i]
			landmark.state.stop()
			landmark.heading = list_of_all_landmark_headings[i]
			landmark.speed = list_of_all_landmark_speeds[i]

		# num_goals_added = 0
		# while num_goals_added < self.num_landmarks:
		# 	landmark_order_per_agent = np.floor(num_goals_added / self.num_agents)
		# 	### for random pos
		# 	random_pos = boundary_thresh * np.random.uniform(-self.world_size/2, 
		# 										self.world_size/2, 
		# 										world.dim_p)

		# 	goal_size = world.landmarks[num_goals_added].size
		# 	obs_collision = self.is_obstacle_collision(random_pos, goal_size, world)
		# 	if not obs_collision:
		# 		world.landmarks[num_goals_added].state.p_pos = random_pos
		# 		world.landmarks[num_goals_added].state.stop()
		# 		if landmark_order_per_agent < self.num_landmark_per_agent-1:
		# 			world.landmarks[num_goals_added].heading = np.random.uniform(-np.pi, np.pi)
		# 		num_goals_added += 1


# actions: [None, ←, →, ↓, ↑, comm1, comm2]
if __name__ == "__main__":

	from multiagent.environment import MultiAgentGraphEnv
	from multiagent.policy import InteractivePolicy

	# makeshift argparser
	class Args:
		def __init__(self):
			self.num_agents:int=3
			self.world_size=2
			self.num_scripted_agents=0
			self.num_obstacles:int=3
			self.collaborative:bool=False 
			self.collision_rew:float=5
			self.goal_rew:float=20
			self.min_dist_thresh:float=0.1
			self.use_dones:bool=True
			self.episode_length:int=25
			self.max_edge_dist:float=1
			self.graph_feat_type:str='relative'
			# self.fair_wt=2
			# self.fair_rew=2
	args = Args()

	scenario = Scenario()
	# create world
	world = scenario.make_world(args)
	# create multiagent environment
	env = MultiAgentGraphEnv(world=world, reset_callback=scenario.reset_world, 
						reward_callback=scenario.reward, 
						observation_callback=scenario.observation, 
						graph_observation_callback=scenario.graph_observation,
						dynamics_type=world.dynamics_type,
						info_callback=scenario.info_callback, 
						done_callback=scenario.done,
						id_callback=scenario.get_id,
						update_graph=scenario.update_graph,
						shared_viewer=False)
	# render call to create viewer window
	env.render()
	# create interactive policies for each agent
	policies = [InteractivePolicy(env,i) for i in range(env.n)]
	# execution loop
	obs_n, agent_id_n, node_obs_n, adj_n = env.reset()
	stp=0

	prev_rewards = []
	while True:
		# query for action from each agent's policy
		act_n = []
		dist_mag = env.world.cached_dist_mag

		for i, policy in enumerate(policies):
			act_n.append(policy.action(obs_n[i]))
		# step environment
		# print(act_n)
		obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n, info_n = env.step(act_n)
		prev_rewards= reward_n

		# render all agent views
		env.render()
		stp+=1
		# display rewards