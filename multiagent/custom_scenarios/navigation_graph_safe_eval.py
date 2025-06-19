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
import scipy.spatial.distance as dist

import os,sys
sys.path.append(os.path.abspath(os.getcwd()))

from multiagent.config import eval_scenario_type
from multiagent.custom_scenarios.navigation_graph_safe import SafeAamScenario
from multiagent.custom_scenarios.utils import map_each_agent_landmarks_to_entire_landmarks, creat_relative_heading_list_from_goal_position_list, randomly_generate_separated_positions
from multiagent.core import EntityDynamicsType
from multiagent.config import AirTaxiConfig

entity_mapping = {'agent': 0, 'landmark': 1, 'obstacle':2, 'wall':3}
	

class Scenario(SafeAamScenario):
	def __init__(self):
		super().__init__()
		self.eval_scenario_type = eval_scenario_type
		assert self.eval_scenario_type in ["circular_config", "left_to_right_merge", "left_to_right_cross", "bottom_to_top_merge", "random", "left_to_right_merge_and_land", "bottom_to_top_merge_and_land", "three_vehicle_conflicting_example", "two_vehicle_conflicting_example"], "Invalid scenario type"

	def random_scenario(self, world) -> None:
		if self.eval_scenario_type == "circular_config":
			self.scenario_circular_config(world)
		elif self.eval_scenario_type == "left_to_right_merge":
			self.scenario_random_left_to_right_merge(world)
		elif self.eval_scenario_type == "bottom_to_top_merge":
			self.scenario_random_bottom_to_top_merge(world)
		elif self.eval_scenario_type == "left_to_right_cross":
			self.scenario_left_to_right_cross(world)
		elif self.eval_scenario_type == "left_to_right_merge_and_land":
			self.scenario_random_left_to_right_merge_and_land(world)
		elif self.eval_scenario_type == "bottom_to_top_merge_and_land":
			self.scenario_random_bottom_to_top_merge_and_land(world)
		elif self.eval_scenario_type == "three_vehicle_conflicting_example":
			self.scenario_three_vehicle_conflicting_example(world)
		elif self.eval_scenario_type == "two_vehicle_conflicting_example":
			self.scenario_two_vehicle_conflicting_example(world)
		else:
			raise NotImplementedError

	def get_default_landmark_num_for_scenario(self) -> int:
		# number of landmark per each agent
		num_left_to_right_cross_landmarks = 3
		num_circular_config_landmarks = 1
		num_random_merge_landmarks = 2
		num_merge_and_land_landmarks = 3
		if self.eval_scenario_type == "circular_config":
			return num_circular_config_landmarks
		elif self.eval_scenario_type == "left_to_right_merge" or self.eval_scenario_type == "bottom_to_top_merge":
			return num_random_merge_landmarks
		elif self.eval_scenario_type == "left_to_right_cross":
			return num_left_to_right_cross_landmarks
		elif self.eval_scenario_type == "left_to_right_merge_and_land":
			return num_merge_and_land_landmarks
		elif self.eval_scenario_type == "bottom_to_top_merge_and_land":
			return num_merge_and_land_landmarks
		elif self.eval_scenario_type == "three_vehicle_conflicting_example":
			return 1
		elif self.eval_scenario_type == "two_vehicle_conflicting_example":
			return 1
		else:
			raise NotImplementedError

	def get_aspect_ratio_for_scenario(self) -> float:
		ar_merge_left_to_right = 2.0
		ar_merge_bottom_to_top = 0.5
		ar_merge_bottom_to_top_land = 1.0
		ar_cross = 1.0
		ar_circular = 1.0
		if self.eval_scenario_type == "circular_config":
			return ar_circular
		elif self.eval_scenario_type == "left_to_right_merge":
			return ar_merge_left_to_right
		elif self.eval_scenario_type == "bottom_to_top_merge":
			return ar_merge_bottom_to_top
		elif self.eval_scenario_type == "left_to_right_cross":
			return ar_cross
		elif self.eval_scenario_type == "left_to_right_merge_and_land":
			return ar_merge_left_to_right
		elif self.eval_scenario_type == "bottom_to_top_merge_and_land":
			return ar_merge_bottom_to_top_land
		elif self.eval_scenario_type == "three_vehicle_conflicting_example":
			return 1.0
		elif self.eval_scenario_type == "two_vehicle_conflicting_example":
			return 1.0
		else:
			raise NotImplementedError

	def scenario_circular_config(self, world):
		"""	Scenario: agent and landmark positioned on a circle
			first landmark is at the opposite side of the agent, heading pointing outward.
			second landmark is returning to the initial state, heading pointing inward.
			third landmark is at the opposite side of the agent, heading pointing outward.
		"""
		# # circle arrangement
		agent_theta = np.linspace(0, 2*np.pi, self.num_agents, endpoint=False)
		radius = 0.92 * self.world_size/2        
		# set agent positions on the circle
		for i, agent in enumerate(world.agents):
			agent.state.p_pos = np.array([radius*np.cos(agent_theta[i]), radius*np.sin(agent_theta[i])])
			# initial heading is pointing inward.
			agent.state.reset_velocity(theta=agent_theta[i] + np.pi)
			agent.state.c = np.zeros(world.dim_c)
		# set first group of landmarks
		for i in range(self.num_agents):
			landmark = world.landmarks[i]
			landmark.state.p_pos = -world.agents[i].state.p_pos
			landmark.state.stop()
			landmark.heading = agent_theta[i] + np.pi
			landmark.speed = 0.5 * (self.goal_speed_max + self.goal_speed_min)
		# # set second group of landmarks
		# for i in range(self.num_agents):
		# 	landmark = world.landmarks[self.num_agents + i]
		# 	landmark.state.p_pos = world.agents[i].state.p_pos
		# 	landmark.state.stop()
		# 	landmark.heading = agent_theta[i] + np.pi
		# 	landmark.speed = 0.5 * (self.goal_speed_max + self.goal_speed_min)
		# # set third group of landmarks
		# for i in range(self.num_agents):
		# 	landmark = world.landmarks[2*self.num_agents + i]
		# 	landmark.state.p_pos = -world.agents[i].state.p_pos
		# 	landmark.state.stop()
		# 	landmark.heading = agent_theta[i] + np.pi
		# 	landmark.speed = 0.5 * (self.goal_speed_max + self.goal_speed_min)

	def scenario_random_left_to_right_merge(self, world):
		layout_unit_height = 0.25 * self.world_size
		layout_unit_width = 0.25 * self.world_size * self.world_aspect_ratio
		## set agents at the left side of the environment.
		init_positions = randomly_generate_separated_positions(self.num_agents, (-2 * layout_unit_width, -layout_unit_width), (-2 * layout_unit_height, 2 * layout_unit_height), 1.5 * self.separation_distance)
		pos_y = np.linspace(-2 * layout_unit_height, 2 * layout_unit_height, self.num_agents)
		init_positions = []
		for i in range(self.num_agents):
			init_positions.append(np.array([-1.5 * layout_unit_width, pos_y[i]]))

		print("init position:",init_positions)
		
		# set initial states for agents
		for i, agent in enumerate(world.agents):
			agent.state.p_pos = init_positions[i]
			agent.state.reset_velocity(theta=0)
			agent.state.c = np.zeros(world.dim_c)
			agent.done = False

		# the first goal point is at the center and the second goal point is at the right side of the environment
		goal_positions = [np.array([0,0]), np.array([layout_unit_width,0])]
		goal_headings = creat_relative_heading_list_from_goal_position_list(goal_positions)
		goal_headings.append(goal_headings[-1])
		if self.dynamics_type == EntityDynamicsType.KinematicVehicleXY:
			goal_speeds = [self.goal_speed_max, self.goal_speed_max]
		else:
			goal_speeds = [self.goal_speed_max, self.goal_speed_min]
		list_of_agent_landmarks = [goal_positions for _ in range(self.num_agents)]
		list_of_agent_landmark_headings = [goal_headings for _ in range(self.num_agents)]
		list_of_agent_landmark_speeds = [goal_speeds for _ in range(self.num_agents)]
		list_of_all_landmarks = map_each_agent_landmarks_to_entire_landmarks(list_of_agent_landmarks)
		list_of_all_landmark_headings = map_each_agent_landmarks_to_entire_landmarks(list_of_agent_landmark_headings)
		list_of_all_landmark_speeds = map_each_agent_landmarks_to_entire_landmarks(list_of_agent_landmark_speeds)
		# set goal positions
		for i, landmark in enumerate(world.landmarks):
			# print(f"landmark id {i}, p_pos {list_of_all_landmarks[i][0]}, {list_of_all_landmarks[i][1]}, heading: {list_of_all_landmark_headings[i]}, speed: {list_of_all_landmark_speeds[i]}")
			landmark.state.p_pos = list_of_all_landmarks[i]
			landmark.state.stop()
			landmark.heading = list_of_all_landmark_headings[i]
			landmark.speed = list_of_all_landmark_speeds[i]

	def scenario_random_left_to_right_merge_and_land(self, world):
		layout_unit_height = 0.25 * self.world_size
		layout_unit_width = 0.25 * self.world_size * self.world_aspect_ratio
		even_pos_y = np.linspace(-2 * layout_unit_height, 2 * layout_unit_height, self.num_agents)

		## set agents at the left side of the environment.
		init_positions = randomly_generate_separated_positions(self.num_agents, (-2 * layout_unit_width, -0.5 * layout_unit_width), (-2 * layout_unit_height, 2 * layout_unit_height), 1.5 * self.separation_distance)
		# init_positions = []
		# for i in range(self.num_agents):
		# 	init_positions.append(np.array([-1.5 * layout_unit_width, even_pos_y[i]]))

		# print("init position:",init_positions)
		
		# set initial states for agents
		for i, agent in enumerate(world.agents):
			agent.state.p_pos = init_positions[i]
			agent.state.reset_velocity(theta=0)
			agent.state.c = np.zeros(world.dim_c)
			agent.done = False

		# the first goal point is at the center and the second goal point is at the right side of the environment
		common_goal_positions = [np.array([0,0]), np.array([layout_unit_width,0])]
		landing_positions = []
		for i in range(self.num_agents):
			landing_positions.append(np.array([2 * layout_unit_width, even_pos_y[i]])
		)
		# print("corridor waypoints:",common_goal_positions)

		goal_headings = creat_relative_heading_list_from_goal_position_list(common_goal_positions)
		goal_headings.append(goal_headings[-1])
		goal_speed_mid = 0.5 * (self.goal_speed_max + self.goal_speed_min)
		goal_speeds = [goal_speed_mid, goal_speed_mid, self.goal_speed_min]
		list_of_each_agent_landmarks = []
		list_of_each_agent_landmark_headings = []
		list_of_each_agent_landmark_speeds = []
		for i, agent in enumerate(world.agents):
			ith_agent_landmarks = common_goal_positions + [landing_positions[i]]
			list_of_each_agent_landmarks.append(ith_agent_landmarks)
			ith_agent_landmark_headings = creat_relative_heading_list_from_goal_position_list(ith_agent_landmarks)
			ith_agent_landmark_headings.append(ith_agent_landmark_headings[-1])
			list_of_each_agent_landmark_headings.append(ith_agent_landmark_headings)
			list_of_each_agent_landmark_speeds.append(goal_speeds)
			list_of_all_landmark_positions = map_each_agent_landmarks_to_entire_landmarks(list_of_each_agent_landmarks)
			list_of_all_landmark_headings = map_each_agent_landmarks_to_entire_landmarks(list_of_each_agent_landmark_headings)
			list_of_all_landmark_speeds = map_each_agent_landmarks_to_entire_landmarks(list_of_each_agent_landmark_speeds)
		for i, landmark in enumerate(world.landmarks):
			# print(f"landmark id {i}, p_pos {list_of_all_landmarks[i][0]}, {list_of_all_landmarks[i][1]}, heading: {list_of_all_landmark_headings[i]}, speed: {list_of_all_landmark_speeds[i]}")
			landmark.state.p_pos = list_of_all_landmark_positions[i]
			landmark.state.stop()
			landmark.heading = list_of_all_landmark_headings[i]
			landmark.speed = list_of_all_landmark_speeds[i]

	def scenario_random_bottom_to_top_merge_and_land(self, world):
		# this scenario is mainly to emulate the crazyflie experiment.
		waypoint_interval = 1.5
		initial_landmark_position = 1.0 # y position
		landing_spot_width = 3.0 # x width
		init_spot_position = -1.0 # y position
		init_spot_width = 2.0 #x width
		# this is needed since in the drone room, origin is not at the center.
		landscape_shift_y = 1.5

		common_goal_positions = [np.array([0.0, initial_landmark_position + waypoint_interval * i - landscape_shift_y]) for i in range(self.num_landmark_per_agent-1)]
		landing_pos_y = initial_landmark_position + (self.num_landmark_per_agent-1) * waypoint_interval
		landing_pos_x = np.linspace(-landing_spot_width/2, landing_spot_width/2, self.num_agents)
		landing_positions = [np.array([x, landing_pos_y - landscape_shift_y]) for x in landing_pos_x]
		init_positions = randomly_generate_separated_positions(self.num_agents, (-init_spot_width/2, init_spot_width/2), (init_spot_position - landscape_shift_y, init_spot_position + 0.5 - landscape_shift_y), self.separation_distance)
		# init_pos_x = np.linspace(-init_spot_width/2, init_spot_width/2, self.num_agents)
		# init_positions = [np.array([x, init_spot_position - landscape_shift_y]) for x in init_pos_x]

		list_of_each_agent_landmarks = []
		list_of_each_agent_landmark_headings = []
		list_of_each_agent_landmark_speeds = []
		for i in range(self.num_agents):
			ith_agent_landmarks = common_goal_positions + [landing_positions[i]]
			list_of_each_agent_landmarks.append(ith_agent_landmarks)
			ith_agent_landmark_headings = creat_relative_heading_list_from_goal_position_list(ith_agent_landmarks)
			ith_agent_landmark_headings.append(ith_agent_landmark_headings[-1])
			list_of_each_agent_landmark_headings.append(ith_agent_landmark_headings)
			ith_agent_landmark_speeds = [0.5 for _ in range(self.num_landmark_per_agent-1)] + [0.1]
			list_of_each_agent_landmark_speeds.append(ith_agent_landmark_speeds)

		list_of_all_landmark_positions = map_each_agent_landmarks_to_entire_landmarks(list_of_each_agent_landmarks)
		list_of_all_landmark_headings = map_each_agent_landmarks_to_entire_landmarks(list_of_each_agent_landmark_headings)
		list_of_all_landmark_speeds = map_each_agent_landmarks_to_entire_landmarks(list_of_each_agent_landmark_speeds)
		# set initial states for agents
		for i, agent in enumerate(world.agents):
			agent.state.p_pos = init_positions[i]
			agent.state.reset_velocity(theta=0)
			agent.state.c = np.zeros(world.dim_c)
			agent.done = False

		for i, landmark in enumerate(world.landmarks):
			print(f"landmark id {i}, p_pos {list_of_all_landmark_positions[i][0]}, {list_of_all_landmark_positions[i][1]}, heading: {list_of_all_landmark_headings[i]}, speed: {list_of_all_landmark_speeds[i]}")
			landmark.state.p_pos = list_of_all_landmark_positions[i]
			landmark.state.stop()
			landmark.heading = list_of_all_landmark_headings[i]
			landmark.speed = list_of_all_landmark_speeds[i]

	def scenario_random_bottom_to_top_merge(self, world):
		""" This is basically the same scenario as left_to_right_merge but in vertical configuration.
			It can be used to test the symmetry of the agent policies.
		"""
		layout_unit_height = 0.25 * self.world_size
		layout_unit_width = 0.25 * self.world_size * self.world_aspect_ratio
		## set agents at the bottom side of the environment.
		# init_positions = randomly_generate_separated_positions(self.num_agents, (-2 * layout_unit_width, 2 * layout_unit_width), (-2 * layout_unit_height, -layout_unit_height), 1.5 * self.separation_distance)
		pos_x = np.linspace(2 * layout_unit_width, -2 * layout_unit_width, self.num_agents)
		init_positions = []
		for i in range(self.num_agents):
			init_positions.append(np.array([pos_x[i], -1.5 * layout_unit_height]))
		
		# set initial states for agents
		for i, agent in enumerate(world.agents):
			agent.state.p_pos = init_positions[i]
			agent.state.reset_velocity(theta=np.pi/2)
			agent.state.c = np.zeros(world.dim_c)
			agent.done = False

		# the first goal point is at the center and the second goal point is at the right side of the environment
		goal_positions = [np.array([0,0]), np.array([0, layout_unit_height])]
		goal_headings = creat_relative_heading_list_from_goal_position_list(goal_positions)
		goal_headings.append(goal_headings[-1])
		if self.dynamics_type == EntityDynamicsType.KinematicVehicleXY:
			goal_speeds = [self.goal_speed_max, self.goal_speed_max]
		else:
			goal_speeds = [self.goal_speed_max, self.goal_speed_min]
		list_of_agent_landmarks = [goal_positions for _ in range(self.num_agents)]
		list_of_agent_landmark_headings = [goal_headings for _ in range(self.num_agents)]
		list_of_agent_landmark_speeds = [goal_speeds for _ in range(self.num_agents)]
		list_of_all_landmarks = map_each_agent_landmarks_to_entire_landmarks(list_of_agent_landmarks)
		list_of_all_landmark_headings = map_each_agent_landmarks_to_entire_landmarks(list_of_agent_landmark_headings)
		list_of_all_landmark_speeds = map_each_agent_landmarks_to_entire_landmarks(list_of_agent_landmark_speeds)
		# print(list_of_agent_landmarks)
		# set goal positions
		for i, landmark in enumerate(world.landmarks):
			# print(f"landmark id {i}, p_pos {list_of_all_landmarks[i][0]}, {list_of_all_landmarks[i][1]}, heading: {list_of_all_landmark_headings[i]}, speed: {list_of_all_landmark_speeds[i]}")
			landmark.state.p_pos = list_of_all_landmarks[i]
			landmark.state.stop()
			landmark.heading = list_of_all_landmark_headings[i]
			landmark.speed = list_of_all_landmark_speeds[i]

	def scenario_three_vehicle_conflicting_example(self, world):
		assert self.num_agents == 3, "This scenario is only for 3 agents."
		assert self.num_landmark_per_agent == 1, "This scenario is only for 1 landmark per agent."
		assert self.dynamics_type == EntityDynamicsType.KinematicVehicleXY, "This scenario is only for AirTaxi dynamics."
		shift_x = 0.0

		v_nom = AirTaxiConfig.V_NOMINAL
		# agent1_pos = np.array([-0.06 + shift_x, 0.0])
		agent1_pos = np.array([0.4 + shift_x, 0.0])
		agent1_heading = 0.0
		agent1_speed = v_nom

		world.agents[0].state.p_pos = agent1_pos
		world.agents[0].state.theta = agent1_heading
		world.agents[0].state.speed = agent1_speed
		world.agents[0].state.c = np.zeros(world.dim_c)
		world.agents[0].done = False

		agent2_pos = np.array([1.7 + shift_x, 0.3])
		agent2_heading = 4 * np.pi / 3
		agent2_speed = v_nom

		world.agents[1].state.p_pos = agent2_pos
		world.agents[1].state.theta = agent2_heading
		world.agents[1].state.speed = agent2_speed
		world.agents[1].state.c = np.zeros(world.dim_c)
		world.agents[1].done = False

		# agent3_pos = np.array([2.2 + shift_x, -1.2])
		# agent3_heading = 2 * np.pi / 3
		# agent3_speed = AirTaxiConfig.V_MAX - 0.005
		agent3_pos = np.array([1.6 + shift_x, -0.6])
		agent3_heading = -np.pi
		agent3_speed = self.goal_speed_min

		world.agents[2].state.p_pos = agent3_pos
		world.agents[2].state.theta = agent3_heading
		world.agents[2].state.speed = agent3_speed
		world.agents[2].state.c = np.zeros(world.dim_c)
		world.agents[2].done = False

		landmark_distance = 4.0
		landmark1_pos = np.array([agent1_pos[0] + landmark_distance, 0.0])
		landmark1_heading = 0.0

		world.landmarks[0].state.p_pos = landmark1_pos
		world.landmarks[0].state.stop()
		world.landmarks[0].heading = landmark1_heading
		world.landmarks[0].speed = self.goal_speed_max

		landmark2_pos = agent2_pos + np.array([np.cos(agent2_heading), np.sin(agent2_heading)]) * landmark_distance
		world.landmarks[1].state.p_pos = landmark2_pos
		world.landmarks[1].state.stop()
		world.landmarks[1].heading = agent2_heading
		world.landmarks[1].speed = self.goal_speed_max

		landmark3_pos = agent3_pos + np.array([np.cos(agent3_heading), np.sin(agent3_heading)]) * landmark_distance

		world.landmarks[2].state.p_pos = landmark3_pos
		world.landmarks[2].state.stop()
		world.landmarks[2].heading = agent3_heading
		world.landmarks[2].speed = self.goal_speed_max

	def scenario_two_vehicle_conflicting_example(self, world):
		""" This example shows that only with two agent, deconfliction is possible if the relative state is outside of the unsafe region.
		"""
		assert self.num_agents == 2, "This scenario is only for 3 agents."
		assert self.num_landmark_per_agent == 1, "This scenario is only for 1 landmark per agent."
		assert self.dynamics_type == EntityDynamicsType.KinematicVehicleXY, "This scenario is only for AirTaxi dynamics."
		shift_x = 0.0
		v_nom = AirTaxiConfig.V_NOMINAL

		# agent1_pos = np.array([-0.5 + shift_x, 0.0])
		agent1_pos = np.array([0.4 + shift_x, 0.0])
		agent1_heading = 0.0
		agent1_speed = v_nom

		world.agents[0].state.p_pos = agent1_pos
		world.agents[0].state.theta = agent1_heading
		world.agents[0].state.speed = agent1_speed
		world.agents[0].state.c = np.zeros(world.dim_c)
		world.agents[0].done = False

		# opponent state candidate 1
		agent2_pos = np.array([1.7 + shift_x, 0.3])
		agent2_heading = 4 * np.pi / 3
		agent2_speed = v_nom
		# opponent state candidate 2
		# agent2_pos = np.array([2.2 + shift_x, -1.2])
		# agent2_heading = 2 * np.pi / 3
		# agent2_speed = AirTaxiConfig.V_MAX - 0.005

		world.agents[1].state.p_pos = agent2_pos
		world.agents[1].state.theta = agent2_heading
		world.agents[1].state.speed = agent2_speed
		world.agents[1].state.c = np.zeros(world.dim_c)
		world.agents[1].done = False

		landmark_distance = 3.5
		landmark1_pos = np.array([agent1_pos[0] + landmark_distance, 0.0])
		landmark1_heading = 0.0

		world.landmarks[0].state.p_pos = landmark1_pos
		world.landmarks[0].state.stop()
		world.landmarks[0].heading = landmark1_heading
		world.landmarks[0].speed = self.goal_speed_max

		landmark2_pos = agent2_pos + np.array([np.cos(agent2_heading), np.sin(agent2_heading)]) * landmark_distance
		world.landmarks[1].state.p_pos = landmark2_pos
		world.landmarks[1].state.stop()
		world.landmarks[1].heading = agent2_heading
		world.landmarks[1].speed = self.goal_speed_max


	def scenario_left_to_right_cross(self, world):
		""" Scenario: agent positioned at the left side of the environment and landmarks at the right side.
		"""
		assert self.num_agents == 2, "This scenario is only for 2 agents."
		# print("Num landmarks",self.num_landmarks)
		assert self.num_landmarks % self.num_agents == 0, "Number of landmarks must be divisible by the number of agents."
		# set agents at the left side of the environment.
		num_agents_added = 0
		agents_added = []
		boundary_thresh = 0.99
		# y positions for agents
		uniform_pos = np.linspace(boundary_thresh*self.world_size/4, -boundary_thresh*self.world_size/4, self.num_agents)
		# uniform_pos = np.linspace(boundary_thresh*self.world_size/2, -boundary_thresh*self.world_size/2, self.num_agents)
		# agent init x position
		agent_x = -boundary_thresh*self.world_size/2

		# set agent positions
		while True:
			if num_agents_added == self.num_agents:
				break

			agent_y_pos = [uniform_pos[num_agents_added]]
			line_pos = np.insert(np.array(agent_y_pos), -1, agent_x)

			agent_size = world.agents[num_agents_added].size
			agent_collision = self.check_agent_collision(line_pos, agent_size, agents_added)
			if not agent_collision:
				world.agents[num_agents_added].state.p_pos = line_pos
				world.agents[num_agents_added].state.reset_velocity(theta=0)
				world.agents[num_agents_added].state.c = np.zeros(world.dim_c)
				world.agents[num_agents_added].done = False
				agents_added.append(world.agents[num_agents_added])
				num_agents_added += 1
		
		# Set landmark (goal) positions on the right side
		landmarks_per_agent = self.num_landmarks // self.num_agents
		goal_x = boundary_thresh * self.world_size / 2

		for i in range(self.num_landmarks//self.num_agents):
			# Generate a set of y-positions for each agent’s landmarks, evenly spaced
			landmark_y_positions = np.linspace(
				-boundary_thresh * self.world_size / (self.num_landmarks - i),
				boundary_thresh * self.world_size / (self.num_landmarks - i),
				self.num_agents
			)
			# print("landmark_y_positions",landmark_y_positions)
			for j, y in enumerate(landmark_y_positions):
				# print("j",j,"y",y)
				landmark_index = i * self.num_agents + j  # Compute the landmark index dynamically
				# print("landmark_index",landmark_index)
				goal_pos = np.insert([y], 0, -1.0 / (1 + landmark_index//2) + goal_x / (landmarks_per_agent - landmark_index//2))
				# print("goal_pos",goal_pos)
				world.landmarks[landmark_index].state.p_pos = goal_pos
				world.landmarks[landmark_index].state.stop()


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
