import gym
from gym import spaces
import numpy as np
import math
import random
from typing import Callable, List, Tuple, Dict, Union, Optional
from multiagent.core import EntityDynamicsType, World, Agent, is_list_of_lists
from multiagent.multi_discrete import MultiDiscrete
from multiagent.config import DoubleIntegratorConfig, AirTaxiConfig
from pyglet import image

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentBaseEnv(gym.Env):
	"""
		Base environment for all multi-agent environments
	"""
	metadata = {
		'render.modes' : ['human', 'rgb_array']
	}

	def __init__(self, world:World, reset_callback:Callable=None, 
					reward_callback:Callable=None,
					observation_callback:Callable=None, 
					info_callback:Callable=None,
					done_callback:Callable=None, 
					agent_reached_goal_callback:Callable=None,
					shared_viewer:bool=True, 
					discrete_action:bool=True,
					scenario_name:str='navigation',
     				dynamics_type:str='airtaxi') -> None:
		self.world = world
		self.world_length = self.world.world_length
		self.world_aspect_ratio = self.world.world_aspect_ratio
		self.current_step = 0
		self.agents = self.world.policy_agents

		# set required vectorized gym env property
		self.n = len(world.policy_agents)
		self.num_agents = len(world.policy_agents)  # for compatibility with offpolicy baseline envs
		# scenario callbacks
		self.reset_callback = reset_callback
		# print("reward_callback: ", reward_callback)
		self.reward_callback = reward_callback
		self.observation_callback = observation_callback
		self.info_callback = info_callback
		self.done_callback = done_callback
		self.agent_reached_goal_callback = agent_reached_goal_callback
		# print("done_callback: ", done_callback)
		self.scenario_name = scenario_name

		self.world_size = self.world.world_size
		self.with_background = self.world.with_background

		# environment parameters
		# self.discrete_action_space = True
		self.discrete_action_space = discrete_action
		if dynamics_type == 'double_integrator':
			self.dynamics_type = EntityDynamicsType.DoubleIntegratorXY
			# self.num_action_options = DoubleIntegratorConfig.ACTION_SPACE
			self.num_accel_y_options = DoubleIntegratorConfig.ACCELY_OPTIONS
			self.num_accel_x_options = DoubleIntegratorConfig.ACCELX_OPTIONS
			self.num_discrete_action = self.num_accel_x_options * self.num_accel_y_options
		elif dynamics_type == 'airtaxi':
			self.dynamics_type = EntityDynamicsType.KinematicVehicleXY
			self.num_accel_options = AirTaxiConfig.MOTION_PRIM_ACCEL_OPTIONS
			self.num_angle_rate_options = AirTaxiConfig.MOTION_PRIM_ANGRATE_OPTIONS
			self.num_discrete_action = self.num_accel_options * self.num_angle_rate_options
		else:
			raise NotImplementedError
		# if true, action is a number 0...N, 
		# otherwise action is a one-hot N-dimensional vector
		self.discrete_action_input = False
		# if true, even the action is continuous, 
		# action will be performed discretely
		self.force_discrete_action = world.discrete_action if hasattr(world, 
												'discrete_action') else False
		# if true, every agent has the same reward
		self.shared_reward = world.collaborative if hasattr(world, 
													'collaborative') else False
		self.time = 0

		# configure spaces
		self.action_space = []
		self.observation_space = []
		self.share_observation_space = []   # adding this for compatibility with MAPPO code
		share_obs_dim = 0
		if is_list_of_lists(world.agents):
			for team in world.agents:
				for agent in team:
					total_action_space = []

					# physical action space
					if self.discrete_action_space:
						u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
					else:
						u_action_space = spaces.Box(low=-agent.u_range, 
													high=+agent.u_range, 
													shape=(world.dim_p,), 
													dtype=np.float32)
					if agent.movable:
						total_action_space.append(u_action_space)

					# communication action space
					if self.discrete_action_space:
						c_action_space = spaces.Discrete(world.dim_c)
					else:
						c_action_space = spaces.Box(low=0.0, 
													high=1.0, 
													shape=(world.dim_c,), 
													dtype=np.float32)

					if not agent.silent:
						total_action_space.append(c_action_space)
					# total action space
					if len(total_action_space) > 1:
						# all action spaces are discrete, 
						# so simplify to MultiDiscrete action space
						if all([isinstance(act_space, spaces.Discrete) 
								for act_space in total_action_space]):
							act_space = MultiDiscrete([[0, act_space.n - 1] 
												for act_space in total_action_space])
						else:
							act_space = spaces.Tuple(total_action_space)
						self.action_space.append(act_space)
					else:
						self.action_space.append(total_action_space[0])

					# observation space
					# for original MPE Envs like simple_spread, simple_reference, etc.
					if 'simple' in self.scenario_name:
						obs_dim = len(observation_callback(agent=agent, world=self.world))
					else:
						obs_dim = len(observation_callback(agent=agent, world=self.world))
					share_obs_dim += obs_dim
					self.observation_space.append(spaces.Box(low=-np.inf, 
															high=+np.inf, 
															shape=(obs_dim,), 
															dtype=np.float32))

					agent.action.c = np.zeros(self.world.dim_c)
		else:
			for agent in world.agents:
				total_action_space = []

				# physical action space
				if self.discrete_action_space:
					u_action_space = spaces.Discrete(self.num_discrete_action)
				else:
					u_action_space = spaces.Box(low=-agent.u_range, 
												high=+agent.u_range, 
												shape=(world.dim_p,), 
												dtype=np.float32)
				if agent.movable:
					total_action_space.append(u_action_space)

				# communication action space
				if self.discrete_action_space:
					c_action_space = spaces.Discrete(world.dim_c)
				else:
					c_action_space = spaces.Box(low=0.0, 
												high=1.0, 
												shape=(world.dim_c,), 
												dtype=np.float32)

				if not agent.silent:
					total_action_space.append(c_action_space)
				# total action space
				if len(total_action_space) > 1:
					# all action spaces are discrete, 
					# so simplify to MultiDiscrete action space
					if all([isinstance(act_space, spaces.Discrete) 
							for act_space in total_action_space]):
						act_space = MultiDiscrete([[0, act_space.n - 1] 
											for act_space in total_action_space])
						# print("total_action_space1", total_action_space, act_space)
					else:
						act_space = spaces.Tuple(total_action_space)
					self.action_space.append(act_space)
				else:
					self.action_space.append(total_action_space[0])
					# print("total_action_space3", total_action_space, self.action_space)

				# observation space
				# for original MPE Envs like simple_spread, simple_reference, etc.
				if 'simple' in self.scenario_name:
					obs_dim = len(observation_callback(agent=agent, world=self.world))
				else:
					obs_dim = len(observation_callback(agent=agent, world=self.world))
				share_obs_dim += obs_dim
				self.observation_space.append(spaces.Box(low=-np.inf, 
														high=+np.inf, 
														shape=(obs_dim,), 
														dtype=np.float32))

				agent.action.c = np.zeros(self.world.dim_c)			
		
		self.share_observation_space = [spaces.Box(low=-np.inf, 
													high=+np.inf, 
													shape=(share_obs_dim,), 
													dtype=np.float32) 
													for _ in range(self.n)]
		

		# rendering
		self.shared_viewer = shared_viewer
		if self.shared_viewer:
			self.viewers = [None]
		else:
			self.viewers = [None] * self.n
		self._reset_render()

	def seed(self, seed=None):
		if seed is None:
			np.random.seed(1)
		else:
			np.random.seed(seed)
	
	def step(self, action_n:List):
		raise NotImplementedError

	def reset(self):
		raise NotImplementedError

	# get info used for benchmarking
	def _get_info(self, agent:Agent) -> Dict:
		if self.info_callback is None:
			return {}
		return self.info_callback(agent, self.world)

	# get observation for a particular agent
	def _get_obs(self, agent:Agent) -> np.ndarray:
		if self.observation_callback is None:
			return np.zeros(0)
		# for original MPE Envs like simple_spread, simple_reference, etc.
		if 'simple' in self.scenario_name:
			return self.observation_callback(agent=agent, world=self.world)
		else:
			return self.observation_callback(agent=agent, world=self.world)

	# get shared observation for the environment
	def _get_shared_obs(self) -> np.ndarray:
		if self.shared_obs_callback is None:
			return None
		return self.shared_obs_callback(self.world)
		
	# # get dones for a particular agent
	# # unused right now -- agents are allowed to go beyond the viewing screen
	# def _get_done(self, agent:Agent) -> bool:
	# 	# print("self.done_callback: ", self.done_callback(agent, self.world))
	# 	if self.done_callback is None:
	# 		if self.current_step >= self.world_length:
	# 			return True
	# 		else:
	# 			return False
	# 	return self.done_callback(agent, self.world)



	def _get_done(self, agent, count = None):
		if agent.done:
			# print(‘did it enter agent status so its returnng true’)
			return True
		# print(‘current step: ’+str(self.current_step)+‘self world lenght ’ + str(self.world_length))
		if self.current_step >= self.world_length:
			return True
		else:
			return False

	# get reward for a particular agent
	def _get_reward(self, agent:Agent) -> float:
		if self.reward_callback is None:
			return 0.0
		return self.reward_callback(agent, self.world)

	def decode_action_index(self, action_index):
		""" action_index: numpy array of the index of the action in the action space
		"""
		if self.dynamics_type == EntityDynamicsType.DoubleIntegratorXY:
			max_accel_x = DoubleIntegratorConfig.ACCELX_MAX
			min_accel_x = DoubleIntegratorConfig.ACCELX_MIN
			max_accel_y = DoubleIntegratorConfig.ACCELY_MAX
			min_accel_y = DoubleIntegratorConfig.ACCELY_MIN
			accel_x_options = np.linspace(min_accel_x, max_accel_x, self.num_accel_x_options)
			accel_y_options = np.linspace(min_accel_y, max_accel_y, self.num_accel_y_options)
			accel_x_index = action_index // self.num_accel_y_options #check
			accel_y_index = action_index % self.num_accel_y_options #check

			u = np.zeros((*action_index.shape, self.world.dim_p))

			u[..., 0] = accel_x_options[accel_x_index]
			u[..., 1] = accel_y_options[accel_y_index]
		elif self.dynamics_type == EntityDynamicsType.KinematicVehicleXY:
			max_angular_rate = AirTaxiConfig.ANGULAR_RATE_MAX
			max_accel = AirTaxiConfig.ACCEL_MAX
			min_accel = AirTaxiConfig.ACCEL_MIN
			accel_options = np.linspace(min_accel, max_accel, self.num_accel_options)
			angle_rate_options = np.linspace(-max_angular_rate, max_angular_rate, self.num_angle_rate_options)
			angle_rate_index = action_index // self.num_accel_options
			accel_index = action_index % self.num_accel_options
			# Initialize the output array with the correct shape
			u = np.zeros((*action_index.shape, self.world.dim_p))

			# Use advanced indexing to fill in the values
			u[..., 0] = angle_rate_options[angle_rate_index]
			u[..., 1] = accel_options[accel_index]
		else:
			raise NotImplementedError
		return u

	# set env action for a particular agent
	def _set_action(self, action, agent:Agent, action_space, 
					time:Optional=None) -> None:
		agent.action.u = np.zeros(self.world.dim_p)
		agent.action.c = np.zeros(self.world.dim_c)
		# process action
		if isinstance(action_space, MultiDiscrete):
			act = []
			size = action_space.high - action_space.low + 1
			index = 0
			for s in size:
				act.append(action[index:(index+s)])
				index += s
			action = act
			# print("multi discrete action space", action)
		else:
			# print("else action_space", agent.id, action)
			if not isinstance(action, list):
				action = [action]
		# actions: [None, ←, →, ↓, ↑, comm1, comm2]
		if agent.movable:
			## physical action
			## print(f'discrete_action_input: {self.discrete_action_input}, force_discrete_action: {self.force_discrete_action}, discrete_action_space: {self.discrete_action_space}')
			action_description = ""

			if self.discrete_action_input:
				## print("Discrete action input",self.discrete_action_input)
				agent.action.u = np.zeros(self.world.dim_p)
				# process discrete action
				if action[0] == 1: agent.action.u[0] = -1.0
				if action[0] == 2: agent.action.u[0] = +1.0
				if action[0] == 3: agent.action.u[1] = -1.0
				if action[0] == 4: agent.action.u[1] = +1.0

			## actions: [accel, omega] whose values can be wither 0, 1, -1
			## actions :[0,0], [0,1], [0,-1], [1,0], [1,1], [1,-1], [-1,0], [-1,1], [-1,-1]
			## create motion primitive based on action
			# if self.discrete_action_input:
			# 	print("Discrete acd ../ction input",self.discrete_action_input)
			# 	agent.action.u = np.zeros(self.world.dim_p)
			# 	# process discrete action
			# 	if action[0] == 0: 
			# 		agent.action.u[0] = 0.0
			# 		agent.action.u[1] = 0.0
			# 	if action[0] == 1: 
			# 		agent.action.u[0] = 0.0
			# 		agent.action.u[1] = 1.0
			# 	if action[0] == 2: 
			# 		agent.action.u[0] = 0.0
			# 		agent.action.u[1] = -1.0
			# 	if action[0] == 3:
			# 		agent.action.u[0] = 1.0
			# 		agent.action.u[1] = 0.0
			# 	if action[0] == 4:
			# 		agent.action.u[0] = 1.0
			# 		agent.action.u[1] = 1.0
			# 	if action[0] == 5:
			# 		agent.action.u[0] = 1.0
			# 		agent.action.u[1] = -1.0
			# 	if action[0] == 6:
			# 		agent.action.u[0] = -1.0
			# 		agent.action.u[1] = 0.0
			# 	if action[0] == 7:
			# 		agent.action.u[0] = -1.0
			# 		agent.action.u[1] = 1.0
			# 	if action[0] == 8:
			# 		agent.action.u[0] = -1.0
			# 		agent.action.u[1] = -1.0

			else:
				if self.force_discrete_action:
					# print("force_discrete_action",self.force_discrete_action)
					d = np.argmax(action[0])
					action[0][:] = 0.0
					action[0][d] = 1.0
				if self.discrete_action_space:
					if self.dynamics_type == EntityDynamicsType.DoubleIntegratorXY:
						accel_x_max = DoubleIntegratorConfig.ACCELX_MAX
						accel_y_max = DoubleIntegratorConfig.ACCELY_MAX
						accel_x_options = np.linspace(-accel_x_max, accel_x_max, self.num_accel_x_options)
						accel_y_options = np.linspace(-accel_y_max, accel_y_max, self.num_accel_y_options)
						action_index = np.argmax(action[0])
						accel_x_index = int(action_index // self.num_accel_y_options)
						accel_y_index = int(action_index - accel_x_index * self.num_accel_y_options)
						agent.action.u[0] = accel_x_options[accel_x_index]
						agent.action.u[1] = accel_y_options[accel_y_index]
						action_description = f"accel x: {agent.action.u[0]}, y: {agent.action.u[1]}"
					elif self.dynamics_type == EntityDynamicsType.KinematicVehicleXY:
						agent.action.u = np.zeros(self.world.dim_p)
						max_angular_rate = AirTaxiConfig.ANGULAR_RATE_MAX
						max_accel = AirTaxiConfig.ACCEL_MAX
						min_accel = AirTaxiConfig.ACCEL_MIN
						accel_options = np.linspace(min_accel, max_accel, self.num_accel_options)
						angle_rate_options = np.linspace(-max_angular_rate, max_angular_rate, self.num_angle_rate_options)
						action_index = np.argmax(action[0])
						angle_rate_index = int(action_index // self.num_accel_options)
						accel_index = int(action_index - angle_rate_index * self.num_accel_options)
						agent.action.u[0] = angle_rate_options[angle_rate_index]
						agent.action.u[1] = accel_options[accel_index]
						action_description = f"turn: {agent.action.u[0]}, accel: {agent.action.u[1]}"
					else:
						raise NotImplementedError
				else:
					# print("else not discrete_action_space",action)
					agent.action.u = action[0]

			# print("agent.action.u",agent.action.u)
			# NOTE: refer offpolicy/envs/mpe/environment.py -> MultiAgentEnv._set_action() for non-silent agent
			action = action[1:]
		if not agent.silent:
			# communication action
			if self.discrete_action_input:
				agent.action.c = np.zeros(self.world.dim_c)
				agent.action.c[action[0]] = 1.0
			else:
				agent.action.c = action[0]
			action = action[1:]
		# make sure we used all elements of action
		assert len(action) == 0

	# reset rendering assets
	def _reset_render(self) -> None:
		self.render_geoms = None
		self.render_geoms_xform = None

	# render environment
	def render(self, mode:str='human', close:bool=False) -> List:
		if close:
			# close any existic renderers
			for i, viewer in enumerate(self.viewers):
				if viewer is not None:
					viewer.close()
				self.viewers[i] = None
			return []

		if mode == 'human':
			alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
			message = ''
			if is_list_of_lists(self.world.agents):
				for team in self.world.agents:
					for agent in team:
						comm = []
						for other in team:
							if other is agent: continue
							if np.all(other.state.c == 0):
								word = '_'
							else:
								word = alphabet[np.argmax(other.state.c)]
							message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
			else:
				for agent in self.world.agents:
					comm = []
					for other in self.world.agents:
						if other is agent: continue
						if np.all(other.state.c == 0):
							word = '_'
						else:
							word = alphabet[np.argmax(other.state.c)]
						message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
			# print(message)

		default_height = 1100
		for i in range(len(self.viewers)):
			# create viewers (if necessary)
			if self.viewers[i] is None:
				# import rendering only if we need it 
				# (and don't import for headless machines)
				# from gym.envs.classic_control import rendering
				from multiagent import rendering
				self.viewers[i] = rendering.Viewer(self.world_aspect_ratio * default_height,
												   default_height)
    
		cam_range_height = self.world_size
		cam_range_width = self.world_size * self.world_aspect_ratio
		separation_distance = self.world.separation_distance_target

		# create rendering geometry
		if self.render_geoms is None:
			# import rendering only if we need it 
			# (and don't import for headless machines)
			# from gym.envs.classic_control import rendering
			from multiagent import rendering
			self.render_geoms = []
			self.render_geoms_xform = []

			self.comm_geoms = []
			self.agent_separation_geoms = []
			self.agent_separation_xforms = []

			for entity in self.world.entities:
				geom = rendering.make_circle(entity.size)
				xform = rendering.Transform()

				entity_comm_geoms = []

				if 'agent' in entity.name:
					# for each agent make a rectangle based on entity.size and entity.state.theta. providethe four vertices
					# width is entity.size and height is entity.size*2
					width = entity.size*2
					height = entity.size
					theta = entity.state.theta
					entity.initial_theta = theta
					# print("theta",theta)
					# Define the vertices relative to the center (0, 0)
					vertices = np.array([
						[-width/2, -height/2],
						[width/2, -height/2],
						[width/2, height/2],
						[-width/2, height/2]
					])

					# Create rotation matrix
					rotation_matrix = np.array([
						[np.cos(theta), -np.sin(theta)],
						[np.sin(theta), np.cos(theta)]
					])

					# Apply rotation to all vertices at once
					rotated_vertices = np.dot(vertices, rotation_matrix.T)

					geom = rendering.make_polygon(rotated_vertices)
					geom.set_color(*entity.color, alpha=0.8)

					if not entity.silent:
						dim_c = self.world.dim_c
						# make circles to represent communication
						for ci in range(dim_c):
							comm = rendering.make_circle(entity.size / dim_c)
							comm.set_color(1, 1, 1)
							comm.add_attr(xform)
							offset = rendering.Transform()
							comm_size = (entity.size / dim_c)
							offset.set_translation(ci * comm_size * 2 -
													entity.size + comm_size, 0)
							comm.add_attr(offset)
							entity_comm_geoms.append(comm)

					geom_separation = rendering.make_circle(separation_distance)
					geom_separation.set_color(0, 0, 0, alpha=0.1)
					xform_separation = rendering.Transform()
					geom_separation.add_attr(xform_separation)

				elif 'landmark' in entity.name:
					if self.with_background:
						if entity.heading is not None:
							geom = rendering.make_triangle_pointer(0.5 * self.world.min_dist_thresh, 0.8 * self.world.min_dist_thresh, entity.heading)
						geom.set_color(*entity.color, alpha=0.5)
					else:
						if entity.heading is not None:
							geom = rendering.make_triangle_pointer(0.33 * self.world.min_dist_thresh, 0.5 * self.world.min_dist_thresh, entity.heading)
						geom.set_color(*entity.color, alpha=0.5)
					# this is actually not separation, but radius of the proximity of the landmark
					geom_separation = rendering.make_circle(self.world.min_dist_thresh)
					geom_separation.set_color(*entity.color, alpha=0.1)
					xform_separation = rendering.Transform()
					geom_separation.add_attr(xform_separation)
				else:
					geom.set_color(*entity.color, alpha=0.1)
					if entity.channel is not None:
						dim_c = self.world.dim_c
						# make circles to represent communication
						for ci in range(dim_c):
							comm = rendering.make_circle(entity.size / dim_c)
							comm.set_color(1, 1, 1)
							comm.add_attr(xform)
							offset = rendering.Transform()
							comm_size = (entity.size / dim_c)
							offset.set_translation(ci * comm_size * 2 -
													entity.size + comm_size, 0)
							comm.add_attr(offset)
							entity_comm_geoms.append(comm)
					geom_separation = []
					xform_separation = []
				geom.add_attr(xform)
				self.render_geoms.append(geom)
				self.render_geoms_xform.append(xform)
				self.comm_geoms.append(entity_comm_geoms)
				self.agent_separation_geoms.append(geom_separation)
				self.agent_separation_xforms.append(xform_separation)

			for wall in self.world.walls:
				corners = ((wall.axis_pos - 0.5 * wall.width, wall.endpoints[0]),
						   (wall.axis_pos - 0.5 *
							wall.width, wall.endpoints[1]),
						   (wall.axis_pos + 0.5 *
							wall.width, wall.endpoints[1]),
						   (wall.axis_pos + 0.5 * wall.width, wall.endpoints[0]))
				if wall.orient == 'H':
					corners = tuple(c[::-1] for c in corners)
				geom = rendering.make_polygon(corners)
				if wall.hard:
					geom.set_color(*wall.color, alpha=0.1)
				else:
					geom.set_color(*wall.color, alpha=0.5)
				self.render_geoms.append(geom)

			# add geoms to viewer
			# for viewer in self.viewers:
			#     viewer.geoms = []
			#     for geom in self.render_geoms:
			#         viewer.add_geom(geom)

			# Background.
			if self.scenario_name == 'navigation_graph_safe_bayarea':
				bg_image = rendering.Image('multiagent/custom_scenarios/data/bayarea.jpg', 
											2 * cam_range_width, 2 * cam_range_height)
				bg_xform = rendering.Transform()
				bg_image.add_attr(bg_xform)
			elif self.scenario_name == 'navigation_graph_safe_bayarea_big':
				bg_image = rendering.Image('multiagent/custom_scenarios/data/bayarea_big.jpg', 
											2 * cam_range_width, 2 * cam_range_height)
				bg_xform = rendering.Transform()
				bg_image.add_attr(bg_xform)
			elif self.scenario_name == 'navigation_graph_safe_bayarea_merge':
				bg_image = rendering.Image('multiagent/custom_scenarios/data/bayarea_merge.jpg', 
											2 * cam_range_width, 2 * cam_range_height)
				bg_xform = rendering.Transform()
				bg_image.add_attr(bg_xform)
			elif self.scenario_name == 'navigation_graph_safe_bayarea_cross':
				bg_image = rendering.Image('multiagent/custom_scenarios/data/bayarea_cross.jpg', 
											2 * cam_range_width, 2 * cam_range_height)
				bg_xform = rendering.Transform()
				bg_image.add_attr(bg_xform)
			else:
				bg_image = None

			if self.with_background:
				bg_text_background = rendering.make_rectangle(8.4, 5)
				bg_text_background.set_color(0, 0, 0, alpha=0.75)
				bg_text_background_xform = rendering.Transform()
				bg_text_background_xform.set_translation(cam_range_width - 4.7, cam_range_height- 4)
				bg_text_background.add_attr(bg_text_background_xform)
			else:
				bg_text_background = None
			# bg_text = rendering.Text("Test", 0, 0, font_size=1)

			for viewer in self.viewers:
				viewer.geoms = []
				if self.with_background and bg_image:
					viewer.add_geom(bg_image)
				for geom in self.render_geoms:
					viewer.add_geom(geom)
				for entity_comm_geoms in self.comm_geoms:
					for geom in entity_comm_geoms:
						viewer.add_geom(geom)
				for geom in self.agent_separation_geoms:
					if geom:
						viewer.add_geom(geom)
				if self.with_background:
					viewer.add_geom(bg_text_background)
				# viewer.add_geom(bg_text)

		results = []
		for i in range(len(self.viewers)):
			from multiagent import rendering

			if self.shared_viewer:
				pos = np.zeros(self.world.dim_p)
			else:
				pos = self.agents[i].state.p_pos
			self.viewers[i].set_bounds(pos[0]- cam_range_width,
										pos[0]+ cam_range_width,
										pos[1]-cam_range_height,
										pos[1]+cam_range_height)
			# update geometry positions
			for e, entity in enumerate(self.world.entities):

				self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
				if 'agent' in entity.name:
					# Get the change in orientation
					delta_theta = entity.state.theta - entity.initial_theta
					# print("theta",entity.state.theta)
					# print("delta_theta",delta_theta)
					# input("Press Enter to continue...")
					self.render_geoms_xform[e].set_rotation(delta_theta)
					# entity.initial_theta = entity.state.theta
					alpha = 1.0 if self.with_background else 0.8
					self.render_geoms[e].set_color(*entity.color, alpha=alpha)

					if not entity.silent:
						for ci in range(self.world.dim_c):
							color = 1 - entity.state.c[ci]
							self.comm_geoms[e][ci].set_color(
								color, color, color)
					self.agent_separation_xforms[e].set_translation(*entity.state.p_pos)
					if self.with_background:
						self.agent_separation_geoms[e].set_color(1, 1, 1, alpha=0.5)
					else:
						self.agent_separation_geoms[e].set_color(0, 0, 0, alpha=0.1)
					alpha = 0.4 if self.with_background else 0.1
					if not entity.done and entity.departed:
						if entity.action_safety_filtered:
							# if action filtered, set color to orange.
							self.agent_separation_geoms[e].set_color(1, 0.66, 0, alpha=alpha)
						for a in self.world.agents:
							if a.name != entity.name and (not a.done and a.departed) and self.is_collision(a, entity, separation_distance):
								# if collision, set color to red.
								self.agent_separation_geoms[e].set_color(1, 0, 0, alpha=alpha)
								break
					elif entity.done:
						# if done, set color to green.
						self.agent_separation_geoms[e].set_color(0, 1, 0, alpha=alpha)
					elif not entity.departed:
						if self.with_background:
							self.agent_separation_geoms[e].set_color(0, 0, 0, alpha=0.4)
						else:
							self.agent_separation_geoms[e].set_color(0, 0, 0, alpha=0.05)
				elif 'landmark' in entity.name:
					alpha = 1.0 if self.with_background else 0.5
					self.render_geoms[e].set_color(*entity.color, alpha=alpha)
					self.agent_separation_xforms[e].set_translation(*entity.state.p_pos)
					alpha = 0.2 if self.with_background else 0.1
					self.agent_separation_geoms[e].set_color(*entity.color, alpha=alpha)
				else:
					self.render_geoms[e].set_color(*entity.color, alpha=0.3)
					if entity.channel is not None:
						for ci in range(self.world.dim_c):
							color = 1 - entity.channel[ci]
							self.comm_geoms[e][ci].set_color(
								color, color, color)

			# render the graph connections
			if hasattr(self.world, 'graph_mode'):
				if self.world.graph_mode:
					edge_list = self.world.edge_list.T
					assert edge_list is not None, ("Edge list should not be None")
					num_entity = len(self.world.entities)
					for (i1, entity1) in enumerate(self.world.entities):
						for i2 in range(i1+1, num_entity):
							entity2 = self.world.entities[i2]
							e1_id, e2_id = entity1.global_id, entity2.global_id
							# if edge exists draw a line
							if [e1_id, e2_id] in edge_list.tolist():
								src = entity1.state.p_pos
								dest = entity2.state.p_pos
								### commenting out edge line drawings
								color = (0.8, 0.8, 0.8) if self.with_background else (0, 0, 0)
								attr = {'color': color, 'linewidth': 1}
								if 'agent' in entity1.name and 'agent' in entity2.name:
									self.viewers[i].draw_line(start=src, end=dest, **attr)
					for (i1, agent1) in enumerate(self.world.agents):
						for (i2, agent2) in enumerate(self.world.agents):
							src = agent1.state.p_pos
							dest = agent2.state.p_pos
							distance_diff = dest - src
							distance = np.linalg.norm(distance_diff)
							if i2 == agent1.deconflicting_agent_index and agent1.safety_filtered:
								attr = {'color': (0.8235294117647058, 0.21568627450980393, 0.3686274509803922), 'linewidth': 1}
								distance_direction = np.arctan2(distance_diff[1], distance_diff[0])
								distance_direction = np.array([np.cos(distance_direction), np.sin(distance_direction)])
								self.viewers[i].draw_line(start=src, end=dest, **attr)
								attr = {'color': (1, 0, 0.23137254901960785), 'linewidth': 2}
								dest_direction_indicator = src + distance_direction * min(separation_distance, distance)
								self.viewers[i].draw_line(start=src, end=dest_direction_indicator, **attr)
							# if distance <= self.world.engagement_distance and not agent2.done and not agent1.done:
							# 	attr = {'color': (1, 0.549, 0), 'linewidth': 2}
							# 	self.viewers[i].draw_line(start=src, end=dest, **attr)
								

			# render to display or array
			results.append(self.viewers[i].render(
						return_rgb_array = mode=='rgb_array'))

		return results

	# create receptor field locations in local coordinate frame
	def _make_receptor_locations(self, agent:Agent) -> List:
		receptor_type = 'polar'
		range_min = 0.05 * 2.0
		range_max = 1.00
		dx = []
		# circular receptive field
		if receptor_type == 'polar':
			for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
				for distance in np.linspace(range_min, range_max, 3):
					dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
			# add origin
			dx.append(np.array([0.0, 0.0]))
		# grid receptive field
		if receptor_type == 'grid':
			for x in np.linspace(-range_max, +range_max, 5):
				for y in np.linspace(-range_max, +range_max, 5):
					dx.append(np.array([x,y]))
		return dx

	@staticmethod
	def is_collision(agent1:Agent, agent2:Agent, dist_min: Optional[float] = None) -> bool:
		delta_pos = agent1.state.p_pos - agent2.state.p_pos
		dist = np.linalg.norm(delta_pos)
		if dist_min is None:
			dist_min = 1.05*(agent1.size + agent2.size)
		return True if dist < dist_min else False

class MultiAgentGraphEnv(MultiAgentBaseEnv):
	metadata = {
		'render.modes' : ['human', 'rgb_array']
	}
	"""
		Parameters:
		–––––––––––
		world: World
			World for the environment. Refer `multiagent/core.py`
		reset_callback: Callable
			Reset function for the environment. Refer `reset()` in 
			`multiagent/navigation_graph.py`
		reward_callback: Callable
			Reward function for the environment. Refer `reward()` in 
			`multiagent/navigation_graph.py`
		observation_callback: Callable
			Observation function for the environment. Refer `observation()` 
			in `multiagent/navigation_graph.py`
		graph_observation_callback: Callable
			Observation function for graph_related stuff in the environment. 
			Refer `graph_observation()` in `multiagent/navigation_graph.py`
		id_callback: Callable
			A function to get the id of the agent in graph
			Refer `get_id()` in `multiagent/navigation_graph.py`
		info_callback: Callable
			Reset function for the environment. Refer `info_callback()` in 
			`multiagent/navigation_graph.py`
		done_callback: Callable
			Reset function for the environment. Refer `done()` in 
			`multiagent/navigation_graph.py`
		update_graph: Callable
			A function to update the graph structure in the environment
			Refer `update_graph()` in `multiagent/navigation_graph.py`
		shared_viewer: bool
			If we want a shared viewer for rendering the environment or 
			individual windows for each agent as the ego
		discrete_action: bool
			If the action space is discrete or not
		scenario_name: str
			Name of the scenario to be loaded. Refer `multiagent/custom_scenarios.py`
	"""
	def __init__(self, world:World, reset_callback:Callable=None, 
					reward_callback:Callable=None,
					observation_callback:Callable=None, 
					graph_observation_callback:Callable=None,
					id_callback:Callable=None,
					info_callback:Callable=None,
					done_callback:Callable=None,
					agent_reached_goal_callback:Callable=None,
					update_graph:Callable=None,
					shared_viewer:bool=True, 
					discrete_action:bool=True,
					scenario_name:str='navigation',
					dynamics_type:str='airtaxi') -> None:
		super(MultiAgentGraphEnv, self).__init__(world, reset_callback, 
											reward_callback,observation_callback, 
											info_callback,done_callback, agent_reached_goal_callback,
											shared_viewer, discrete_action,
											scenario_name, dynamics_type)
		self.update_graph = update_graph
		self.graph_observation_callback = graph_observation_callback
		self.id_callback = id_callback

		# variabls to save episode data
		self.dt = self.world.dt
		self.episode_length = self.world.world_length
		self.coordination_range = self.world.coordination_range
		# This are the values saved in info. (static per each episode)
		# travel metric
		self.prev_episode_travel_time_mean = self.world.world_length
		self.prev_episode_travel_distance_mean = 0.0
		self.prev_episode_done_percentage = 0.0 # percentage between 0 and 1
		self.prev_episode_num_reached_goal_mean = 0.0
		# safety metric
		self.prev_episode_conflict_occurance_percentage = 0.0 # percentage between 0 and 1
		self.prev_episode_min_distance_mean = 0.0
		self.prev_episode_min_distance_min = 0.0
		self.prev_episode_multiple_engagement_percentage = 0.0 # percentage between 0 and 1

		# travel metric
		self.episode_agent_travel_length_list = None
		self.episode_agent_travel_distance_list = None
		self.episode_agent_done_list = None
		# safety metric
		self.episode_agent_conflict_occurance_list = None
		self.episode_agent_in_multiple_engagement_list = None
		self.episode_agent_min_distance_list = None
		self.init_episode_agent_info()

		self.set_graph_obs_space()

	def save_summary_of_episode(self) -> None:
		# save episode data to previous episode data
		self.prev_episode_travel_time_mean = self.dt * np.mean(self.episode_agent_travel_length_list)
		self.prev_episode_travel_distance_mean = np.mean(self.episode_agent_travel_distance_list)
		self.prev_episode_done_percentage = np.mean(self.episode_agent_done_list)
		self.prev_episode_num_reached_goal_mean = np.mean(self.episode_agent_reached_goals_list)
		# safety metric
		# prevent division by zero when there is no travel length.
		self.episode_agent_travel_length_list = np.where(self.episode_agent_travel_length_list == 0, 1, self.episode_agent_travel_length_list)
		self.prev_episode_conflict_occurance_percentage = np.mean(self.episode_agent_conflict_occurance_list / self.episode_agent_travel_length_list)
		self.prev_episode_min_distance_mean = np.mean(self.episode_agent_min_distance_list)
		self.prev_episode_multiple_engagement_percentage = np.mean(self.episode_agent_in_multiple_engagement_list / self.episode_agent_travel_length_list)
		if self.prev_episode_min_distance_mean == np.inf:
			self.prev_episode_min_distance_mean = self.coordination_range
		self.prev_episode_min_distance_min = np.min(self.episode_agent_min_distance_list)
		if self.prev_episode_min_distance_min == np.inf:
			self.prev_episode_min_distance_min = self.coordination_range

	def init_episode_agent_info(self) -> None:
		if self.episode_agent_travel_length_list is not None:
			# save episode data to previous episode data
			self.save_summary_of_episode()
		# initialize episode data
		# travel metric
		self.episode_agent_travel_length_list = np.zeros(len(self.world.agents))
		self.episode_agent_travel_distance_list = np.zeros(len(self.world.agents))
		self.episode_agent_done_list = np.zeros(len(self.world.agents))
		self.episode_agent_reached_goals_list = np.zeros(len(self.world.agents))
		# safety metric
		self.episode_agent_conflict_occurance_list = np.zeros(len(self.world.agents))
		self.episode_agent_min_distance_list = np.inf * np.ones(len(self.world.agents))
		self.episode_agent_in_multiple_engagement_list = np.zeros(len(self.world.agents))

	def set_graph_obs_space(self):
		self.node_observation_space = []
		self.adj_observation_space = []
		self.edge_observation_space = []
		self.agent_id_observation_space = []
		self.share_agent_id_observation_space = []
		num_agents = len(self.agents)
		for agent in self.agents:
			node_obs, adj = self.graph_observation_callback(agent, self.world)
			node_obs_dim = node_obs.shape
			adj_dim = adj.shape
			edge_dim = 1      # NOTE hardcoding edge dimension
			agent_id_dim = 1  # NOTE hardcoding agent id dimension
			self.node_observation_space.append(spaces.Box(low=-np.inf,
														high=+np.inf,
														shape=node_obs_dim,
														dtype=np.float32))
			self.adj_observation_space.append(spaces.Box(low=-np.inf,
														high=+np.inf,
														shape=adj_dim,
														dtype=np.float32))
			self.edge_observation_space.append(spaces.Box(low=-np.inf,
														high=+np.inf,
														shape=(edge_dim,),
														dtype=np.float32))
			self.agent_id_observation_space.append(spaces.Box(low=-np.inf,
														high=+np.inf,
														shape=(agent_id_dim,),
														dtype=np.float32))
			self.share_agent_id_observation_space.append(spaces.Box(low=-np.inf,
														high=+np.inf,
														shape=(num_agents*agent_id_dim,),
														dtype=np.float32))
	

	def step(self, action_n:List) -> Tuple[List, List, List, List, List, List, List]:
		if self.update_graph is not None:
			self.update_graph(self.world)
		self.current_step += 1
		obs_n, reward_n, done_n, info_n = [], [], [], []
		node_obs_n, adj_n, agent_id_n = [], [], []
		cooperate_n, defect_n = [], []
		self.world.current_time_step += 1
		self.agents = self.world.policy_agents
		# set action for each agent
		for i, agent in enumerate(self.agents):
			self._set_action(action_n[i], agent, self.action_space[i])
		# advance world state
		# print("Self.horizon",horizon)
		self.world.step()
		# record observation for each agent
		for (i, agent) in enumerate(self.agents):
			obs_n.append(self._get_obs(agent))
			agent_id_n.append(self._get_id(agent))

			### for prisoners dilemma changes
			## reward, cooperate, defect = self._get_reward(agent)
			reward = self._get_reward(agent)
			reward_n.append(reward)
			# cooperate_n.append(cooperate)
			# defect_n.append(defect)

			# print("reward",reward)
			# updated_reward = self.modify_reward(reward)
			# print(updated_reward)
			# reward_n.append(updated_reward)
			# print("updated_rew",updated_reward.shape)

			node_obs, adj = self._get_graph_obs(agent)
			# print("node_obs 1 ag", node_obs.shape)
			node_obs_n.append(node_obs)
			# print("node_obs_n ", len(node_obs_n))
			
			# node_obs_n.append(updated_reward)
			# print("node_obs_n ", len(node_obs_n))
			adj_n.append(adj)
			distance_to_other_agents = adj[i, :self.num_agents]
			distance_to_other_agents = distance_to_other_agents[distance_to_other_agents != 0]
			# print("distance_to_other_agents",distance_to_other_agents)

			if agent.departed and not agent.done:
				self.episode_agent_travel_length_list[i] += 1
				self.episode_agent_travel_distance_list[i] += np.linalg.norm(agent.state.p_vel) * self.dt
				# check if distance_to_other_agents is not empty
				if distance_to_other_agents.size > 0:
					# count how many agents are in separation distance target
					num_agent_in_engagement = np.sum(distance_to_other_agents < self.world.engagement_distance)
					if num_agent_in_engagement > 1:
						self.episode_agent_in_multiple_engagement_list[i] += 1
					if np.min(distance_to_other_agents) < self.world.separation_distance_target:
						self.episode_agent_conflict_occurance_list[i] += 1
					if np.min(distance_to_other_agents) < self.episode_agent_min_distance_list[i]:
						self.episode_agent_min_distance_list[i] = min(distance_to_other_agents)
			if agent.done:
				self.episode_agent_done_list[i] = 1

			done_n.append(self._get_done(agent))
			info = {'individual_reward': reward}

			env_info = self._get_info(agent)
			info.update(env_info)   # nothing fancy here, just appending dict to dict
			info_n.append(info)

		# all agents get total reward in cooperative case
		reward = np.sum(reward_n)
		# print("env",reward_n)
		if self.shared_reward:
			reward_n = [[reward]] * self.n  # NOTE this line is similar to PPOEnv
		else:
			reward_n = reward_n
		# print("shared_reward", reward_n)
		# new_node_obs = modify_reward(node_obs_n, adj,reward_n)
		# print("rewards",(reward_n), done_n)
		# print("---------~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~-----------------")
		return obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n, info_n

		# return obs_n, agent_id_n, node_obs_n, adj_n, reward_n,cooperate_n, defect_n, done_n, info_n

	def reset(self, num_current_episode: Optional[int] = 0) -> Tuple[List, List, List, List]:
		for (i, agent) in enumerate(self.agents):
			self.episode_agent_reached_goals_list[i] = self.agent_reached_goal_callback(agent)

		self.current_step = 0
		# reset world
		self.reset_callback(self.world, num_current_episode)
		# reset renderer
		self._reset_render()
		# record observations for each agent
		obs_n, node_obs_n, adj_n, agent_id_n = [], [], [], []
		self.agents = self.world.policy_agents
		for agent in self.agents:
			obs_n.append(self._get_obs(agent))
			agent_id_n.append(self._get_id(agent))
			node_obs, adj = self._get_graph_obs(agent)
			node_obs_n.append(node_obs)
			adj_n.append(adj)
		self.init_episode_agent_info()
		env_info = {}
		env_info['travel_time_mean'] = self.prev_episode_travel_time_mean
		env_info['travel_distance_mean'] = self.prev_episode_travel_distance_mean
		env_info['done_percentage'] = self.prev_episode_done_percentage
		env_info['num_reached_goal_mean'] = self.prev_episode_num_reached_goal_mean
		env_info['conflict_percentage'] = self.prev_episode_conflict_occurance_percentage
		env_info['min_distance_mean'] = self.prev_episode_min_distance_mean
		env_info['min_distance_min'] = self.prev_episode_min_distance_min
		env_info['multiple_engagement_percentage'] = self.prev_episode_multiple_engagement_percentage
		return obs_n, agent_id_n, node_obs_n, adj_n, env_info
	
	def _get_graph_obs(self, agent:Agent):
		if self.graph_observation_callback is None:
			return None, None, None
		return self.graph_observation_callback(agent, self.world)
	
	def _get_id(self, agent:Agent):
		if self.id_callback is None:
			return None
		return self.id_callback(agent)

# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
	metadata = {
		'runtime.vectorized': True,
		'render.modes' : ['human', 'rgb_array']
	}

	def __init__(self, env_batch):
		self.env_batch = env_batch

	@property
	def n(self):
		return np.sum([env.n for env in self.env_batch])

	@property
	def action_space(self):
		return self.env_batch[0].action_space

	@property
	def observation_space(self):
		return self.env_batch[0].observation_space

	def step(self, action_n, time):
		obs_n = []
		shared_obs_n = []
		reward_n = []
		done_n = []
		info_n = {'n': []}
		i = 0
		for env in self.env_batch:
			obs, shared_obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
			i += env.n
			obs_n += obs
			shared_obs_n += shared_obs
			# reward = [r / len(self.env_batch) for r in reward]
			reward_n += reward
			done_n += done
		return obs_n, shared_obs_n, reward_n, done_n, info_n

	def reset(self):
		obs_n = []
		shared_obs_n = []
		for env in self.env_batch:
			obs, shared_obs = env.reset()
			obs_n += obs
			shared_obs_n += shared_obs
		return obs_n, shared_obs

	# render environment
	def render(self, mode='human', close=True):
		results_n = []
		for env in self.env_batch:
			results_n += env.render(mode, close)
		return results_n
