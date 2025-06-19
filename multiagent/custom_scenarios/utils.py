from typing import Optional, Tuple, List
import math
import numpy as np
from numpy import ndarray as arr

from multiagent.core import EntityDynamicsType, World, Agent, Landmark, Entity, Wall, BaseEntityState

entity_mapping = {'agent': 0, 'landmark': 1, 'obstacle':2, 'wall':3}

def map_each_agent_landmarks_to_entire_landmarks(each_agent_landmarks_list:List[List[Landmark]]) -> List[Landmark]:
	"""
		Args:
			each_agent_landmarks_list: List of List of landmarks where each list of landmarks corresponds to a single agent
		Returns:
			landmarks_list: List of all the landmarks concatenated in the correct order.
			- Order example:
			- agent1-landmark1, agent2-landmark1, agent3-landmark1, agent1-landmark2, agent2-landmark2, agent3-landmark2, ...
	"""
	# make sure that the number of landmarks for each agent is the same
	assert len(set([len(agent_landmarks) for agent_landmarks in each_agent_landmarks_list])) == 1, "Number of landmarks for each agent should be the same"
	landmarks_list = []
	for i in range(len(each_agent_landmarks_list[0])):
		for j in range(len(each_agent_landmarks_list)):
			landmarks_list.append(each_agent_landmarks_list[j][i])
	return landmarks_list

def creat_relative_heading_list_from_goal_position_list(goal_position: List[arr]) -> List[arr]:
	""" Returns the list of relative heading to the next goal position.
		Goal_position_list is specified for each agent.
	"""
	assert len(goal_position) > 1, "Goal position list should have more than 1 element"
	heading_list = []
	for i in range(len(goal_position)-1):
		heading = goal_position[i+1] - goal_position[i]
		heading = np.arctan2(heading[1], heading[0])
		heading_list.append(heading)
	return heading_list

def randomly_generate_separated_positions(
	num_positions:int, x_range:Tuple[float,float], y_range:Tuple[float,float],
	min_distance:float=0.0, max_distance:float=np.inf) -> List[arr]:
	"""
		Args:
			num_agents: Number of agents
			x_range: Tuple of x range
			y_range: Tuple of y range
			separation_distance: Minimum separation distance between positions
		Returns:
			List of positions
	"""
	positions = []
	num_iteration_max = 1000
	for i in range(num_positions):
		# compare distance to other agents
		if i > 0:
			for j in range(num_iteration_max):
				x = np.random.uniform(x_range[0], x_range[1])
				y = np.random.uniform(y_range[0], y_range[1])
				dist_to_existing_positions = np.min(np.linalg.norm(np.array(positions) - np.array([x,y]), axis=1))
				if dist_to_existing_positions > min_distance and dist_to_existing_positions < max_distance:
					break
				if j == num_iteration_max - 1:
					print("Warning: Could not find a position that satisfies the separation distance")
		else:
			x = np.random.uniform(x_range[0], x_range[1])
			y = np.random.uniform(y_range[0], y_range[1])
		positions.append(np.asarray([x,y]))
	return positions

def generate_goal_points_along_line(start_position:arr, end_position:arr, num_points:int) -> List[arr]:
	""" Generate goal points along the line between start_position and end_position.
		(exclude start position and include end position)
	"""
	goal_positions = []
	for i in range(num_points):
		goal_positions.append(start_position + (end_position - start_position) * (i+1) / num_points)
	return goal_positions

def direction_alignment_error(heading_current, heading_ref):
	# 0 if aligned, 1 if opposite.
	return 0.5 - 0.5 * math.cos(heading_current - heading_ref)

def cross_track_error(position_current, heading_current, position_ref):
	# positive is current position is at left of the reference waypoint.
	position_diff = position_ref - position_current
	cross_track_error = position_diff[0] * np.sin(heading_current) - position_diff[1] * np.cos(heading_current)
	# normalization to (0, 1)
	cross_track_error = np.abs(cross_track_error) / np.maximum(np.linalg.norm(position_diff), 1e-6)
	return np.clip(cross_track_error, 0, 1)

def is_in_front_of_ref_state(x: float, y: float, x_ref: float, y_ref: float, theta_ref: float):
    """
        Check whether state is in front of ref_state, based on ref_state's heading.
    """
    return (np.cos(theta_ref) * (x - x_ref) + np.sin(theta_ref) * (y - y_ref) > 0)

def get_adjacent_dubins_circles(x: float, y: float, theta: float, turning_radius: float):
	assert turning_radius > 0, "turning_radius must be positive."
	# returns two circles (center) that are tangent to the current state and have the turning radius.
	circle_left = [x - turning_radius * np.sin(theta), y + turning_radius * np.cos(theta)]
	circle_right = [x + turning_radius * np.sin(theta), y - turning_radius * np.cos(theta)]
	return circle_left, circle_right

def get_relative_position_from_reference(query_position: np.ndarray,
	reference_position: np.ndarray, reference_heading: float):
	# returns relative position from the reference state.
	assert query_position.shape == (2,), "query_position should be a 2D array."
	assert reference_position.shape == (2,), "reference_position should be a 2D array."
	relative_position = query_position - reference_position
	rot_matrix = np.array([[np.cos(reference_heading), np.sin(reference_heading)], [-np.sin(reference_heading), np.cos(reference_heading)]])
	relative_position_rotated = np.dot(rot_matrix, relative_position)
	return relative_position_rotated

def get_agent_observation_relative_with_heading(agent_position: np.ndarray, agent_heading: float, agent_speed: float,
												goal_position: np.ndarray, goal_heading: float, goal_speed: float):
	# Returns observations relatively defined with respect to the agent's state.
	# Used for kinematic vehicle type where heading is important (non-holonomic).
	assert goal_heading is not None, "goal_heading should not be None."
	assert goal_speed is not None, "goal_speed should not be None."

	relative_goal_position = get_relative_position_from_reference(goal_position, agent_position, agent_heading)
	relative_goal_heading = goal_heading - agent_heading
	relative_goal_heading_sincos = np.array([np.sin(relative_goal_heading), np.cos(relative_goal_heading)])
	obs = np.concatenate([np.array([agent_speed]), relative_goal_position, relative_goal_heading_sincos, np.array([goal_speed])])
	return obs

def get_agent_observation_relative_without_heading(agent_position: np.ndarray, agent_velocity: np.ndarray,
												goal_position: np.ndarray, goal_heading: float, goal_speed: float):
	# Returns observations relatively defined with respect to the agent's state.
	# Used for double integrator type where heading is not important (holonomic).
	assert goal_heading is not None, "goal_heading should not be None."
	assert goal_speed is not None, "goal_speed should not be None."

	relative_goal_position = goal_position - agent_position
	goal_heading_sincos = np.array([np.sin(goal_heading), np.cos(goal_heading)])
	obs = np.concatenate([agent_velocity, relative_goal_position, goal_heading_sincos, np.array([goal_speed])])
	return obs

def get_agent_node_observation_relative_with_heading(agent_state: BaseEntityState, 
													agent_goal_position: np.ndarray, agent_goal_heading: float, agent_goal_speed: float,
													reference_agent_state: BaseEntityState):
	
	# Returns node_observation of agent relatively defined with respect to the reference agent's state.
	# Used for kinematic vehicle type where heading is important (non-holonomic).
	assert agent_goal_heading is not None, "agent_goal_heading should not be None."
	assert agent_goal_speed is not None, "agent_goal_speed should not be None."
	reference_position = reference_agent_state.p_pos
	reference_velocity = reference_agent_state.p_vel
	reference_heading = reference_agent_state.theta

	agent_position = agent_state.p_pos
	agent_velocity = agent_state.p_vel
	agent_heading = agent_state.theta
 
	relative_agent_position = get_relative_position_from_reference(agent_position, reference_position, reference_heading)
	relative_agent_heading = agent_heading - reference_heading
	relative_agent_heading_sincos = np.array([np.sin(relative_agent_heading), np.cos(relative_agent_heading)])
	relative_speed = np.linalg.norm(agent_velocity - reference_velocity)
 
	relative_agent_goal_position = get_relative_position_from_reference(agent_goal_position, reference_position, reference_heading)
	relative_agent_goal_heading = agent_goal_heading - reference_heading
	relative_agent_goal_heading_sincos = np.array([np.sin(relative_agent_goal_heading), np.cos(relative_agent_goal_heading)])

	entity_type = entity_mapping['agent']
	node_obs = np.concatenate([relative_agent_position, 
								np.array([relative_speed]),
								relative_agent_heading_sincos,
								relative_agent_goal_position,
								relative_agent_goal_heading_sincos,
								np.array([agent_goal_speed]),
								np.array([entity_type])])
	return node_obs

def get_landmark_node_observation_relative_with_heading(landmark_position: np.ndarray, landmark_heading: float, landmark_speed: float,
														reference_agent_state: BaseEntityState):
	# Returns node_observation of landmark relatively defined with respect to the reference agent's state.
	# Used for kinematic vehicle type where heading is important (non-holonomic).
	assert landmark_heading is not None, "landmark_heading should not be None."
	assert landmark_speed is not None, "landmark_speed should not be None."
	reference_position = reference_agent_state.p_pos
	reference_heading = reference_agent_state.theta
	
	relative_landmark_position = get_relative_position_from_reference(landmark_position, reference_position, reference_heading)
	relative_landmark_heading = landmark_heading - reference_heading
	relative_landmark_heading_sincos = np.array([np.sin(relative_landmark_heading), np.cos(relative_landmark_heading)])
 
	dummy_speed = reference_agent_state.speed
	dummy_position_info = relative_landmark_position
	dummy_heading_info = relative_landmark_heading_sincos
	entity_type = entity_mapping['landmark']
	
	node_obs = np.concatenate([relative_landmark_position,
								np.array([dummy_speed]),
								relative_landmark_heading_sincos,
								dummy_position_info,
								dummy_heading_info,
								np.array([landmark_speed]),
								np.array([entity_type])])
	return node_obs

def get_agent_node_observation_relative_without_heading(agent_state: BaseEntityState, 
														agent_goal_position: np.ndarray, agent_goal_heading: float, agent_goal_speed: float,
														reference_agent_state: BaseEntityState):
	# Returns node_observation of agent relatively defined with respect to the reference agent's state.
	# Used for double integrator type where heading is not important (holonomic).
	assert agent_goal_heading is not None, "agent_goal_heading should not be None."
	assert agent_goal_speed is not None, "agent_goal_speed should not be None."

	reference_position = reference_agent_state.p_pos
	reference_velocity = reference_agent_state.p_vel

	agent_position = agent_state.p_pos
	agent_velocity = agent_state.p_vel
 
	relative_agent_position = agent_position - reference_position
	relative_agent_velocity = agent_velocity - reference_velocity
 
	relative_agent_goal_position = agent_goal_position - reference_position
	agent_goal_heading_sincos = np.array([np.sin(agent_goal_heading), np.cos(agent_goal_heading)])

	entity_type = entity_mapping['agent']

	node_obs = np.concatenate([relative_agent_position, 
								relative_agent_velocity,
								relative_agent_goal_position,
								agent_goal_heading_sincos,
								np.array([agent_goal_speed]),
								np.array([entity_type])])
	return node_obs

def get_landmark_node_observation_relative_without_heading(landmark_position: np.ndarray, landmark_heading: float, landmark_speed: float,
														reference_agent_state: BaseEntityState):
	# Returns node_observation of landmark relatively defined with respect to the reference agent's state.
	# Used for double integrator type where heading is not important (holonomic).
	assert landmark_heading is not None, "landmark_heading should not be None."
	assert landmark_speed is not None, "landmark_speed should not be None."

	reference_position = reference_agent_state.p_pos
	reference_velocity = reference_agent_state.p_vel
	
	relative_landmark_position = landmark_position - reference_position
	relative_landmark_velocity = -reference_velocity
 
	dummy_position_info = relative_landmark_position
	landmark_heading_sincos = np.array([np.sin(landmark_heading), np.cos(landmark_heading)])

	entity_type = entity_mapping['landmark']
	
	node_obs = np.concatenate([relative_landmark_position,
								relative_landmark_velocity,
								dummy_position_info,
								landmark_heading_sincos,
								np.array([landmark_speed]),
								np.array([entity_type])])
	return node_obs

def get_heading_aware_distance_penalty(relative_position) -> float:
	"relative position: agent's positive relatively defined w.r.t. goal (goal heading considered)"
	distance_to_goal = np.linalg.norm(relative_position)
	agent_position_angle = np.arctan2(relative_position[1], relative_position[0])
	if relative_position[0] >= 0:
		return distance_to_goal
	else:
		return distance_to_goal * np.abs(np.sin(agent_position_angle))

def get_speed_error_penalty(speed_error: float, speed_error_max: float,
                           distance_to_goal: float, min_dist_thresh: float=0.1) -> float:
	penalty_distance_threshold = 2 * min_dist_thresh
	if distance_to_goal > penalty_distance_threshold:
		return 0.0
	speed_error_normalized = np.clip(speed_error / speed_error_max, 0, 1)
	distance_based_factor = 1 - np.clip(distance_to_goal / penalty_distance_threshold, 0, 1)
	return speed_error_normalized * distance_based_factor


def _reference_heading_based_on_magnetic_field(position: np.ndarray, radius: float, scale_x=0.5):
    """
    position: agent's positive relatively defined w.r.t. goal (goal heading considered)"
    Compute desired heading at position (x, y),
    derived based on the magnetic field of a circular loop.
    """
    assert position.shape == (2,)
    if np.abs(position[0]) < 1e-6:
        return 0.0
    scale_x = 0.5
    position[0] = scale_x * position[0]

    # Number of integral segments
    N = 50
    # Parametric angles of the loop
    phi = np.linspace(0, 2*np.pi, N, endpoint=False)
    # Combine Lx, Ly, Lz into a single array, L
    # Loop coordinates: (0, -R cos(phi), -R sin(phi))
    L = np.column_stack([
        np.zeros_like(phi),        # Lx
        -radius * np.cos(phi),     # Ly
        -radius * np.sin(phi)      # Lz
    ])
    # Combine dLx, dLy, dLz into a single array, dL
    # (0, R sin(phi), -R cos(phi))
    dL = np.column_stack([
        np.zeros_like(phi),        # dLx
        radius * np.sin(phi),      # dLy
        -radius * np.cos(phi)      # dLz
    ])
    # Perform the integral by summation
    magnetic_field = np.zeros(2)
    for li, dli in zip(L, dL):
        # Construct the full 3D position by appending a z=0
        # then compute r = (x, y, 0) - (Lx, Ly, Lz)
        r = np.array([position[0], position[1], 0.0]) - li
        r_mag_3 = np.linalg.norm(r)**3
        # Cross product
        cross_vector = np.cross(dli, r)
        # Only the x-y components contribute to the in-plane direction
        dB = cross_vector[:2] / r_mag_3
        magnetic_field += dB

    magnetic_field[0] = magnetic_field[0] / scale_x

    return np.arctan2(magnetic_field[1], magnetic_field[0])

def double_integrator_velocity_error_from_magnetic_field_reference(agent_state: BaseEntityState,
                                                                   agent_goal: Landmark,
                                                                   min_dist_thresh: float,
                                                                   min_speed: float=0.1,
                                                                   max_speed: float=1.0,
                                                                   speed_adjustment_reference_distance: float=1.5):
	relative_position = get_relative_position_from_reference(agent_state.p_pos, agent_goal.state.p_pos, agent_goal.heading)
	dist_to_goal = np.linalg.norm(relative_position)
	relative_polar_angle = np.arctan2(relative_position[1], relative_position[0])
	relative_polar_angle_range = np.pi/6
 	
	relative_velocity = get_relative_position_from_reference(agent_state.p_vel, np.zeros(2), agent_goal.heading)
	reference_heading_magnetic_field = _reference_heading_based_on_magnetic_field(relative_position, min_dist_thresh)
	reference_speed = max(agent_goal.speed, min_speed)
	distance_ratio = np.clip(dist_to_goal / speed_adjustment_reference_distance, 0, 1)
	# reference_speed becomes max_speed if distance is larger than speed_adjustment_reference_distance.
	reference_speed = reference_speed * (1 - distance_ratio) + max_speed * distance_ratio
	reference_velocity = reference_speed * np.array([np.cos(reference_heading_magnetic_field), np.sin(reference_heading_magnetic_field)])
	error_from_ref = np.linalg.norm(relative_velocity - reference_velocity)
	
	# print(f"Velocity: relative: {relative_velocity[0]:.2f}, {relative_velocity[1]:.2f}, reference: {reference_velocity[0]:.2f}, {reference_velocity[1]:.2f}, error: {error_from_ref:.2f}")

	if np.cos(relative_polar_angle) < np.cos(relative_polar_angle_range):
		return error_from_ref

	angle_ratio = np.clip((np.cos(relative_polar_angle) - np.cos(relative_polar_angle_range)) / (1 - np.cos(relative_polar_angle_range)), 0, 1)
	return error_from_ref * (1 - angle_ratio) + dist_to_goal * angle_ratio

def heading_reward_inside_landmark_zone(agent_state: BaseEntityState, agent_goal: Landmark, min_dist_thresh: float):
	distance_to_goal = np.linalg.norm(agent_state.p_pos - agent_goal.state.p_pos)
	if distance_to_goal > min_dist_thresh:
		return 0.0
	return math.cos(agent_state.theta - agent_goal.heading)

def get_random_landmark_speeds_from_fixed_speed_candidates(num_landmarks: int, speed_candidates: List[float]) -> List[float]:
	landmark_speeds = []
	for i in range(num_landmarks):
		landmark_speeds.append(np.random.choice(speed_candidates))
	return landmark_speeds