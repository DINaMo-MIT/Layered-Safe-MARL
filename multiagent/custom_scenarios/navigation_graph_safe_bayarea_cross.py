"""
        Navigation for `n` agents to `n` goals from random initial positions
        With random obstacles added in the environment
        Each agent is destined to get to its own goal unlike
        `simple_spread.py` where any agent can get to any goal (check `reward()`)
"""
from multiagent.core import EntityDynamicsType, World, Agent, Landmark, Entity, Wall
from multiagent.custom_scenarios.utils import map_each_agent_landmarks_to_entire_landmarks, creat_relative_heading_list_from_goal_position_list, randomly_generate_separated_positions, generate_goal_points_along_line
from multiagent.custom_scenarios.navigation_graph_safe import SafeAamScenario, RealisticScenario
from multiagent.config import eval_scenario_type
from typing import Optional, Tuple, List
import argparse
import numpy as np
from numpy import ndarray as arr
from scipy import sparse
import scipy.spatial.distance as dist

import os
import sys
sys.path.append(os.path.abspath(os.getcwd()))


entity_mapping = {'agent': 0, 'landmark': 1, 'obstacle': 2, 'wall': 3}
# NORTH # 183 255
offset_x = 0
offset_y = 0

# SF TO FREEMONT	
CORRIDOR1_WAYPOINTS_PIXEL = [(611, 558),
                             (1016, 1015),
                             (1421, 1472),
                             (1794, 1678),
                             (2114, 1840),
                             (2550, 2048),	
                             (3106, 2340)] 
CORRIDOR1_WAYPOINTS_PIXEL.reverse()
# OAK TO RWC	
CORRIDOR2_WAYPOINTS_PIXEL = [(1569, 908),
                                (1556, 1320),
                                (1536, 1692),
                                (1536, 2048),
                                (1535, 2420),
                                (1535, 2764)]
class Scenario(RealisticScenario):
    def __init__(self):
        super().__init__(scenario_image_file_name='bayarea_cross.jpg', km_in_pixel=73.6)
        self.eval_scenario_type = "fixed_schedule"
        assert self.eval_scenario_type in ["fixed_schedule", "single"], "Invalid scenario type"

    def reset_world(self, world, num_current_episode = 0):
        super().reset_world(world, num_current_episode)
        for i, landmark in enumerate(world.landmarks):
            landmark_agent = i % self.num_agents
            if landmark_agent % 2 == 0:
                landmark.color = np.array([0.3058823529411765, 0.32941176470588235, 0.9372549019607843]) # custom blue
            else:
                landmark.color = np.array([0.8901960784313725, 0.8470588235294118, 0.49411764705882355]) # custom yellow

    def random_scenario(self, world) -> None:
        if self.eval_scenario_type == "fixed_schedule":
            self.scenario_fixed_schedule(world)
        elif self.eval_scenario_type == "single":
            self.scenario_single_agent(world)
        else:
            raise NotImplementedError

    def get_default_landmark_num_for_scenario(self) -> int:
        return 6

    def get_aspect_ratio_for_scenario(self) -> float:
        ar_image = self.world_width_pixel / self.world_height_pixel
        return ar_image

    def scenario_fixed_schedule(self, world):

        assert self.num_agents % 2 == 0, "Number of agents should be even"
        num_departure = int(self.num_agents / 2)
        depart_positions = [self.convert_pixel_to_world_coordinates(CORRIDOR1_WAYPOINTS_PIXEL[0]), 
                            self.convert_pixel_to_world_coordinates(CORRIDOR2_WAYPOINTS_PIXEL[0])]
        corridor1_waypoints = [self.convert_pixel_to_world_coordinates(waypoint) for waypoint in CORRIDOR1_WAYPOINTS_PIXEL]
        corridor1_waypoints = corridor1_waypoints[1:]
        corridor1_last_heading = np.arctan2(corridor1_waypoints[-1][1] - corridor1_waypoints[-2][1], corridor1_waypoints[-1][0] - corridor1_waypoints[-2][0])
        corridor2_waypoints = [self.convert_pixel_to_world_coordinates(waypoint) for waypoint in CORRIDOR2_WAYPOINTS_PIXEL]
        corridor2_waypoints = corridor2_waypoints[1:]
        corridor2_last_heading = np.arctan2(corridor2_waypoints[-1][1] - corridor2_waypoints[-2][1], corridor2_waypoints[-1][0] - corridor2_waypoints[-2][0])
        # to match the number of landmark..
        corridor2_waypoints.append(corridor2_waypoints[-1])
        corridor1_depart_heading = np.arctan2(corridor1_waypoints[0][1] - depart_positions[0][1], corridor1_waypoints[0][0] - depart_positions[0][0])
        corridor2_depart_heading = np.arctan2(corridor2_waypoints[0][1] - depart_positions[1][1], corridor2_waypoints[0][0] - depart_positions[1][0])
        
        departure_time_interval = 90
        depart_time_offset = 250
        list_of_agent_landmarks = []
        list_of_agent_landmark_headings = []
        list_of_agent_landmark_speeds = []
        num_landmarks_per_agent = self.get_default_landmark_num_for_scenario()
        for i in range(self.num_agents):
            agent = world.agents[i]
            agent.state.p_pos = depart_positions[i % 2]
            agent.departed = False
            agent.state.init_theta = corridor1_depart_heading if i % 2 == 0 else corridor2_depart_heading
            agent.done = False
            random_size = 15
            agent.departure_timer = (i // 2) * departure_time_interval + np.random.randint(-random_size, random_size)
            if i % 2 == 1:
                agent.departure_timer += depart_time_offset
            landmark_positions = corridor1_waypoints if i % 2 == 0 else corridor2_waypoints
            goal_headings = creat_relative_heading_list_from_goal_position_list(landmark_positions)
            last_heading = corridor1_last_heading if i % 2 == 0 else corridor2_last_heading
            if i % 2 == 1:
                goal_headings[-1] = corridor2_last_heading
            goal_headings.append(last_heading)
            print(goal_headings)
            goal_speeds = [self.goal_speed_max] * 7
            list_of_agent_landmarks.append(landmark_positions)
            list_of_agent_landmark_headings.append(goal_headings)
            list_of_agent_landmark_speeds.append(goal_speeds)
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