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
offset_x = 13
offset_y = 12

SAN_RAFAEL_PIXEL = (170+offset_x, 243+offset_y)
CORTE_MADERA_PIXEL = (260+offset_x, 444+offset_y)
SAN_PABLO_PIXEL = (1466+offset_x, 160+offset_y)
RICHMOND_BART_PIXEL = (1287+offset_x, 525+offset_y)
RICHMOND_SHORE_PIXEL = (1189+offset_x, 695+offset_y)
ALBANY_PIXEL = (1562+offset_x, 937+offset_y)
UCB_PIXEL = (1916+offset_x, 1032+offset_y)
BERKELEY_MARINA_PIXEL = (1573+offset_x, 1125+offset_y)

INTERMEDIATE_POINT_FOR_SAN_PABLO = (1106, 494)

OAKLAND_PIXEL = (1959+offset_x, 1765+offset_y)
INBOUND_NORTH_WAYPOINT0 = (662+offset_x, 597+offset_y)
INBOUND_NORTH_WAYPOINT1 = (910.5+offset_x, 862+offset_y)
INBOUND_NORTH_WAYPOINT2 = (1159+offset_x, 1127+offset_y)
INBOUND_NORTH_WAYPOINT3  = (1102.5+offset_x, 1412.5+offset_y)

EMBARCADERO_PIXEL = (1046+offset_x, 1698+offset_y)

class Scenario(RealisticScenario):
    def __init__(self):
        super().__init__(scenario_image_file_name='bayarea_merge.jpg', km_in_pixel=73.6)
        self.eval_scenario_type = "city_inbound"
        # self.eval_scenario_type = eval_scenario_type
        assert self.eval_scenario_type in ["oracle_park_to_ucb", "city_inbound"], "Invalid scenario type"

    def reset_world(self, world, num_current_episode = 0):
        super().reset_world(world, num_current_episode)
        for i, landmark in enumerate(world.landmarks):
            landmark_order = i // self.num_agents
            if landmark_order < self.num_landmark_per_agent - 1:
                landmark.color = np.array([0.3058823529411765, 0.32941176470588235, 0.9372549019607843]) # custom blue
            else:
                landmark.color = np.array([0.8901960784313725, 0.8470588235294118, 0.49411764705882355]) # custom yellow

    def random_scenario(self, world) -> None:
        if self.eval_scenario_type == "oracle_park_to_ucb":
            self.scenario_oracle_park_to_ucb(world)
        elif self.eval_scenario_type == "city_inbound":
            self.scenario_city_inbound(world)
        else:
            raise NotImplementedError

    def get_default_landmark_num_for_scenario(self) -> int:
        num_default_landmark = 2
        if self.eval_scenario_type == "oracle_park_to_ucb":
            return num_default_landmark
        elif self.eval_scenario_type == "city_inbound":
            return 5
        else:
            raise NotImplementedError

    def get_aspect_ratio_for_scenario(self) -> float:
        ar_image = self.world_width_pixel / self.world_height_pixel
        return ar_image

    def scenario_city_inbound(self, world):
        depart_positions = [self.convert_pixel_to_world_coordinates(CORTE_MADERA_PIXEL),
                            self.convert_pixel_to_world_coordinates(SAN_RAFAEL_PIXEL),
                            self.convert_pixel_to_world_coordinates(SAN_PABLO_PIXEL),
                            self.convert_pixel_to_world_coordinates(RICHMOND_BART_PIXEL),
                            self.convert_pixel_to_world_coordinates(RICHMOND_SHORE_PIXEL),
                            self.convert_pixel_to_world_coordinates(ALBANY_PIXEL),
                            self.convert_pixel_to_world_coordinates(UCB_PIXEL),
                            self.convert_pixel_to_world_coordinates(BERKELEY_MARINA_PIXEL)]
        # goal_positions = [EMBARCADERO_PIXEL, PIER29_PIXEL]
        # goal_positions = [EMBARCADERO_PIXEL, EMBARCADERO_PIXEL]
        # goal_positions = [self.convert_pixel_to_world_coordinates(EMBARCADERO_PIXEL), self.convert_pixel_to_world_coordinates(OAKLAND_PIXEL)]
        goal_positions = [self.convert_pixel_to_world_coordinates(EMBARCADERO_PIXEL)]
        intermediate_positions = [self.convert_pixel_to_world_coordinates(INBOUND_NORTH_WAYPOINT0),
                                    self.convert_pixel_to_world_coordinates(INBOUND_NORTH_WAYPOINT1),
                                    self.convert_pixel_to_world_coordinates(INBOUND_NORTH_WAYPOINT2),
                                    self.convert_pixel_to_world_coordinates(INBOUND_NORTH_WAYPOINT3)]
        intermediate_point_for_san_pablo = self.convert_pixel_to_world_coordinates(INTERMEDIATE_POINT_FOR_SAN_PABLO)
        landing_approach_angles =[]
        for goal_position in goal_positions:
            angle = np.arctan2(goal_position[1] - intermediate_positions[-1][1], goal_position[0] - intermediate_positions[-1][0])
            landing_approach_angles.append(angle)
        depart_angles = []
        for (i, depart_position) in enumerate(depart_positions):
            if i < 2:
                first_waypoint_index = 0
            elif i < 6:
                first_waypoint_index = 2
            else:
                first_waypoint_index = 3
            angle = np.arctan2(intermediate_positions[first_waypoint_index][1] - depart_position[1], intermediate_positions[first_waypoint_index][0] - depart_position[0])
            depart_angles.append(angle)
        depart_angles[2] = np.arctan2(intermediate_point_for_san_pablo[1] - depart_positions[2][1], intermediate_point_for_san_pablo[0] - depart_positions[2][0])
        assert self.num_agents == len(depart_positions) * len(
            goal_positions), "Number of agents should be equal to the product of number of depart positions and goal positions"

        speed_mid1 = 0.8 * self.goal_speed_max + 0.2 * self.goal_speed_min
        speed_mid2 = 0.4 * self.goal_speed_max + 0.6 * self.goal_speed_min
        list_of_agent_landmarks = []
        list_of_agent_landmark_headings = []
        list_of_agent_landmark_speeds = []
        num_landmarks_per_agent = self.get_default_landmark_num_for_scenario()
        for (i, depart_pos) in enumerate(depart_positions):
            for (j, goal_pos) in enumerate(goal_positions):
                depart_angle = depart_angles[i]
                landing_approach_angle = landing_approach_angles[j]
                agent_ij_index = i * len(goal_positions) + j
                agent = world.agents[agent_ij_index]
                agent.state.p_pos = depart_pos
                agent.state.init_theta = depart_angle
                agent.departed = False
                agent.done = False
                agent.departure_timer = j * 150 + np.random.randint(-30, 30)
                # agent.departure_timer = j * 150 + (i % 4) * 40
                # agent.departure_timer = 0

                self.freeze_agent(agent)
                if i < 2:
                    landmark_positions = [intermediate_positions[0], intermediate_positions[1], intermediate_positions[2], intermediate_positions[3], goal_pos]
                    goal_headings = creat_relative_heading_list_from_goal_position_list(landmark_positions)
                    goal_headings.append(landing_approach_angle)
                    # landmark_positions[3][0] -= 0.5
                    # goal_headings[3] = -0.5 * np.pi
                    # goal_speeds = [self.goal_speed_max] * 4 + [speed_mid1, speed_mid2, self.goal_speed_min]
                    for i in range(len(landmark_positions) - 1):
                        print(f"distance between {i} and {i+1} is {np.linalg.norm(landmark_positions[i] - landmark_positions[i+1])}")
                elif i == 2:
                    landmark_positions = [intermediate_point_for_san_pablo,
                                          intermediate_positions[1], intermediate_positions[2], intermediate_positions[3], goal_pos]
                    goal_headings = creat_relative_heading_list_from_goal_position_list(landmark_positions)
                    goal_headings.append(landing_approach_angle)
                elif i < 6:
                    landmark_positions = [intermediate_positions[2],
                                            intermediate_positions[3],
                                            goal_pos,
                                            goal_pos,
                                            goal_pos]
                    goal_headings = creat_relative_heading_list_from_goal_position_list(landmark_positions)
                    goal_headings[-2] = landing_approach_angle
                    goal_headings[-1] = landing_approach_angle
                    goal_headings.append(landing_approach_angle)
                    # landmark_positions[0][0] += 1.0
                    # goal_headings[0] -= np.pi / 9
                    # landmark_positions[1][0] -= 0.5
                    # goal_headings[1] = -0.5 * np.pi
                    # goal_speeds = [self.goal_speed_max] * 2 + [speed_mid1, speed_mid2] + [self.goal_speed_min] * 3
                else:
                    landmark_positions = [intermediate_positions[3], goal_pos, goal_pos, goal_pos, goal_pos]
                    # landmark_positions[0][0] += 0.5
                    goal_headings = creat_relative_heading_list_from_goal_position_list(landmark_positions)
                    goal_headings[-3] = landing_approach_angle
                    goal_headings[-2] = landing_approach_angle
                    goal_headings[-1] = landing_approach_angle
                    goal_headings.append(landing_approach_angle)
                    for i in range(len(landmark_positions) - 1):
                        dist = np.linalg.norm(np.array(landmark_positions[i]) - np.array(landmark_positions[i + 1]))
                        print(f"Distance between landmark {i} and {i + 1} is {dist}")
                        
                    # goal_speeds = [self.goal_speed_max, speed_mid1, speed_mid2] + [self.goal_speed_min] * 4
                goal_speeds = [self.goal_speed_max] * 5
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

    def scenario_oracle_park_to_ucb(self, world):
        ucb_pos = self.convert_pixel_to_world_coordinates(UCB_PIXEL)
        park_pos = self.convert_pixel_to_world_coordinates(ORACLE_PARK_PIXEL)
        ucb_heading = np.pi
        park_heading = 0
        init_positions = []
        init_headings = []
        final_goal_positions = []
        final_goal_headings = []
        for i in range(self.num_agents):
            if i % 2 == 0:
                init_positions.append(ucb_pos)
                init_headings.append(ucb_heading)
                final_goal_positions.append(park_pos)
                final_goal_headings.append(park_heading + np.pi)
            else:
                init_positions.append(park_pos)
                init_headings.append(park_heading)
                final_goal_positions.append(ucb_pos)
                final_goal_headings.append(ucb_heading - np.pi)
        # set initial states for agents
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = init_positions[i]
            agent.state.reset_velocity(theta=init_headings[i])
            agent.state.c = np.zeros(world.dim_c)
            agent.done = False

        list_of_agent_landmarks = []
        list_of_agent_landmark_headings = []

        for i in range(self.num_agents):
            goal_positions = generate_goal_points_along_line(start_position=init_positions[i],
                                                             end_position=final_goal_positions[i],
                                                             num_points=self.num_landmark_per_agent)
            goal_headings = creat_relative_heading_list_from_goal_position_list(goal_positions)
            goal_headings.append(final_goal_headings[i])
            list_of_agent_landmarks.append(goal_positions)
            list_of_agent_landmark_headings.append(goal_headings)

        list_of_all_landmarks = map_each_agent_landmarks_to_entire_landmarks(list_of_agent_landmarks)
        list_of_all_landmark_headings = map_each_agent_landmarks_to_entire_landmarks(list_of_agent_landmark_headings)
        # print(list_of_agent_landmarks)
        # set goal positions
        for i, landmark in enumerate(world.landmarks):
            # print(f"landmark id {i}, p_pos {list_of_all_landmarks[i][0]}, {list_of_all_landmarks[i][1]}, heading: {list_of_all_landmark_headings[i]}, speed: {list_of_all_landmark_speeds[i]}")
            landmark.state.p_pos = list_of_all_landmarks[i]
            landmark.state.stop()
            landmark.heading = list_of_all_landmark_headings[i]