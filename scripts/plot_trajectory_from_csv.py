import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import argparse
from matplotlib.patches import Circle
from hj_reachability_torch.grid import get_hj_numpy_grid_from_meta_data, HjNumpyGrid
from hj_reachability_torch.vis import vis_2d_level_set
from hj_reachability_torch.utils import to_numpy
# DISTANCE_TO_GOAL_THRESHOLD = 750 * 0.0003048
DISTANCE_TO_GOAL_THRESHOLD = 0.3
V_MAX = 175 * 0.514444  * 0.001 # knots to km/s

def plot_landmark(ax, x, y, heading, color, radius=DISTANCE_TO_GOAL_THRESHOLD, linewidth=2):
    circle = Circle((x, y), radius, edgecolor=color, facecolor='none', linewidth=linewidth, alpha=0.5)
    ax.add_patch(circle)
    arrow_size = radius * 0.7
    dx = arrow_size * np.cos(heading)
    dy = arrow_size * np.sin(heading)
    ax.arrow(
        x, y, dx, dy,
        head_width=radius * 0.15, head_length=radius * 0.1,
        fc=color, ec=color, linewidth=linewidth
    )

def vis_vehicle(ax, x_position, y_position, heading, length=0.2, width=0.1, vehicle_color='k', arrow_color='w'):
    rect = plt.Rectangle((-length / 2, -width / 2), length, width, color=vehicle_color, alpha=0.5)
    transform = (
        plt.matplotlib.transforms.Affine2D()
        .rotate(heading)
        .translate(x_position, y_position)
        + ax.transData
    )
    rect.set_transform(transform)
    ax.add_patch(rect)
    arrow_length = 0.75 * length
    dx = arrow_length * np.cos(heading)
    dy = arrow_length * np.sin(heading)
    ax.arrow(
        x_position, y_position, dx, dy,
        head_width=0.04, head_length=0.06, fc=arrow_color, ec=arrow_color, width=0.01
    )

def plot_combined_visualization(model, scenario_name, position_grid, values_other_vehicle1, values_other_vehicle2):
    x_shift = 1.0
    init_agent_positions = [
        [-0.5, 0.0],
        [2.0, 1.0],
        [2.2, -1.2]
    ]
    headings = [0.0, 4 * np.pi / 3, 2 * np.pi / 3]

    position_file = model + '/' + f'log_position_{scenario_name}.csv'
    safety_file = model + '/' + f'log_safety_{scenario_name}.csv'

    position_data = pd.read_csv(position_file)
    safety_data = pd.read_csv(safety_file)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_facecolor([0.1, 0.1, 0.1])

    base_colors = [(0.3, 0.3, 0.9), (0.3, 0.8, 0.8), (0.9, 0.8, 0.4)]
    safety_colors = {0: None, 1: (1.0, 0.5, 0.2), 2: (1.0, 0.0, 0.2)}

    # Determine available robots dynamically
    available_robots = [col.split('_')[1] for col in position_data.columns if col.startswith('x_')]

    for i, robot in enumerate(available_robots):
        x = position_data[f'x_{robot}'] + x_shift
        y = position_data[f'y_{robot}']
        safety_levels = safety_data[f'{robot}']
        alphas = np.linspace(0.2, 1.0, len(position_data['step']))
        for j in range(len(position_data['step']) - 1):
            safety_level = safety_levels.iloc[j]
            color = safety_colors[safety_level] if safety_level in safety_colors and safety_colors[safety_level] else base_colors[int(robot)]
            ax.plot(x[j:j+2], y[j:j+2], color=color, alpha=alphas[j], linewidth=5)

        # Plot landmarks corresponding to available robots
        landmark = np.array(init_agent_positions[int(robot)]) + np.array([5.0 * np.cos(headings[int(robot)]), 5.0 * np.sin(headings[int(robot)])])
        plot_landmark(ax, *landmark, headings[int(robot)], base_colors[int(robot)])

    vis_2d_level_set(fig, ax, position_grid, values_other_vehicle1, colormap=False, contour_color=base_colors[1])
    vis_2d_level_set(fig, ax, position_grid, values_other_vehicle2, colormap=False, contour_color=base_colors[2])

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(True, color='white')
    ax.set_ylim(-4.0, 4.0)
    ax.set_aspect('equal', adjustable='datalim')

    plt.savefig(model + '/' + f'plot_combined_{scenario_name}.png', dpi=300)
    plt.savefig(model + '/' + f'plot_combined_{scenario_name}.pdf')
    plt.show()

def get_relative_state_from_two_vehicle_state(state_ref, state_other):
    """ states: x, y, theta, v
    """
    x_rel = state_other[0, ...] - state_ref[0, ...]
    y_rel = state_other[1, ...] - state_ref[1, ...]
    x_rel_rotated = x_rel * np.cos(state_ref[2, ...]) + y_rel * np.sin(state_ref[2, ...])
    y_rel_rotated = -x_rel * np.sin(state_ref[2, ...]) + y_rel * np.cos(state_ref[2, ...])
    theta_rel = state_other[2, ...] - state_ref[2, ...]    
    v_ref = state_ref[3, ...]
    v_other = state_other[3, ...]
    return np.stack([x_rel_rotated, y_rel_rotated, theta_rel, v_ref, v_other], axis=0)

def get_value_2d_with_respect_to_other_vehicle_state(grid, values, position_grid, ego_vehicle_heading, ego_vehicle_speed, other_vehicle_state):
    values_other_vehicle = np.zeros(position_grid.shape)
    for (i_x, x) in enumerate(position_grid.coordinate_vectors[0]):
        for (i_y, y) in enumerate(position_grid.coordinate_vectors[1]):
            ego_state = np.array([x, y, ego_vehicle_heading, ego_vehicle_speed])
            relative_state = get_relative_state_from_two_vehicle_state(other_vehicle_state, ego_state)
            values_other_vehicle[i_x, i_y] = grid.eval_value_and_deriv_from_table(relative_state, values)
    return values_other_vehicle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined visualization of trajectories and safe sets.")
    parser = argparse.ArgumentParser(description="Plot robot trajectories with safety levels.")
    parser.add_argument(
        '--scenario_name', 
        type=str, 
        nargs='?',  # Makes this argument optional
        default='two_vehicle_conflicting_example_num_agent2_landmark1_safety_True_world_size5.0_episode_0',  # Default value
        help="Tail string for the CSV filenames."
    )
    parser.add_argument(
        '--model',
        type=str,
        help="Model to run experiment."
    )
    args = parser.parse_args()

    hj_file_name = 'data/airtaxi_value_function.pkl'

    with open(hj_file_name, 'rb') as f:
        hj_data_loaded = pickle.load(f)

    grid_meta_data = hj_data_loaded.grid_meta_data
    domain_lo = grid_meta_data.domain_lo
    domain_hi = grid_meta_data.domain_hi
    position_grid_shape = grid_meta_data.shape
    grid_dim = len(position_grid_shape)
    grid = get_hj_numpy_grid_from_meta_data(grid_meta_data)
    values = to_numpy(hj_data_loaded.values)
    del hj_data_loaded
    del grid_meta_data
    
    # Example grid and value function data for demonstration
    position_domain_lo = np.array([-1, -2.5])
    position_domain_hi = np.array([3, 2.5])
    position_grid_shape = [80, 100]
    position_grid = HjNumpyGrid(domain_lo=position_domain_lo, domain_hi=position_domain_hi, shape=position_grid_shape)

    other_vehicle_state1 = np.array([2.0, 1.0, 4 * np.pi / 3, V_MAX - 0.005])
    other_vehicle_state2 = np.array([2.2, -1.2, 2 * np.pi / 3, V_MAX - 0.005])
    ego_vehicle_heading = 0.0
    ego_vehicle_speed = V_MAX - 0.005
    values_other_vehicle1 = get_value_2d_with_respect_to_other_vehicle_state(grid, values, position_grid, ego_vehicle_heading, ego_vehicle_speed, other_vehicle_state1)
    values_other_vehicle2 = get_value_2d_with_respect_to_other_vehicle_state(grid, values, position_grid, ego_vehicle_heading, ego_vehicle_speed, other_vehicle_state2)

    plot_combined_visualization(args.model, args.scenario_name, position_grid, values_other_vehicle1, values_other_vehicle2)
