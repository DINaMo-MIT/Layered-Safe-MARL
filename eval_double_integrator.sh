#!/bin/bash
experiment_name_str="double_integrator_safety_informed"

# "double_integrator" or "airtaxi"
dynamics_type="double_integrator"
# for custom scenario, use "navigation_graph_safe_eval" and check the last line of multiagent/config.py to see what custom scenario is used
scenario_name="navigation_graph_safe"
use_safety_filter="True"
world_size=4
# Read last line of config.py and echo
config_file="multiagent/config.py"
original_string=$(tail -n 1 $config_file)
model_dir_str="trained_models/${experiment_name_str}"

num_eval_episodes=3
num_eval_agents=4
eval_episode_length=250
echo "Running ${num_eval_episodes} episodes with ${num_eval_agents} agents and episode length ${eval_episode_length}."

python scripts/eval_mpe.py \
--model_dir=${model_dir_str} \
--dynamics_type ${dynamics_type} --render_episodes=${num_eval_episodes} \
--world_size=${world_size} --num_landmarks=2 \
--num_agents=${num_eval_agents} \
--num_obstacles=0 \
--seed=0 \
--episode_length=${eval_episode_length} \
--use_dones=False --collaborative=False \
--scenario_name=${scenario_name} --horizon=1 --save_gifs --use_render --num_walls=0 \
--discrete_action=True \
--use_masking "True" \
--use_safety_filter ${use_safety_filter}