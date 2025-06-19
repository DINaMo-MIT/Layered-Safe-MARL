#!/bin/bash
logs_folder="logs"
mkdir -p $logs_folder
# Run the script

seed_max=2

# Change this to your name.
user_name="jason"

## MAJOR ARGUMENTS TO CHECK!
# GPU number (check unused GPU with nvidia-smi)
cuda_device=1
# "double_integrator" or "airtaxi"
dynamics_type="airtaxi"
use_safety_filter="True"
scenario_name="navigation_graph_safe"
n_agents=4
seed=0

if [ "$dynamics_type" == "double_integrator" ]; then
    episode_length=250
    world_size=4
    n_landmarks=2
    num_env_steps=5000000
elif [ "$dynamics_type" == "airtaxi" ]; then
    episode_length=350
    world_size=6
    n_landmarks=2
    num_env_steps=5000000
else
    echo "Error: Unsupported dynamics type '$dynamics_type'"
    exit 1  # Exit with a non-zero status to indicate an error
fi
num_internal_step=1


datetime_str=$(date '+%y%m%d_%H%M%S')
# for seed in `seq ${seed_max}`;
# do
# # seed=`expr ${seed} + 3`
# echo "seed: ${seed}"
# execute the script with different params
if [ "$dynamics_type" == "kinematic_vehicle" ]; then
    str_dynamics_type="kv"
elif [ "$dynamics_type" == "double_integrator" ]; then
    str_dynamics_type="di"
elif [ "$dynamics_type" == "airtaxi" ]; then
    str_dynamics_type="airtaxi"
else
    echo "Error: Unsupported dynamics type '$dynamics_type'"
    exit 1  # Exit with a non-zero status to indicate an error
fi

str_scenario="random"
if [ "$scenario_name" == "navigation_graph_safe_eval" ]; then
    str_scenario="eval"
    # num_env_steps=3000000
    num_env_steps=6000000
fi

if [ "$use_safety_filter" == "True" ]; then
    str_safety_filter=1
    num_rollout_threads=32
else
    str_safety_filter=0
    num_rollout_threads=32
fi

echo "datetime_str: ${datetime_str}"
echo "dynamics_type: ${dynamics_type}"
echo "use_safety_filter: ${use_safety_filter}"
echo "scenario_name: ${scenario_name}"
echo "gpu: ${cuda_device}"
echo "num_threads: ${num_rollout_threads}"
echo "num_env_steps: ${num_env_steps}"
echo "seed: ${seed}"
log_file_str=$logs_folder/${datetime_str}_${str_dynamics_type}_env_${str_scenario}_safety_${str_safety_filter}_agent${n_agents}_landmark${n_landmarks}_eplength${episode_length}_world${world_size}_${seed}
experiment_name_str=${user_name}_${datetime_str}_agent${n_agents}_landmark${n_landmarks}_eplength${episode_length}_world${world_size}

# For warmstart, include this in the command line:
# --model_dir="trained_models/250118_233010_di_safety_blind_baseline" \

echo "============================================================"
echo "Running training..."
CUDA_VISIBLE_DEVICES=${cuda_device} \
python -u scripts/train_mpe.py --use_valuenorm --use_popart \
--project_name "${str_dynamics_type}_env_${str_scenario}_safety_${str_safety_filter}" \
--env_name "GraphMPE" \
--algorithm_name "rmappo" \
--seed ${seed} \
--experiment_name ${experiment_name_str} \
--scenario_name ${scenario_name} \
--dynamics_type ${dynamics_type} \
--num_agents=${n_agents} \
--num_landmarks=${n_landmarks} \
--n_training_threads 1 --n_rollout_threads ${num_rollout_threads} \
--num_mini_batch 1 \
--episode_length ${episode_length} \
--num_env_steps ${num_env_steps} \
--ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
--user_name "ucb_mit_marl" \
--use_cent_obs "False" \
--graph_feat_type "relative" \
--use_dones "False" \
--collaborative "False" \
--num_walls 0 \
--world_size=${world_size} \
--auto_mini_batch_size --target_mini_batch_size 4096 \
--use_safety_filter ${use_safety_filter} \
--soft_filter_type ${soft_filter_type} \
--num_internal_step ${num_internal_step} \
--use_masking "True" \
--checkpoint_interval 50 \
--use_wandb
# &> ${log_file_str}