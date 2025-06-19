model_dir='trained_models/airtaxi_safety_informed'

# merge sceanrio
python scripts/eval_mpe.py \
--model_dir=${model_dir} \
--dynamics_type "airtaxi" --render_episodes=1 \
--num_agents=8 \
--num_obstacles=0 \
--seed=0 \
--episode_length=750 \
--use_dones=False --collaborative=False \
--scenario_name='navigation_graph_safe_bayarea_merge' --horizon=1 --save_gifs --use_render --num_walls=0 \
--discrete_action=True \
--use_masking "True" \
--use_safety_filter "True"

# intersection scenario
python scripts/eval_mpe.py \
--model_dir='trained_models/airtaxi_for_warmstart' \
--dynamics_type "airtaxi" --render_episodes=1 \
--num_agents=16 \
--num_obstacles=0 \
--seed=0 \
--episode_length=750 \
--use_dones=False --collaborative=False \
--scenario_name='navigation_graph_safe_bayarea_cross' --horizon=1 --save_gifs --use_render --num_walls=0 \
--discrete_action=True \
--use_masking "True" \
--use_safety_filter "True"