# Layered-Safe-MARL
Multi-agent reinforcement learning for navigation with safety

Project webpage is [here](https://dinamo-mit.github.io/Layered-Safe-MARL/).

Paper:
```
@inproceedings{choi2025resolving,
  title={Resolving Conflicting Constraints in Multi-Agent Reinforcement Learning with Layered Safety},
  author={Choi, Jason J and Aloor, Jasmine Jerry and Li, Jingqi and Mendoza, Maria G and Balakrishnan, Hamsa and Tomlin, Claire J},
  booktitle={Proceedings of Robotics: Science and Systems},
  year={2025}
}
```
Our code is built on [InforMARL source code](https://github.com/nsidn98/InforMARL).

## Installation

To get started with the Safe-MARL method, clone this repository and install the required dependencies. Ensure you have pip version pip==23.1.2. Installing torch beforehand ensures the correct installation of other components.

> NOTE: Using a conda environment is preferred. Please use the following command to create a conda environment with the correct python version.

```
conda create -n safemarl python=3.11
conda activate safemarl
```

Complete Installation instructions:

```bash
conda create -n safemarl python=3.11
conda activate safemarl
git clone https://github.com/Jaroan/Safe-MARL.git
cd Safe-MARL
pip install pip==23.1.2
pip install torch==2.2.1
pip install -r requirements.txt
git clone https://github.com/ChoiJangho/hj_reachability_utils.git
cd hj_reachability_utils
pip install -e .
## need to add this as a git submodule maybe with git submodule update --init --recursive
```
Run the double integrator model from 

## Dependencies:

- Python 3.11+
- PyTorch
- OpenAI Gym
- Multi-Agent Particle Environment (MPE)


Some inforMARL code baselines:
* Pulled the MADDPG code from [here](https://github.com/shariqiqbal2810/maddpg-pytorch) for baselines.

* Pulled the MAPPO code from [here](https://github.com/marlbenchmark/on-policy) which was used in this [paper](https://arxiv.org/abs/2103.01955). Also worth taking a look at this [branch](https://github.com/marlbenchmark/on-policy/tree/222626ebef82adbb809adbc011923cf837dd6e89) for their benchmarked code


## Troubleshooting:

1. Known issues with pytorch geometric and torch-scatter packege installation. Please refer to the requirements.txt to note the versions being used in the code.
 Correct order of installation of Pytorch Geometric packages if encountering any errors:
 ```bash
    pip install --verbose git+https://github.com/pyg-team/pyg-lib.git
    pip install --verbose torch_scatter
    pip install --verbose torch_sparse
    pip install --verbose torch_cluster
    pip install --verbose torch_spline_conv
```
2. Rendering issues with Linux users: Follow the instructions to access display for the visualization of evaluation tests.

3. `AttributeError: 'NoneType' object has no attribute 'origin'`: This error arises whilst using `torch-geometric` with CUDA. Uninstall `torch_geometric`, `torch-cluster`, `torch-scatter`, `torch-sparse`, and `torch-spline-conv`. Then re-install using:
    ```
    TORCH="1.8.0"
    CUDA="cu102"
    pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html --user
    pip install --no-index torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html --user
    pip install torch-geometric --user
    ```
## Safety Value Functions
Download the safety value functions from [this directory](https://drive.google.com/drive/folders/1ba12lWbN69u0EOm4TUfz67VfRO4vS9dw?usp=sharing), and add them under `data` in the root directory.

## Scripts

Models are saved in `trained_models`.

Training: `train.sh`, or write custom python command using `script/train_mpe.py`.
- To follow the training procedure of our paper, you have to run the training script twice, one for the warmstart policy without safety filter used in training, and the other for the final policy.
- For warmstart, set `use_safety_filter` False. Make sure all the flags are set to False in `multiagent/config.py` `RewardBinaryConfig`.
- For the second phase training, 1) set `use_safety_filter` to True, and 2) add `--model_dir="WARMSTART_MODEL_NAME"` to the python command argument where `WARMSTART_MODEL_NAME` is the model trained in the first phase. To use our method, set `POTENTIAL_CONFLICT` to True in `multiagent/config.py` `RewardBinaryConfig`.

Evaluation / Simulation: For double integrator (crazyflie) dynamics, use `eval_double_integrator.sh`. For airtaxi dynamics, use `eval_airtaxi.sh`. Or write custom python command using `script/eval_mpe.py`.

## Quick navigation

- Our main scenario for training, including the reward an the curriculum learning is defined in `multiagent/custom_scenario/navigation_graph_safe.py`. Other custom scenarios are defined in `multiagent/custom_scenario/`.
- Our dynamics and world simulation are defined in `multiagent/core.py`.
- Our safety filter is implemented in `multiagent/safety_filter.py`.
