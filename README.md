# Safe-MARL
Multi agent navigation with safety

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



## Experiments to try out:
* Minor changes required:
    * Train for more episodes
    * Give higher penalties for collisions - Tested: Doesn't improve; rather makes it worse than earlier
    * Check impact of shared obervations - Tested (partially); Doesn't improve; Probably train for more steps and with slightly bigger networks 
    * Change network architecture size (hidden layers)
    * Make this work with more agents and obstacles
* Major changes required (need to write extra code):
    * Graph neural networks for aggregating local information
    * Modify MAPPO code for our case
    * Add communication channel

## Experiment: Optimal Minibatch:
* In safe_aam.sh:
    * num_training_threads = 4
    * n_rollout_threads = 2
    * num_mini_batch = 1
    * episode_lengths=(25n), we vary n
    * num_env_steps = episode_lengths * n_rollout_threads
    * Then mini_batch_size = n * n_rollout_threads/2 * 15/num_mini_batch
* Run with different values of n and record Training steps per second for each mini_batch_size
* Results: https://www.desmos.com/calculator/b97lzbau8q
    * Optimal mini_batch_size ~= 250