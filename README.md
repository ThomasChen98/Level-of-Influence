# Level of Influence (LoI) Implementation
We adopted this repo from [DeepMind Melting Pot](https://github.com/google-deepmind/meltingpot) for the paper *Quantifying Agent Interaction in Multi-agent Reinforcement
Learning for Cost-efficient Generalization*. All additional implementations are located in `./MARL/`. We also created some customized configurations located in `./meltingpot/python/configs/substrates`.

# File Structure

## Files

### Training File

`SP_train_origin.py` from Deep Mind sample self-play training code

`SP_train.py` for custom self-play training

`PP_train.py` for custom population-play training

`SP_eval_train.py` for custom self-play training (1 seed) use by tournament evaluation

### Evaluation File

`index_evaluate.py` for index evaluation from checkpoints obtained by `SP_train_origin.py`

`tournament.py` for generating tournament heat map

`correlation.ipynb` for calculating correlation between index and performance

### Plotting File

`bars.ipynb` for visualizatin tournament bar plot

`index.ipynb` for visualizing index

`index_seed.ipynb` for visualizing the effect of LoI approximation on resource allocation performance

`tournament.ipynb` for visualizing tournament heat map

`learning_curves_SP.ipynb` for custom self-play training curve

`learning_curves_SP_eval.ipynb` for custom self-play (1 seed) training curve

`learning_curves_PP.ipynb` for custom population-play training curve

`learning_curves_tune.ipynb` for Deep Mind sample training curve

`resource_allocation.ipynb` for visualizing the resource allocation comparison

## Folders

### Naming Rule

`{E}{S}{C}`

- `{E}` for environment
    - `c` - chicken
    - `pc` - pure coordination
    - `pd` - prisoners dilemma
    - `sh` - stag hunt

- `{C}` for configuration size
    - `s` - small
    - `m` - medium
    - `l` - large
    - `o` - obstacle
- `{S}` for seed number

### Checkpoints Folder

`SP_checkpoints` for custom self-play checkpoints

`SP_eval_checkpoints` for custom self-play (1 seed) checkpoints

`PP_checkpoints` for custom population-play checkpoints

`~/ray_results/PPO/` for Deep Mind demo self-play checkpoints

File structure

`{METHOD}_checkpoints`

→`{E}{S}{C}` 

→`seed_{S}` 

→`gen_{GEN}`

### Outputs Folder

`SP_outputs` for custom self-play output

`SP_eval_outputs` for custom self-play (1 seed) output

`PP_outputs` for custom population-play output

File name

`{E}{S}{C}.txt` 

### Logs Folder

`SP_logs` for custom self-play output

`SP_eval_logs` for custom self-play (1 seed) output

`PP_logs` for custom population-play output

File name

`{E}{S}{C}.npz`

Numpy file structure

`data['timesteps']` - (# seed, # step)

`data['policy_reward_min']` - (# seed, agent id, # step)

`data['policy_reward_mean']` - (# seed, agent id, # step)

`data['policy_reward_max']` - (# seed, agent id, # step)

### Plots Folder

`plots_bar` for tournament bar plots and resource allocation

`plots_index` for index plots

`plots_seed` for the effect of LoI approximation on resource allocation performance

`plots_tournament` for tournament heat map plots

`plots_SP` for custom self-play training curve

`plots_SP_eval` for custom self-play (1 seed) training curve

`plots_PP` for custom population-play training curve

`plots_tune` for Deep Mind sample training curve

### Data Folder

`index_data` for index `.npz` files

`tournament_data` for tournament `.npz` files

# Training Procedure

## Create Folder & Files

1. Create folders for logs `{METHOD}_logs`, outputs `{METHOD}_outputs/{E}{S}{C}.txt`, and checkpoints `{METHOD}_checkpoints/{E}{S}{C}`

## Modify Codes

### New Training

1. Change `substrate_name` string name of function `get_config()`
2. Change `save_path`, `checkpoints_path`, `log_path` in function `main()`
3. Change `num_gens` and `seeds` in function `main()`
4. Change `continuous_training` to `False`

### Continuous Training

1. Clean output `.txt` file to the nearest separate line `###`
2. Delete excessive generation folders `gen_{GEN}` in checkpoints folder to make sure the latest generation are the same across all different seeds
3. Change `substrate_name` string name of function `get_config()`
4. Change `save_path`, `checkpoints_path`, `log_path` in function `main()`
5. Change `num_gens` and `seeds` in function `main()`
6. Change `continuous_training` to `True`
7. Change `starting_gen` to the latest generation number in checkpoints folder

## Training Commands

1. Create `tmux` session

```jsx
tmux new -s meltingpot
cd meltingpot
```

1. Run training code

```jsx
python {METHOD}_train.py
```
