# Melting Pot

*A suite of test scenarios for multi-agent reinforcement learning.*


[![meltingpot-tests](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

<div align="center">
  <img src="docs/images/meltingpot_montage.gif"
       alt="Melting Pot substrates"
       height="250" width="250" />
</div>

[Melting Pot 2.0 Tech Report](https://arxiv.org/abs/2211.13746)

## About

Melting Pot assesses generalization to novel social situations involving both
familiar and unfamiliar individuals, and has been designed to test a broad range
of social interactions such as: cooperation, competition, deception,
reciprocation, trust, stubbornness and so on. Melting Pot offers researchers a
set of over 50 multi-agent reinforcement learning _substrates_ (multi-agent
games) on which to train agents, and over 256 unique test _scenarios_ on which
to evaluate these trained agents. The performance of agents on these held-out
test scenarios quantifies whether agents:

*   perform well across a range of social situations where individuals are
    interdependent,
*   interact effectively with unfamiliar individuals not seen during training

The resulting score can then be used to rank different multi-agent RL algorithms
by their ability to generalize to novel social situations.

We hope Melting Pot will become a standard benchmark for multi-agent
reinforcement learning. We plan to maintain it, and will be extending it in the
coming years to cover more social interactions and generalization scenarios.

If you are interested in extending Melting Pot, please refer to the
[Extending Melting Pot](docs/extending.md) documentation.

## Installation

Melting Pot is built on top of
[DeepMind Lab2D](https://github.com/deepmind/lab2d).

### Devcontainer (x86 only)

*NOTE: This Devcontainer only works for x86 platforms. For arm64 (newer M1 Macs) users will have to follow the manual installation steps.*

This project includes a pre-configured development environment ([devcontainer](https://containers.dev)).

You can launch a working development environment with one click, using e.g. [Github
Codespaces](https://github.com/features/codespaces) or the [VSCode
Containers](https://code.visualstudio.com/docs/remote/containers-tutorial)
extension.

#### CUDA support

To enable CUDA support (required for GPU training), make sure you have the
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
package installed, and then run Docker with the `---gpus all` flag enabled. Note
that for GitHub Codespaces this isn't necessary, as it's done for you
automatically.

### Manual install

The installation steps are as follows:

1.  (Optional) Activate a virtual environment, e.g.:

    ```shell
    python3 -m venv "${HOME}/meltingpot_venv"
    source "${HOME}/meltingpot_venv/bin/activate"
    ```

2.  Install `dmlab2d` from the
    [dmlab2d wheel files](https://github.com/deepmind/lab2d/releases/tag/release_candidate_2022-03-24), e.g.:

    ```shell
    pip3 install https://github.com/deepmind/lab2d/releases/download/release_candidate_2022-03-24/dmlab2d-1.0-cp39-cp39-manylinux_2_31_x86_64.whl
    ```

    If there is no appropriate wheel (e.g. M1 chipset) you will need to install
    [`dmlab2d`](https://github.com/deepmind/lab2d) and build the wheel yourself
    (see
    [`install-dmlab2d.sh`](https://github.com/deepmind/meltingpot/blob/main/install-dmlab2d.sh)
    for an example installation script that can be adapted to your setup).

3.  Test the `dmlab2d` installation in `python3`:

    ```python
    import dmlab2d
    import dmlab2d.runfiles_helper

    lab = dmlab2d.Lab2d(dmlab2d.runfiles_helper.find(), {"levelName": "chase_eat"})
    env = dmlab2d.Environment(lab, ["WORLD.RGB"])
    env.step({})
    ```

4.  Install Melting Pot (see
    [`install-meltingpot.sh`](https://github.com/deepmind/meltingpot/blob/main/install-meltingpot.sh)
    for an example installation script):

    ```shell
    git clone -b main https://github.com/deepmind/meltingpot
    cd meltingpot
    curl -L https://storage.googleapis.com/dm-meltingpot/meltingpot-assets-1.0.0.tar.gz \
        | tar -xz --directory=meltingpot
    pip3 install .
    ```

5.  Test the Melting Pot installation:

    ```shell
    pip3 install pytest
    pytest meltingpot
    ```

6.  (Optional) Install the examples (see
    [`install-extras.sh`](https://github.com/deepmind/meltingpot/blob/main/install-meltingpot.sh)
    for an example installation script):

    ```shell
    pip install .[rllib,pettingzoo]
    ```

## Example usage

You can try out the substrates interactively with the
[human_players](meltingpot/python/human_players) scripts. For example, to play the
`clean_up` substrate, you can run:

```shell
python3 meltingpot/python/human_players/play_clean_up.py
```

You can move around with the `W`, `A`, `S`, `D` keys, Turn with `Q`, and `E`,
fire the zapper with `1`, and fire the cleaning beam with `2`. You can switch
between players with `TAB`. There are other substrates available in the
[human_players](meltingpot/python/human_players) directory. Some have multiple variants,
which you select with the `--level_name` flag.

NOTE: If you get a `ModuleNotFoundError: No module named 'meltingpot.python'`
      error, you can solve it by exporting the meltingpot home directory as
      `PYTHONPATH` (e.g. by calling `export PYTHONPATH=$(pwd)`).

### Training agents
We provide two example scripts using RLlib and [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) with [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (SB3) respectively. Note that Melting Pot is agnostic to how you train your agents, and as such, these scripts are not meant to be a suggestion on how to achieve good scores in the task suite.

#### RLlib
This example uses [RLLib](https://github.com/ray-project/ray) to train agents in self-play on a Melting Pot substrate.

First you will need to install the dependencies needed by the RLlib example:

```shell
cd <meltingpot_root>
pip3 install -e .[rllib]
```

Then you can run the training experiment using:

```shell
cd <meltingpot_root>/examples/rllib
python3 self_play_train.py
```

#### PettingZoo and Stable-Baselines3
This example uses a PettingZoo wrapper with a fully parameter shared PPO agent from SB3.

The PettingZoo wrapper can be used separately from SB3 and
can be found at [meltingpot_env.py](examples/pettingzoo/meltingpot_env.py)

```shell
cd <meltingpot_root>
pip3 install -e .[pettingzoo]
```

```shell
cd <meltingpot_root>/examples/pettingzoo
python3 sb3_train.py
```

### Evaluation

Evaluation results from the [Melting Pot 2.0 Tech Report](https://arxiv.org/abs/2211.13746)
can be viewed in the [Evaluation Notebook](notebooks/evaluation_results.ipynb).
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/meltingpot/blob/main/notebooks/evaluation_results.ipynb)

### Documentation

Full documentation is available [here](docs/index.md).

## Citing Melting Pot

If you use Melting Pot in your work, please cite the accompanying article:

```bibtex
@inproceedings{leibo2021meltingpot,
    title={Scalable Evaluation of Multi-Agent Reinforcement Learning with
           Melting Pot},
    author={Joel Z. Leibo AND Edgar Du\'e\~nez-Guzm\'an AND Alexander Sasha
            Vezhnevets AND John P. Agapiou AND Peter Sunehag AND Raphael Koster
            AND Jayd Matyas AND Charles Beattie AND Igor Mordatch AND Thore
            Graepel},
    year={2021},
    journal={International conference on machine learning},
    organization={PMLR}
}
```

## Disclaimer

This is not an officially supported Google product.

# Level of Influence (LoI) Implementation
We adopted this repo from DeepMind for the paper *Quantifying Agent Interaction in Multi-agent Reinforcement
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

`plots_FCP` for custom fictitious co-play training curve

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