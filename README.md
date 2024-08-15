# CraftFactory

The main code of craftfactory

## Config Environment

- Depend on [minerl](https://github.com/minerllabs/minerl) platform. 
- Backbone code and pretrained model is based on [Video Pre-Training](https://github.com/openai/Video-Pre-Training) 
- Simulator and part of code is based on [CraftJarvis/GROOT](https://github.com/CraftJarvis/GROOT)
- Python>=3.9 pytorch openjdk=8 gym=0.19
- Config overall varibles in your "\~/.bashrc" or "\~/.zshrc", and also config in "jarvis/assembly/__init__.py"
    - JARVISBASE_ROOT: project directory.
    - JARVISBASE_PRETRAINED: pretrained model directory where model checkpoint is put here prehead.
    - JARVISBASE_OUTPUT: the output directory of project where output checkpoint is saved.
    - JARVISBASE_TMP: temporary directory to check the rollout results.
- Config train, evaluate, and rollout settting in jarvis/arm/configs

## Dataset

- We provide the basic dataset for task-1 and task-2, you can download [here](https://drive.google.com/file/d/1LDQgUE-6EfBzTznZqTPC0B5S9c1F5y4F/view?usp=drive_link)


## Train and evaluation

Train the model:

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' python offline.py \
    policy=vpt_cursor \
    data.mode=raw \
    optimize.support=ray-lightning \
    optimize.logger=none \
    optimize.project_name=efficient_cursor
```

Evaluate the model:

```bash
CUDA_VISIBLE_DEVICES='0,1' python evaluation.py \
    policy=vpt_cursor \
    policy.from.weights="/path/to/output-weights"\
    evaluation.env=test \
    evaluation.num_workers=1 \
    evaluation.num_episodes_per_worker=30 
```

Get rollout videos:

```bash
python jarvis/assembly/mark.py [mode_code]
```

The model checkpoint should be put in `f'{JARVISBASE_PRETRAINED}/{version}/{model_code}'` (version is from `jarvis/assembly/__init__.py`) with `config.yaml` (this config.yaml will be output with checkpoint)

## Methods for Compoistional Generalization

- Group Permutation and equivariant mapping: `jarvis/arm/src/utils/permutation.py`
- Self Attention: `jarvis/arm/src/utils/selfatten.py`

## Other

Complimentary codes and datasets will be update soon.

