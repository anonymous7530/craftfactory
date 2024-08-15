import os
import av
import cv2
import time
from pathlib import Path
from typing import (
    Dict, List, Union, Sequence, Mapping, Any, Optional
)
from functools import partial

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import itertools
from tqdm import tqdm

from omegaconf import DictConfig, OmegaConf
from rich import print
from rich.console import Console
from rich.pretty import Pretty
from pprint import pprint
from watermark import watermark

import wandb
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger, MLFlowLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import ray
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.lightning import (
    LightningTrainer,
    LightningConfigBuilder,
    LightningCheckpoint,
)

import GPUtil

from jarvis import JARVISBASE_TMP, JARVISBASE_OUTPUT
from jarvis.stark_tech.ray_bridge import MinecraftWrapper
from jarvis.arm.src.dataset import MineRLDataModule
from jarvis.arm.src.minecraft_agent import AgentModule, to_dict


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

seed = 42
seed_everything(seed)

console = Console()


class SpeedMonitor(pl.Callback):
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.time_start = time.time()
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        time_end = time.time()
        time_cost = time_end - self.time_start
        trainer.logger.log_metrics({'training/it_p_s': 1/time_cost}, step=trainer.global_step)


def build_common_configs(params: DictConfig) -> Dict:
    data_module_config = dict(
        mode=params.data.mode,
        batch_size=params.optimize.batch_size,
        num_workers=params.optimize.num_workers,
        train_shuffle=params.optimize.train_shuffle,
        # below are parameters for dataset manager
        dataset_dirs=params.data.dataset_dirs,
        enable_video=params.data.enable_video,
        enable_action=params.data.enable_action,
        enable_clip=params.data.enable_clip,
        enable_contractor_info=params.data.enable_contractor_info,
        enable_condition_info=params.data.enable_condition_info,
        enable_cursor=params.data.enable_cursor,  # <- bingo
        frame_width=params.data.frame_width,
        frame_height=params.data.frame_height,
        decode_library=params.data.decode_library,
        # below are parameters for extension dataset
        extension_dirs=params.data.get('extension_dirs'),
        sample_mode=params.data.get('sample_mode'),
        samples_per_goal=params.data.get('samples_per_goal'),
        padding_left=params.data.get('padding_left'),
        padding_right=params.data.get('padding_right'),
        win_len=params.data.get('win_len'),
        skip_frame=params.data.get('skip_frame'),
        split_ratio=params.data.get('split_ratio'),
        goal_list=params.data.get('goal_list'), 
        fixed_start=params.data.get('fixed_start', False),
        percent_datause=params.data.get('percent_datause'),
        composition_dropout=params.data.get('composition_dropout'),
    )
    speed_monitor = SpeedMonitor()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints', 
        monitor='validation/loss',
        filename='best-epoch-{epoch}',
    )
    callbacks = [
        lr_monitor, checkpoint_callback, speed_monitor, 
    ]
    agent_config = dict(
        obs_space=MinecraftWrapper.get_obs_space(),
        action_space=MinecraftWrapper.get_action_space(), 
        policy_config=params.policy, 
        lightning_config=params, 
        name="Agent(VPT)", 
    )
    
    loggers = []
    if 'wandb' in params.optimize.logger:
        wandb_logger = WandbLogger(project=params.optimize.project_name, id=params.optimize.experiment_name)
        loggers.append(wandb_logger)
    elif 'mlflow' in params.optimize.logger:
        mlf_logger = MLFlowLogger(experiment_name=params.optimize.project_name, tracking_uri="file:./mlruns")
        loggers.append(mlf_logger)
    else:
        csv_logger = CSVLogger(".")
        loggers.append(csv_logger)

    trainer_config = dict(
        logger=loggers,
        max_epochs=params.optimize.max_epochs,
        accelerator=params.optimize.accelerator,
        accumulate_grad_batches=params.optimize.accumulate_grad_batches,
        devices=params.optimize.devices,
        strategy=params.optimize.strategy,
        precision=params.optimize.precision,
        callbacks=callbacks,
    )
    return {
        'data_module_config': data_module_config,
        'agent_config': agent_config,
        'trainer_config': trainer_config,
    }


def train_with_ray_lightning(params: DictConfig):
    os.environ['WANDB_DISABLE_SERVICE'] = 'True'
    common_configs = build_common_configs(params)
    minerl_data = MineRLDataModule(**common_configs['data_module_config']) 
    lightning_config = (
        LightningConfigBuilder()
        .strategy('ddp')
        .module(cls=AgentModule, **common_configs['agent_config'])
        .trainer(**common_configs['trainer_config'])
        .fit_params(datamodule=minerl_data)
        .build()
    )
    scaling_config = ScalingConfig(
        num_workers=params.optimize.devices,
        use_gpu=True,
        resources_per_worker={'CPU': params.optimize.num_workers, 'GPU': 1},
    )
    project_name = params.optimize.project_name
    run_config = RunConfig(
        name=project_name,
        local_dir=f"{JARVISBASE_OUTPUT}/ray_results",
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="validation/loss",
            checkpoint_score_order="min",
        ),
    )

    lightning_trainer = LightningTrainer(
        lightning_config=lightning_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    result = lightning_trainer.fit()
    print(result)
    
    results = tuner.fit()
    best_result = results.get_best_result(metric="validation/loss", mode="min")


def train_with_lightning(params: DictConfig):
    common_configs = build_common_configs(params)
    minerl_data = MineRLDataModule(**common_configs['data_module_config'])
    agent = AgentModule(**common_configs['agent_config'])
    trainer = pl.Trainer(**common_configs['trainer_config'])
    if params.mode == 'train':
        trainer.fit(agent, datamodule=minerl_data)
    else:
        trainer.validate(agent, datamodule=minerl_data)

def tree_launch(input: Dict, device='cuda'):
    for k, v in input.items():
        if isinstance(v, torch.Tensor):
            input[k] = v.to(device)
        elif isinstance(v, dict):
            input[k] = tree_launch(v, device)
    return input


def plot_embedding_2D(data, label, title="traj_space"):
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).values()
    color_cycle = itertools.cycle(colors)
    label_to_color = dict()
    
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure(dpi=150, figsize=(10, 10))
    for i in range(data.shape[0]):
        
        if label[i] in label_to_color:
            c = label_to_color[label[i]]
        else:
            c = next(color_cycle)
            label_to_color[label[i]] = c

        plt.scatter(
            data[i, 0], 
            data[i, 1], 
            color=c,
            s=10,
        )
    
    # draw labels in the bottom right of the figure
    for i, (k, v) in enumerate(label_to_color.items()):
        plt.text(
            0.75, 
            0.05 + i * 0.025, 
            k, 
            color=v,
            fontdict={'weight': 'bold', 'size': 9},
            transform=plt.gca().transAxes
        )
        
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


@torch.no_grad()
def visualize_traj_space(params: DictConfig):
    
    common_configs = build_common_configs(params)
    minerl_data = MineRLDataModule(**common_configs['data_module_config'])
    minerl_data.setup(stage='fit')
    agent = AgentModule(**common_configs['agent_config']).to("cuda")
    agent.eval()
    
    texts = []
    ce_latents = []
    # draw T-SNE
    for idx, obs in enumerate(tqdm(minerl_data.train_dataloader())):
        if idx > params.visualize.num_points:
            break
        # obs is tree of dict, now launch them to cuda
        tree_launch(obs, device='cuda')
        forward_result, _, latents = agent.train_forward(obs=obs)
        texts += obs['text']
        ce_latents += list(latents['ce_latent'][:, -1, :].cpu().numpy())
    ce_latents = np.array(ce_latents)

    print("Start ploting T-SNE...")
    # texts is the label of each point
    # ce_latents is the latent of each point
    tsne = TSNE(n_components=2, random_state=0)
    result = tsne.fit_transform(ce_latents)
    fig = plot_embedding_2D(result, texts)
    plt.savefig("traj_space.png")


@hydra.main(config_path="configs", config_name="offline_defaults")
def main(params):
    OmegaConf.resolve(params)
    print(params)
    
    if params.mode in ['train', 'validate']:
        if params.optimize.support == 'ray-lightning':
            ray.init(_temp_dir=f'{os.getenv("HOME")}/ray')
            train_with_ray_lightning(params)
        elif params.optimize.support == 'lightning':
            train_with_lightning(params)
    elif params.mode == 'visualize':
        visualize_traj_space(params)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()