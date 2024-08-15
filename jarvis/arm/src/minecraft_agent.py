'''
author:        caishaofei-MUS2 <1744260356@qq.com>
date:          2023-05-06 19:56:05
Copyright © Team CraftJarvis All rights reserved
'''
import av 
import os
import time
import math
import pickle
from pathlib import Path
from typing import (
    Dict, List, Tuple, Union, Sequence, Mapping, Any, Optional
)
from rich.console import Console
from omegaconf import DictConfig, OmegaConf
from functools import partial
from collections import OrderedDict

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import torch
import torch.nn.functional as F

from rich import print
from pprint import pprint

import ray
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.utils.framework import TensorType
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.utils.annotations import override
from ray.rllib.algorithms.callbacks import DefaultCallbacks

import pytorch_lightning as pl
from pytorch_lightning import LightningModule

from jarvis.arm.src.utils.vpt_lib.policy import MinecraftAgentPolicy
from jarvis.stark_tech.ray_bridge import MinecraftWrapper

def tree_get(obj: Dict, keys: List, default=None):
    for key in keys:
        if key in obj:
            obj = obj[key]
        else:
            return default
    return obj

def flatten_logits(
    logits: Union[Sequence[TensorType], Dict[str, TensorType]]
) -> TensorType:
    if isinstance(logits, tuple): 
        return torch.cat(logits, dim=-1)
    elif isinstance(logits, dict):
        key_order = sorted(logits.keys())
        return torch.cat([logits[k] for k in key_order], dim=-1)
    else:
        raise ValueError("logits must be tuple or dict")

def restore_logits(
    logits: TensorType, struct_space: Union[spaces.Tuple, spaces.Dict]
) -> Union[Sequence[TensorType], Dict[str, TensorType]]:
    if isinstance(struct_space, spaces.Tuple):
        res = []
        start = 0
        for ele in struct_space:
            logit_len = ele.nvec.item()
            res.append(logits[..., start:start+logit_len].unsqueeze(1))
            start += logit_len
        return tuple(res)
    elif isinstance(struct_space, spaces.Dict):
        key_order = sorted(struct_space.spaces.keys())
        res = dict()
        start = 0
        for key in key_order:
            logit_len = struct_space.spaces[key].nvec.item()
            res[key] = logits[..., start:start+logit_len].unsqueeze(1)
            start += logit_len
        return res
    else:
        raise ValueError("struct_space must be Tuple or Dict")

def restore_action(
    actions: TensorType, struct_space: Union[spaces.Tuple, spaces.Dict]
) -> Union[Sequence[TensorType], Dict[str, TensorType]]:
    if isinstance(struct_space, spaces.Tuple):
        res = []
        for i in range(actions.shape[-1]):
            res.append(actions[..., i:i+1].long())
        return tuple(res)
    elif isinstance(struct_space, spaces.Dict):
        key_order = sorted(struct_space.spaces.keys())
        res = dict()
        for i, key in enumerate(key_order):
            res[key] = actions[..., i:i+1].long()
        return res
    else:
        raise ValueError("struct_space must be spaces.Tuple or spaces.Dict")

def resize_image(img, target_resolution = (224, 224)):
    return cv2.resize(img, dsize=target_resolution, interpolation=cv2.INTER_LINEAR)

def to_dict(kwargs: DictConfig):
    result = dict()
    for k, v in kwargs.items():
        if type(v) == DictConfig:
            result[k] = to_dict(v)
        else:
            result[k] = v
    return result

def get_num_outputs(action_space: gym.spaces.Space):
    if isinstance(action_space, spaces.Dict):
        return sum([f.nvec.item() for f in action_space.values()])
    elif isinstance(action_space, spaces.Tuple):
        return sum([f.nvec.item() for f in action_space])
    raise ValueError("action_space must be Tuple or Dict")

def build_policy(**build_kwargs):
    policy_name = build_kwargs['name']
    action_space = build_kwargs['action_space']
    
    if policy_name == 'vpt':
        model_path = build_kwargs['from'].get('model', None)
        if model_path and Path(model_path).is_file(): 
            Console().log(f"Loading predefined model from {model_path}. ")
            agent_parameters = pickle.load(Path(model_path).open("rb"))
            policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
            pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
            pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
        else:
            policy_kwargs = to_dict(build_kwargs['policy_kwargs'])
            pi_head_kwargs = to_dict(build_kwargs['pi_head_kwargs'])
        auxiliary_head_kwargs = to_dict(build_kwargs.get('auxiliary_head_kwargs', {}))
        agent_kwargs = dict(
            action_space=action_space, 
            policy_kwargs=policy_kwargs, 
            pi_head_kwargs=pi_head_kwargs, 
            auxiliary_head_kwargs=auxiliary_head_kwargs, 
        )
        policy = MinecraftAgentPolicy(**agent_kwargs)
        
        weights_path = build_kwargs['from'].get('weights', None)
        if weights_path:
            Console().log('Loaded pretrained weights from checkpoint {}'.format(weights_path))
            if Path(weights_path).is_dir():
                weights_path = os.path.join(weights_path, 'model')
            checkpoint = torch.load(weights_path)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            filter_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('policy.'):
                    filter_state_dict[k.replace('policy.', '')] = v
                else:
                    filter_state_dict[k] = v
            policy.load_state_dict(filter_state_dict, strict=False)
        
        backbone_weights_path = build_kwargs['from'].get('backbone_weights', None)
        if backbone_weights_path and Path(backbone_weights_path).exists():
            if Path(backbone_weights_path).is_dir():
                backbone_weights_path = str(Path(backbone_weights_path) / 'model')
            Console().log("Loading pretrained backbone weights from {}. ".format(backbone_weights_path))
            state_dict = torch.load(backbone_weights_path)['state_dict']
            backbone_state_dict = dict()
            param_prefix = 'agent_model.img_process.'
            for k, param in state_dict.items():
                if k.startswith(param_prefix):
                    backbone_state_dict[k[len(param_prefix):]] = param
            policy.net.img_process.load_state_dict(backbone_state_dict, strict=True)
        return policy
    
    else:
        raise NotImplementedError


SUBGOAL_SPACE = spaces.Dict({
    'recipe': spaces.Box(low=0, high=2, shape=(), dtype=np.int64),
    'table': spaces.Box(low=0, high=2, shape=(), dtype=np.int64),
    'layout': spaces.Box(low=0, high=2, shape=(9, ), dtype=np.int64),
})

HORIZON_SPACE = spaces.Box(low=-100, high=100, shape=(), dtype=np.float32)


class AgentModule(TorchRNN, LightningModule):
    
    def __init__(
        self, 
        # below are TorchModelV2 arguments
        obs_space: gym.spaces.Space, 
        action_space: gym.spaces.Space, 
        num_outputs: Optional[int] = None, 
        model_config: Optional[DictConfig] = {}, 
        name: Optional[str] = None,
        *,
        policy_config: Optional[DictConfig] = None,
        lightning_config: Optional[DictConfig] = None,
    ) -> None:
        TorchRNN.__init__(
            self, obs_space, action_space, num_outputs, model_config, name,
        )
        LightningModule.__init__(self)
        assert policy_config is not None
        self.save_hyperparameters(lightning_config)
        self.obs_space = obs_space
        self.action_space = action_space
        self.policy_config = policy_config
        self.lightning_config = lightning_config
        if self.num_outputs is None:
            self.num_outputs = get_num_outputs(self.action_space)
        self.policy = build_policy(**self.policy_config, action_space=self.action_space)
        self.init_view_requirements()
        
        self.timesteps = tree_get(
            obj=self.policy_config,
            keys=['policy_kwargs', 'timesteps'],
            default=128,
        )

    def init_view_requirements(self):
        """Maximal view requirements dict for `learn_on_batch()` and
        `compute_actions` calls.
        Specific policies can override this function to provide custom
        list of view requirements.
        """
        # Maximal view requirements dict for `learn_on_batch()` and
        # `compute_actions` calls.
        # View requirements will be automatically filtered out later based
        # on the postprocessing and loss functions to ensure optimal data
        # collection and transfer performance.
        view_reqs = self._get_default_view_requirements()
        if not hasattr(self, "view_requirements"):
            self.view_requirements = view_reqs
        else:
            for k, v in view_reqs.items():
                if k not in self.view_requirements:
                    self.view_requirements[k] = v
        
        if tree_get(
            obj=self.policy_config, 
            keys=['policy_kwargs', 'condition_embedding', 'name'], 
        ) == 'subgoal_embedding':
            self.view_requirements['subgoal'] = ViewRequirement(
                data_col="subgoal", 
                shift=-1, 
                space=SUBGOAL_SPACE
            )

        if tree_get(
            obj=self.policy_config,
            keys=['auxiliary_head_kwargs', 'horizon_head', 'enable'],
        ):
            self.view_requirements['horizon'] = ViewRequirement(
                data_col="horizon", 
                shift=-1, 
                space=HORIZON_SPACE
            )
        
        if tree_get(
            obj=self.policy_config, 
            keys=['auxiliary_head_kwargs', 'temp_ranking_head', 'enable'],
        ):
            self.view_requirements['ranking_score'] = ViewRequirement(
                data_col="ranking_score",
                shift=-1,
                space=HORIZON_SPACE,
            )

        past_obs_fusion = tree_get(
            obj=self.policy_config,
            keys=['policy_kwargs', 'past_obs_fusion'],
        )

        if past_obs_fusion:
            self.view_requirements['prev_obs'] = ViewRequirement(
                data_col="obs", 
                shift=f"-{past_obs_fusion['num_past_obs']}:-1", 
                space=self.obs_space
            )
            

    def _get_default_view_requirements(self):
        """Returns a default ViewRequirements dict.

        Note: This is the base/maximum requirement dict, from which later
        some requirements will be subtracted again automatically to streamline
        data collection, batch creation, and data transfer.

        Returns:
            ViewReqDict: The default view requirements dict.
        """
        # Default view requirements (equal to those that we would use before
        # the trajectory view API was introduced).
        return {
            SampleBatch.PREV_ACTIONS: ViewRequirement(
                data_col=SampleBatch.ACTIONS, shift=0, space=self.action_space
            ),
        }

    @override(TorchModelV2)
    def get_initial_state(self) -> List[TensorType]:
        state = [s.squeeze(0) for s in self.policy.initial_state(1)]
        return state

    @override(TorchRNN)
    def forward(
        self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        """Adds time dimension to batch before sending inputs to forward_rnn().

        You should implement forward_rnn() in your subclass."""
        flat_inputs = input_dict["obs_flat"].float()
        # This function is used in **online training**, called by rllib
        # Note that max_seq_len != input_dict.max_seq_len != seq_lens.max()
        # as input_dict may have extra zero-padding beyond seq_lens.max().
        # Use add_time_dimension to handle this
        self.time_major = self.model_config.get("_time_major", False)
        prev_action = add_time_dimension(
            input_dict["prev_actions"].long(),
            seq_lens=seq_lens,
            framework="torch",
            time_major=self.time_major,
        )
        inputs = add_time_dimension(
            flat_inputs,
            seq_lens=seq_lens,
            framework="torch",
            time_major=self.time_major,
        )
        inputs = restore_original_dimensions(inputs, self.obs_space, self.framework)
        
        if 'prev_obs' in input_dict:
            prev_obs = restore_original_dimensions(input_dict['prev_obs'], self.obs_space, self.framework)
            inputs['past_img'] = prev_obs['img']
        
        inputs['prev_action'] = restore_action(prev_action, self.action_space)
        if 'subgoal' in input_dict:
            subgoal = {
            key: add_time_dimension(
                val,
                seq_lens=seq_lens,
                framework="torch",
                time_major=self.time_major,
            ) for key, val in input_dict['subgoal'].items()}
            inputs['subgoal'] = subgoal
        output, new_state = self.forward_rnn(inputs, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])
        
        if 't' in input_dict:
            self.extra_out['t'] = input_dict['t']
        
        return output, new_state


    @override(TorchRNN)
    def forward_rnn(
        self, 
        inputs: TensorType, 
        state: List[TensorType], 
        seq_lens: TensorType, 
    ) -> Tuple[TensorType, List[TensorType]]:
        # This function is only used in online finetuning and rollout worker evaluation. 
        # In offline mode, we use loss() instead of forward_rnn(). 
        B, T = inputs['img'].shape[:2]
        print('B, T is ', B, T)
        inputs['text'] = MinecraftWrapper.tensor_to_text(inputs['text'].long())
        if isinstance(inputs['text'][0], list):
            inputs['text'] = [x[0] for x in inputs['text']]
        
        # load input condition embedding if the obs['text'] indicates  
        # a text modality source file. Otherwise, return None. 
        ice_latent = self.load_ice_latent(texts=inputs['text'])
        
        state_in = state
        dummy_first = torch.from_numpy(np.array((False,))).unsqueeze(0).repeat(B, T).to(self.device)
        result, state_out, latents = self.policy.forward( 
            obs=inputs, 
            first=dummy_first, 
            state_in=state_in, 
            stage='rollout', 
            ice_latent=ice_latent, 
        )
        self.vpred = result["vpred"]
        flatten_pi_logits = flatten_logits(result["pi_logits"])
        
        self.extra_out = {}
        
        if result.get('subgoal_logits', None) is not None:
            # import ray; ray.util.pdb.set_trace()
            pred_subgoal = OrderedDict(
                recipe=result["subgoal_logits"]['recipe'].argmax(dim=-1)[:, -1, ...],
                table=result["subgoal_logits"]['table'].argmax(dim=-1)[:, -1, ...],
                layout=result["subgoal_logits"]['layout'].argmax(dim=-1)[:, -1, ...],
            )
            self.extra_out['subgoal'] = pred_subgoal
        
        if result.get('horizon_logits', None) is not None:
            # remove temporal dimension 
            self.extra_out['horizon'] = result["horizon_logits"][:, -1]

        if 'cursor_mask' in result:
            self.extra_out['cursor_mask'] = result['cursor_mask']
            state_out.append(result['cursor_mask'])  # only use in eval
            
        if tree_get(result, ['temp_rank_head', 'feat_logits']):
            self.extra_out['ranking_score'] = tree_get(result, ['temp_rank_head', 'feat_logits'])
        
        return flatten_pi_logits, state_out

    def load_ice_latent(self, texts: List[str]) -> torch.Tensor: 
        '''
        Load the latent vector of the given text (indicates a source file). 
        '''
        assert len(texts) > 0, 'texts should not be empty. '
        example = texts[0]
        
        if example.startswith('traj:'):
            
            if not hasattr(self, 'cache_ice_latents'):
                self.cache_ice_latents = {}
            
            ice_latent = []
            for text in texts:
                video_path = text[5:]
                if video_path in self.cache_ice_latents:
                    ice_latent.append(self.cache_ice_latents[video_path])
                else:
                    # read video frames with pyav
                    traj_frames = []
                    with av.open(video_path, "r") as container:
                        for fid, frame in enumerate(container.decode(video=0)):
                            frame = frame.to_ndarray(format="rgb24")
                            traj_frames.append(frame)

                    # Split the input video into several segments, and encode each segment. 
                    num_segs = len(traj_frames) // self.timesteps
                    seg_ce_latents = []
                    for i in range(num_segs):
                        seg_frames = traj_frames[i*self.timesteps:(i+1)*self.timesteps]
                        seg_frames = (
                            torch.from_numpy( np.stack(seg_frames, axis=0) )
                            .unsqueeze(0)
                            .to(self.device)
                        )
                        T = seg_frames.shape[1]
                        # Forward once to get the trajectory latent.
                        latents = self.dummy_forward(
                            obs={
                                'img': seg_frames, 
                                'prev_action': MinecraftWrapper.get_dummy_action(B=1, T=T),
                            }, 
                        )[-1]
                        # remove temporal dimension, since they are all the same. 
                        ce_latent = latents['ce_latent'].squeeze(0)[-1, :]
                        seg_ce_latents.append(ce_latent)
                    # average the segment latents.
                    ce_latent = torch.stack(seg_ce_latents, dim=0).mean(dim=0)
                    Console().log(
                        f"Encode trajectory embedding from: {video_path};\n"
                        f"Num frames: {len(traj_frames)}, num segments: {num_segs}. ", 
                        style='yellow'
                    )
                    self.cache_ice_latents[video_path] = ce_latent
                    ice_latent.append(ce_latent)
            return torch.stack(ice_latent, dim=0)
        
        else:
            return None
            

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        return self.vpred.reshape([-1, ])

    def dummy_forward(self, obs):
        try: 
            num_past_obs = self.policy.net.past_obs_fusion['num_past_obs']
        except:
            num_past_obs = 0
            
        if num_past_obs > 0:
            shape = obs['img'].shape
            past_img = torch.zeros(
                (shape[0], num_past_obs, *shape[2:])
            ).to(device=obs['img'].device, dtype=obs['img'].dtype)
            obs['past_img'] = past_img
        
        B, T = obs['img'].shape[:2]
        state_in = self.policy.initial_state(B)
        dummy_first = (
            torch.from_numpy(np.array((False,)))
            .unsqueeze(0)
            .repeat(B, T)
            .to(self.device)
        )
        forward_result, state_out, latents = self.policy.forward(
            obs=obs, 
            first=dummy_first, 
            state_in=state_in, 
        )
        return forward_result, state_out, latents
    
    def loss(self, obs): 
        # This function is used in **offline training** to compute supervised loss,
        # called by pytorch-lightning's training loop. 
        mask = obs.pop('mask')
        forward_result, state_out, latents = self.dummy_forward(obs)
        cursor_loss_d = {}
        if 'cursor' in obs and 'pos_cursor' in forward_result:
            
            truth_pos = obs.pop('cursor')
            prob_cursor = forward_result['pos_cursor']
            mse = torch.nn.MSELoss()
            loss_cursor = mse(prob_cursor, truth_pos)
            assert 'cursor_loss_scale' in self.policy_config['policy_kwargs'], \
                f'cursor_loss_scale not found in policy_config'
            cursor_loss_scale = self.policy_config['policy_kwargs']['cursor_loss_scale']
            cursor_loss_d['loss_cursor'] = loss_cursor * cursor_loss_scale
        
        pi_logits = forward_result['pi_logits']
        vpred = forward_result['vpred']
        minerl_action = {k: v for k, v in obs['action'].items()}
        agent_action = MinecraftWrapper.env_action_to_agent(minerl_action, check_if_null=True, device=self.device)
        log_prob = self.policy.pi_head.logprob(agent_action, pi_logits)
        camera_mask = (agent_action['camera'] != 60).float().squeeze(-1)
        loss_buttons = - log_prob['buttons'].reshape(-1)[mask.reshape(-1) > 0].mean()
        loss_camera = - log_prob['camera'].reshape(-1)[(mask*camera_mask).reshape(-1) > 0].mean()

        
        if not torch.isnan(loss_camera):
            loss = loss_buttons + loss_camera
        else:
            loss = loss_buttons
            loss_camera = torch.tensor(0.0).to(self.device)
        result = {
            'loss_buttons': loss_buttons, 
            'loss_camera': loss_camera, 
            **cursor_loss_d,
        }

        if 'loss_cursor' in result:
            loss = loss + result['loss_cursor']
        
        result['loss'] = loss
        return result

    def step_core(self, batch, batch_idx, stage):
        result = self.loss(obs=batch)
        for key, value in result.items():
            self.log(f'{stage}/{key}', value, sync_dist=True)
        return {
            "loss": result['loss'],
        }

    def training_step(self, batch, batch_idx):
        # This function is used in offline training, 
        # called by pytorch-lightning's training loop.
        return self.step_core(batch, batch_idx, 'training')

    def validation_step(self, batch, batch_idx):
        # This function is used in offline training,
        # called by pytorch-lightning's training loop.
        return self.step_core(batch, batch_idx, 'validation')
    
    def configure_optimizers(self):
        # This function is used in offline training,
        # called by pytorch-lightning's training loop.
        
        learning_rate = self.lightning_config.optimize.learning_rate
        selected_discount = self.lightning_config.optimize.selected_discount
        other_discount = self.lightning_config.optimize.other_discount
        weight_decay = self.lightning_config.optimize.weight_decay
        warmup_steps = self.lightning_config.optimize.warmup_steps
        training_steps = self.lightning_config.optimize.training_steps
        
        if self.lightning_config.optimize.frozen_other:
            for name, param in self.policy.named_parameters():
                if all( (param_key not in name) for param_key in self.lightning_config.optimize.selected_keys ):
                    param.requires_grad = False
        
        all_named_parameters = dict(self.policy.named_parameters())
        all_named_parameters = dict(filter(
            lambda pair: pair[1].requires_grad,
            all_named_parameters.items()
        ))
        
        selected_keys = self.lightning_config.optimize.selected_keys
        selected_parameters = filter( 
            lambda pair: any( 
                ( param_key in pair[0] ) for param_key in selected_keys
            ), 
            all_named_parameters.items()
        )

        other_parameters = filter(
            lambda pair: all(
                ( param_key not in pair[0] ) for param_key in selected_keys
            ), 
            all_named_parameters.items()
        )

        optimizable_parameters = [
            {'params': [p for n, p in selected_parameters], 'lr': learning_rate*selected_discount}, 
            {'params': [p for n, p in other_parameters], 'lr': learning_rate*other_discount},
        ]
        
        optimizer = torch.optim.AdamW(
            params=optimizable_parameters,
            weight_decay=weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps+1)/warmup_steps, 1)
        )
        
        return {
            'optimizer': optimizer, 
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }, 
            'monitor': 'validation/loss',
        }
        

class HierarchicalActionDist(TorchDistributionWrapper):
    '''
    This action distribution servers as a wrapper for the rllib online training loop.
    '''
    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return get_num_outputs(action_space)

    def __init__(self, inputs, model):
        super(HierarchicalActionDist, self).__init__(inputs, model)
        assert model.num_outputs == HierarchicalActionDist.required_model_output_shape(model.action_space, model.model_config)
        self.restored_inputs = restore_logits(self.inputs, model.action_space)
        # self.restored_inputs = [val.unsqueeze(1) for val in self.restored_inputs]
        self.pi_head = model.policy.pi_head

    def sample(self): 
        self.last_sample = self.pi_head.sample(self.restored_inputs, deterministic=False)
        return self.last_sample
    
    def deterministic_sample(self):
        self.last_sample = self.pi_head.sample(self.restored_inputs, deterministic=True)
        return self.last_sample
    
    def logp(self, actions): 
        # to access camera off flag
        if isinstance(actions, torch.Tensor):
            actions = restore_action(
                actions=actions, struct_space=self.model.action_space
            )
        
        log_prob = self.pi_head.logprob(actions, self.restored_inputs)

        if isinstance(log_prob, tuple):
            log_prob_buttons = log_prob[0]
            log_prob_camera = log_prob[1]
        elif isinstance(log_prob, dict):
            log_prob_buttons = log_prob['buttons']
            log_prob_camera = log_prob['camera']

        minerl_action = MinecraftWrapper.agent_action_to_env(actions)
        camera_mask = (
            torch.from_numpy( (minerl_action['camera'] == 0.).sum(-1) != 0. )
            .float()
            .squeeze(-1)
            .to(log_prob_buttons.device)
        )

        res_log_prob = log_prob_buttons + log_prob_camera * camera_mask
        return torch.reshape(res_log_prob, [-1,])
    
    def entropy(self): 
        # tuple_entropy: is ((B, ), (B, )), return: (B, )
        entropy = self.pi_head.entropy(self.restored_inputs)
        # ray.util.pdb.set_trace()
        return entropy
    
    def kl(self, other: ActionDistribution) -> TensorType:
        return self.pi_head.kl_divergence(self.restored_inputs, other.restored_inputs)
