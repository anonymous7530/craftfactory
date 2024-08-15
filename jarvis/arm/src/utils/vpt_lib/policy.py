'''
author:        caishaofei-MUS2 <1744260356@qq.com>
date:          2023-05-05 15:44:33
Copyright © Team CraftJarvis All rights reserved
'''
from copy import deepcopy
from email import policy
from typing import (
    List, Dict, Optional, Callable
)
import torch

import clip
import numpy as np
import torch as th
import gymnasium as gym
from gym3.types import DictType
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from efficientnet_pytorch import EfficientNet

from jarvis.arm.src.utils.vpt_lib.action_head import make_action_head
from jarvis.arm.src.utils.vpt_lib.scaled_mse_head import ScaledMSEHead
from jarvis.arm.src.utils.vpt_lib.tree_util import tree_map
from jarvis.arm.src.utils.vpt_lib.util import FanInInitReLULayer, ResidualRecurrentBlocks
from jarvis.arm.src.utils.vpt_lib.misc import transpose

from jarvis.arm.src.utils.factory import ( 
    build_backbone, 
    build_condition_embedding_layer, 
    build_conditioning_fusion_layer, 
    build_auxiliary_heads, 
    ActionEmbedding, 
    PastObsFusion, 
) 

from jarvis.arm.src.utils.hogepoge import (
    PositionalBox,
    IndexEmbeddingSimple,
    # IndexListEmbedding,
    ItemEmbedding,
    gaussian_heatmap,
)

from jarvis.arm.src.utils.hogepoge import CondEmbeddingWrapper as IndexListEmbedding

class ScalableMinecraftPolicy(nn.Module):
    """
    :param recurrence_type:
        None                - No recurrence, adds no extra layers
        lstm                - (Depreciated). Singular LSTM
        multi_layer_lstm    - Multi-layer LSTM. Uses n_recurrence_layers to determine number of consecututive LSTMs
            Does NOT support ragged batching
        multi_masked_lstm   - Multi-layer LSTM that supports ragged batching via the first vector. This model is slower
            Uses n_recurrence_layers to determine number of consecututive LSTMs
        transformer         - Dense transformer
    :param init_norm_kwargs: kwargs for all FanInInitReLULayers.
    """

    def __init__(  
        self,
        recurrence_type="lstm",
        # backbone_type="IMPALA",
        obs_processing_width=256,
        hidsize=512,
        # single_output=False,  # True if we don't need separate outputs for action/value outputs
        init_norm_kwargs={},
        # Unused argument assumed by forc.
        input_shape=None,  # pylint: disable=unused-argument
        active_reward_monitors=None,      
        img_statistics=None,
        first_conv_norm=False,
        attention_mask_style="clipped_causal",
        attention_heads=8,
        attention_memory_size=2048,
        use_pointwise_layer=True,
        pointwise_ratio=4,
        pointwise_use_activation=False,
        n_recurrence_layers=1,
        recurrence_is_residual=True,
        timesteps=None,
        use_pre_lstm_ln=True,  # Not needed for transformer
        # below are custimized arguments
        condition_embedding=None,
        action_embedding=None,
        conditioning_fusion=None,
        past_obs_fusion=None,
        action_space=None,
        backbone_kwargs={},
        auxiliary_backbone_kwargs={},
        condition_before_vision=False,
        **unused_kwargs,
    ):
        super().__init__()
        assert recurrence_type == "transformer"

        for key in ['cursor_on', 'posbox_on', 'recipe_on', 'recipe_start_on',
                    'item_on', 'item_mode', 'recipe_start_dim', 'recipe_end_dim']:
            assert key in unused_kwargs, f'{key} not in unused kwargs...'
            setattr(self, key, unused_kwargs[key])

        if 'cond_type' in unused_kwargs:
            cond_type = unused_kwargs['cond_type']
        else:
            cond_type = 'self'
        # cond_type = 'poliarch'  # self, poliarch, howm, equirnn
        print(f'<<---   using cond_type: {cond_type} --->>')
        
        self.hidsize = hidsize
        self.cursor_mid = 128

        if self.cursor_on:
            self.cursor_ly1 = FanInInitReLULayer(hidsize, self.cursor_mid, 
                layer_type='linear', use_activation=False)
            self.cursor_ly2 = FanInInitReLULayer(self.cursor_mid, 2,
                layer_type='linear', use_activation=False)
                
        if self.recipe_on:
            if self.recipe_start_on:
                # self.recipe_embedding_start = IndexEmbeddingSimple(output_dim=512, 
                #                                     emb_dim=self.recipe_start_dim)
                self.recipe_embedding_end = IndexListEmbedding(output_dim=512, 
                                                               emb_dim=47, cond=cond_type)
            # self.recipe_embedding_end = IndexEmbeddingSimple(output_dim=512,
            #                                     emb_dim=self.recipe_end_dim)
            self.recipe_embedding_end = IndexListEmbedding(output_dim=512, 
                                                           emb_dim=47, cond=cond_type)
        
        if 'use_film' in unused_kwargs and unused_kwargs['use_film']:
            print('>>>-- init with use_film parameters --<<<')
            self.use_film = unused_kwargs['use_film']
        if 'fuse_cond_hidden' in unused_kwargs and unused_kwargs['fuse_cond_hidden']:
            print('>>>-- init with fuse_cond_hidden parameters --<<<')
            self.fuse_cond_hidden = unused_kwargs['fuse_cond_hidden']
            self.cond_hidden_mapping = nn.Linear(512, hidsize)
            self.cond_hidden_mapping.weight.data.normal_(0, 0.001)  # initial nearly from zeros

        ''' 
            choices: 'index', 'clip-text', 'clip-image', 'image',
            when item_mode='image', the input should be frame like [..., H, W, 3],
                and need to change RawDataset as well
        '''
        if self.item_on:
            # self.item_embedding = ItemEmbedding(output_dim=512, emb_dim=631, 
            #                                     mid_dim=512, mode=self.item_mode)
            self.item_embedding = IndexListEmbedding(output_dim=512, 
                                                     emb_dim=631, cond=cond_type)
        
        # self.single_output = single_output
        self.init_norm_kwargs = init_norm_kwargs
        self.dense_init_norm_kwargs = deepcopy(init_norm_kwargs)
        if self.dense_init_norm_kwargs.get("group_norm_groups", None) is not None:
            self.dense_init_norm_kwargs.pop("group_norm_groups", None)
            self.dense_init_norm_kwargs["layer_norm"] = True
        if self.dense_init_norm_kwargs.get("batch_norm", False):
            self.dense_init_norm_kwargs.pop("batch_norm", False)
            self.dense_init_norm_kwargs["layer_norm"] = True
        
        backbone_kwargs = {**backbone_kwargs, **unused_kwargs}
        backbone_kwargs['hidsize'] = hidsize
        backbone_kwargs['init_norm_kwargs'] = init_norm_kwargs
        backbone_kwargs['dense_init_norm_kwargs'] = self.dense_init_norm_kwargs

        result_modules = build_backbone(**backbone_kwargs)
        self.img_preprocess = result_modules['preprocessing']
        self.img_process = result_modules['obsprocessing']

        # build auxiliary backbones
        
        self.condition_before_vision = condition_before_vision

        self.condition_embedding = condition_embedding
        self.condition_embedding_layer = build_condition_embedding_layer(
            hidsize=hidsize,
            **self.condition_embedding
        ) if self.condition_embedding else None
        
        # conditioning_fusion
        # action_embedding
        
        self.recurrence_type = recurrence_type

        # The recurrent layer is implemented by OpenAI
        self.recurrent_layer = ResidualRecurrentBlocks(
            hidsize=hidsize,
            timesteps=timesteps,
            recurrence_type=recurrence_type,
            is_residual=recurrence_is_residual,
            use_pointwise_layer=use_pointwise_layer,
            pointwise_ratio=pointwise_ratio,
            pointwise_use_activation=pointwise_use_activation,
            attention_mask_style=attention_mask_style,
            attention_heads=attention_heads,
            attention_memory_size=attention_memory_size,
            n_block=n_recurrence_layers,
        ) 

        self.lastlayer = FanInInitReLULayer(hidsize, hidsize, layer_type="linear", **self.dense_init_norm_kwargs)
        self.final_ln = th.nn.LayerNorm(hidsize)

    def output_latent_size(self):
        return self.hidsize

    def extract_vision_feats(
        self, 
        obs_preprocess: Callable, 
        obs_process: Callable,
        obs: Dict, 
        ce_latent: Optional[th.Tensor] = None
    ) -> th.Tensor:
        '''
        Extract vision features from the input image sequence. 
        The image sequence can be current episode's obsercations or 
        trajectoty of past observations (reused to encode trajectory), 
        in such case, the pre_cond should be `None`. 
        Also, you can specify whether to use the past observations. 
        args:
            obs_preprocess: Callable, preprocess the input image.
            obs_process: Callable, process the input image.
            obs['past_img']: (B, num_past, C, H, W) or Optional None 
            obs['img']: (B, T, C, H, W) 
            ce_latent: (B, T, C) 
            enable_past_fusion: bool, whether to use past observations. 
        return: 
            vision_feats: (B, T, C) or (B, T, C, X, X) or 
                          (B, T + num_past, C) or (B, T + num_past, C, X, X)
        '''
        img = obs['img']
        B, T = img.shape[:2]
        x = obs_preprocess(img)  # transpose, scale
        vision_feats = obs_process(x, cond=ce_latent)
        vision_feats = vision_feats.reshape((B, T) + vision_feats.shape[2:])
        return vision_feats

    def forward(self, obs, state_in, context, ice_latent=None):
        '''
        Args:
            obs: Dict, observation. 
            state_in: Dict, input state. 
            ice_latent: Optional[th.Tensor], input condition embedding. 
                For example, in the inference stage, the condition embedding 
                is encoded before running this forward function. Then it 
                should use ice_latent argument. 
        '''

        B, T = obs['img'].shape[:2]
        ce_latent = torch.zeros((B, T, 512), dtype=torch.float32, device=obs['img'].device)
        if self.recipe_on:
            
            rcp_start, rcp_end = obs.get('index_1').long(), obs.get('index_2').long()
            assert (rcp_start >= 0).all() and (rcp_start < self.recipe_start_dim).all(), \
                f'rcp start not >= 0 and < {self.recipe_start_dim}'
            assert (rcp_end >= 0).all() and (rcp_end < self.recipe_end_dim).all(), \
                f'rcp end not >= 0 and < {self.recipe_end_dim}'
            
            recipe_emb_end = self.recipe_embedding_end(rcp_end)
            ce_latent = ce_latent + recipe_emb_end
            if self.recipe_start_on:
                recipe_emb_start = self.recipe_embedding_start(rcp_start)
                ce_latent = ce_latent + recipe_emb_start
            
        if self.item_on:
            item_data = obs.get('item')
            # it's a remedy, todo check why sometimes item_data will be float32 <--
            if not item_data.dtype == torch.int64:
                item_data = item_data.long()
            ce_latent = ce_latent + self.item_embedding(item_data)

        # Extract vision features from the input image sequence. 
        # The result feature may be a 3D tensor or a 5D tensor. 
        if getattr(self, 'use_film', None) is None or self.use_film:
            vi_latent = self.extract_vision_feats(
                obs_preprocess=self.img_preprocess, 
                obs_process=self.img_process, 
                obs=obs, 
                ce_latent=ce_latent,
            )
        else:
            vi_latent = self.extract_vision_feats(
                obs_preprocess=self.img_preprocess, 
                obs_process=self.img_process, 
                obs=obs, 
                ce_latent=None,
            )
        
        ''' predict cursor position by visual feature '''
        cursor_pos = {}
        if self.cursor_on:
            curs_emb = self.cursor_ly1(vi_latent)
            curs_emb = self.cursor_ly2(curs_emb)
            pred_cursor = torch.sigmoid(curs_emb)
            cursor_pos['pos_cursor'] = pred_cursor
            cursor_pos['cursor_mask'] = gaussian_heatmap(
                height=128, width=128, center=pred_cursor*128,
                sigma=5., device=pred_cursor.device)

        av_latent = None  # auxiliry head
        
        # Use the condition embeddings to condition the vision features. 
    
        if getattr(self, 'fuse_cond_hidden', None) is not None and self.fuse_cond_hidden:
            vi_latent = vi_latent + self.cond_hidden_mapping(ce_latent)

        ov_latent = vi_latent 
        oa_latent = av_latent 
        # Here, we use the original vision features for decision making. 
        x = ov_latent
       
        # Transformer
        x, state_out = self.recurrent_layer(x, context["first"], state_in)
        tf_latent = x

        x = F.relu(x, inplace=False)
        x = self.lastlayer(x)
        x = self.final_ln(x)
        pi_latent = vf_latent = x

        # Return intermediate latents for decision making and other auxiliary tasks. 
        result_latents = {
            "vi_latent": vi_latent,
            "ov_latent": ov_latent,
            "av_latent": av_latent,
            "oa_latent": oa_latent,
            "pi_latent": pi_latent,
            "vf_latent": vf_latent,
            "tf_latent": tf_latent,
            "ce_latent": ce_latent,
            **cursor_pos,
        }
        
        return result_latents, state_out

    def initial_state(self, batchsize):
        return self.recurrent_layer.initial_state(batchsize)

class MinecraftAgentPolicy(nn.Module):
    def __init__(self, action_space, policy_kwargs, pi_head_kwargs, auxiliary_head_kwargs):
        super().__init__()
        self.net = ScalableMinecraftPolicy(**policy_kwargs, action_space=action_space)

        self.action_space = action_space

        self.value_head = self.make_value_head(self.net.output_latent_size())
        self.pi_head = self.make_action_head(self.net.output_latent_size(), **pi_head_kwargs)
        self.auxiliary_heads = nn.ModuleDict(build_auxiliary_heads(
            auxiliary_head_kwargs=auxiliary_head_kwargs, 
            hidsize=policy_kwargs['hidsize'],
        ))

    def make_value_head(self, v_out_size: int, norm_type: str = "ewma", norm_kwargs: Optional[Dict] = None):
        return ScaledMSEHead(v_out_size, 1, norm_type=norm_type, norm_kwargs=norm_kwargs)

    def make_action_head(self, pi_out_size: int, **pi_head_opts):
        return make_action_head(self.action_space, pi_out_size, **pi_head_opts)

    def initial_state(self, batch_size: int):
        return self.net.initial_state(batch_size)

    def reset_parameters(self):
        super().reset_parameters()
        self.net.reset_parameters()
        self.pi_head.reset_parameters()
        self.value_head.reset_parameters()

    def forward(
        self, 
        obs: Dict, 
        first: th.Tensor, 
        state_in: List[th.Tensor], 
        stage: str = 'train', 
        ice_latent: Optional[th.Tensor] = None, 
        **kwargs
    ) -> Dict[str, th.Tensor]:
        if isinstance(obs, dict):
            # We don't want to mutate the obs input.
            obs = obs.copy()
            mask = obs.pop("mask", None)
        else:
            mask = None
        latents, state_out = self.net(
            obs=obs, 
            state_in=state_in, 
            context={"first": first}, 
            ice_latent=ice_latent,
        )
        result = {
            'pi_logits': self.pi_head(latents['pi_latent']),
            'vpred': self.value_head(latents['vf_latent']),
        }

        if 'pos_cursor' in latents:
            result['pos_cursor'] = latents['pos_cursor']
            result['cursor_mask'] = latents['cursor_mask']
        
        for head, module in self.auxiliary_heads.items():
            result[head] = module(latents, stage=stage)
        
        return result, state_out, latents
