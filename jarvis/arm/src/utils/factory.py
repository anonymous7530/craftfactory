'''
author:        caishaofei <1744260356@qq.com>
date:          2023-04-20 13:44:08
Copyright © Team CraftJarvis All rights reserved
'''
import math
import random
from rich.console import Console
from copy import deepcopy
from functools import partial
from typing import Dict, Optional, Union, List, Any
from collections import OrderedDict


from gymnasium import spaces
import clip
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms as T
from torch.nn.parameter import Parameter

from jarvis.stark_tech.ray_bridge import MinecraftWrapper
from jarvis.arm.src.utils.vpt_lib.impala_cnn import ImpalaCNN
from jarvis.arm.src.utils.vpt_lib.util import FanInInitReLULayer, ResidualRecurrentBlocks
from jarvis.arm.src.utils.sam_lib.image_encoder import ImageEncoderViT
from jarvis.arm.src.utils.efficientnet_lib import EfficientNet
from jarvis.arm.src.utils.transformers import GPT 
from jarvis.arm.src.utils.hogepoge import SimpleCrossAttention


class PositionalEncoding(nn.Module): 
    "Implement the PE function." 
    
    def __init__(self, d_model, max_len=256): 

        super(PositionalEncoding, self).__init__() 
        # Compute the positional encodings once in log space. 
        pe = th.zeros(max_len, d_model) 
        position = th.arange(0, max_len).unsqueeze(1) 
        div_term = (
            th.exp(th.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)) 
        )
        pe[:, 0::2] = th.sin(position * div_term) 
        pe[:, 1::2] = th.cos(position * div_term) 
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe) 
    
    def forward(self, x): 
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False) 
        return x 

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        B, T, C = x.shape
        pos = th.arange(T, device=x.device).repeat(B, 1)
        return x + self.pos_embed(pos)

class SelfAttentionNet(nn.Module):
    
    def __init__(
        self, 
        input_size: int, 
        num_heads: int, 
        num_layers: int = 1, 
        pos_enc: str = 'learnable',
        **kwargs
    ) -> None:
        super().__init__()
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(input_size, num_heads, batch_first=True) for _ in range(num_layers)
        ])
        if pos_enc == 'learnable':
            self.pos_embed = LearnablePositionalEncoding(input_size)
        else:
            self.pos_embed = PositionalEncoding(input_size)
    
    def forward(self, qkv):
        qkv = self.pos_embed(qkv)
        for _ in range(len(self.attentions)):
            qkv, _ = self.attentions[_](qkv, qkv, qkv)
        return qkv

class CrossAttentionLayer(nn.Module):
    
    def __init__(
        self, 
        input_size: int, 
        num_heads: int, 
        num_layers: int = 1, 
        pos_enc: str = 'learnable',
        **kwargs
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(input_size, num_heads, batch_first=True)

        if pos_enc == 'learnable':
            self.pos_embed = LearnablePositionalEncoding(input_size)
        else:
            self.pos_embed = PositionalEncoding(input_size)

    def forward(self, q, kv):
        q = self.pos_embed(q)
        kv = self.pos_embed(kv)
        
        attn_output, attn_weights = self.attention(q, kv, kv)
        return attn_output

class PastObsFusion(nn.Module):
    
    def __init__(
        self, 
        hidsize: int, 
        num_past_obs: int = 3, 
        num_heads: int = 4, 
        **kwargs, 
    ):
        super().__init__()
        self.hidsize = hidsize
        self.num_past_obs = num_past_obs
        self.attention_net = SelfAttentionNet(input_size=hidsize, num_heads=num_heads, num_layers=2)
    
    def forward(self, imgs):
        '''
        args:
            imgs: (B, num_past_obs + T, C)
        '''
        B, seq, C = imgs.shape
        T = seq - self.num_past_obs
        imgs_clauses = []
        for i in range(self.num_past_obs+1):
            imgs_clauses += [imgs[:, i:i+T, :]]
        extended_imgs = th.stack(imgs_clauses, dim=2) # (B, T, num_past_obs+1, C)
        x = extended_imgs.reshape(B*T, self.num_past_obs+1, C)
        
        x = self.attention_net(qkv=x)
        x = x[:, -1, :].reshape(B, T, C)
        return x

class BaseHead(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
    
    def forward(self, latents: Dict[str, th.Tensor], **kwargs) -> Any:
        '''
        Predict auxiliary task results based on the latents. 
        The returned results will be feed to the loss function as the `pred` term. 
        '''
        raise NotImplementedError
    
    def loss(self, obs, pred, mask=None, **kwargs) -> Any:
        '''
        `obs` terms refers to the original info that sampled from the dataset. 
        `pred` terms refers to the predicted results from the forward function. 
        You are supposed to return metric dict in this function. 
        '''
        raise NotImplementedError

class TempRankHead(BaseHead):
    '''
    The Temporal Ranking Head serves as the auxiliary task for learning task progress
    by teach the model to rank the temporal order of sampled frames. This method eli-
    mates the need for the exact temporal distance between frames,  which varies from 
    trajectories to trajectories w.r.t dynamic partially observable environments. 
    '''
    def __init__(
        self, 
        hidsize: int, 
        num_intra_samples: int = 1, 
        num_inter_samples: int = 1, 
        interval: int = 1,
        enable_condition: bool = False, 
        enable_diff_traj: bool = False,
        condition_dim: int = 512, 
        compare_method: Union['bilinear', 'scalar'] = 'bilinear', 
        vision_feature: str = 'vi_latent', 
        **kwargs, 
    ) -> None:
        '''
        args:
            hidsize: the hidden size of the temporal rank head.
            num_intra_samples: the number of paired frames to be sampled from the input trajectory. 
            num_inter_samples: the number of paired frames to be sampled from the other trajectories. 
            interval: the maximum temporal distance between the paired frames. 
            enable_condition: whether to base on the condition to compute the temporal rank.
            enable_diff_traj: whether to sample the paired frames from different trajectories. 
            condition_dim: the dimension of the condition feature. 
            compared_method: the method to compare the paired frames. 
        '''    
        super().__init__()
        self.hidsize = hidsize
        self.num_intra_samples = num_intra_samples
        self.num_inter_samples = num_inter_samples
        self.interval = interval
        self.enable_condition = enable_condition
        self.enable_diff_traj = enable_diff_traj
        self.condition_dim = condition_dim
        self.compare_method = compare_method
        self.vision_feature = vision_feature
        Console().log(f"[TempRankHead INFO] with ranking method {compare_method}. ")
        
        if self.compare_method == 'bilinear':
            if self.enable_condition:
                self.lz = nn.Linear(self.condition_dim, self.hidsize)
            self.A = nn.Parameter(th.randn(self.hidsize, self.hidsize) * 1e-3)
            self.B = nn.Parameter(th.randn(self.hidsize, self.hidsize) * 1e-3)
        
        elif self.compare_method == 'scalar':
            if self.enable_condition:
                self.lz = nn.Linear(self.condition_dim, self.hidsize)
                self.W = nn.Parameter(th.randn(self.hidsize, self.hidsize) * 1e-3)
            else:
                self.W = nn.Sequential(
                    nn.Linear(self.hidsize, self.hidsize // 2),
                    nn.ReLU(),
                    nn.Linear(self.hidsize // 2, 1), 
                )
        
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        self.all_pairs = [(0, 0)]
        self.acc_marks = [0]
        for i in range(256):
            if i >= self.interval:
                for j in range(self.interval, i - self.interval):
                    self.all_pairs += [(i, j), (j, i)]
            self.acc_marks += [len(self.all_pairs)]
        
    
    def sample_neg_traj_idxs(
        self, 
        ce_latent: Optional[th.Tensor] = None, 
        epsilon : float = 1e-5, 
    ) -> th.Tensor: 
        '''
        Sample the trajectories with different condition latents. 
        '''
        B = ce_latent.shape[0]
        res = []
        for i in range(B):
            options = []
            for j in range(B):
                if (
                    i == j or (
                        ce_latent is not None and 
                        ((ce_latent[i] - ce_latent[j]).abs().sum() <= epsilon).all().item()
                    )
                ): 
                    continue
                options.append(j)
            
            if len(options) > 0:
                lucky_traj = [random.choice(options)]
            else:
                lucky_traj = []
            res.append(lucky_traj)

        return res
    
    def sample_pair_idxs(
        self, 
        traj_len: int,
        num_samples: int, 
        diff_traj: bool = False
    ) -> th.Tensor:
        '''
        Generate the paired frame indexes for the temporal rank head. 
        args:
            traj_len: the length of the input trajectory. 
            num_samples: the number of paired frames to be sampled. 
            diff_traj: whether to sample the paired frames from different trajectories. 
        '''
        lft_frames_idxs = []
        rgt_frames_idxs = []
        labels = []
        if diff_traj:
            options = list(range(self.interval, traj_len))
            for _ in range(num_samples):
                lft_frames_idxs += [random.choice(options)]
                rgt_frames_idxs += [random.choice(options)]
        else:
            acc_mark = self.acc_marks[traj_len]
            for _ in range(num_samples):
                lucky_pair = random.choice(self.all_pairs[:acc_mark])
                lft_frames_idxs += [lucky_pair[0]]
                rgt_frames_idxs += [lucky_pair[1]]
        
        return lft_frames_idxs, rgt_frames_idxs
    
    def compute_bilinear_ranking_logits(
        self, 
        lft_frame_feats: th.Tensor, 
        rgt_frame_feats: th.Tensor, 
        ce_latent: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        '''
        args:
            lft_frame_feats: (B, C), the features of the left frames.
            rgt_frame_feats: (B, C), the features of the right frames.
            ce_latent: (B, C) the features of the condition, can be None. 
        return:
            paired bilinear ranking logits: (B, 2). 
        '''
        if self.enable_condition:
            ce_latent = self.lz(ce_latent)
            diag_z = th.diag_embed(self.lz(ce_latent))
            # Use einsum to implement the matrix chain multiplication. 
            paired_ranking_logits = th.einsum(
                "Ba,Bab,bc,cd,Bde,Be->B", 
                lft_frame_feats, diag_z, self.A, self.B, diag_z, rgt_frame_feats
            )
        else: 
            paired_ranking_logits = th.einsum(
                "Ba,ab,bc,Bc->B", lft_frame_feats, self.A, self.B, rgt_frame_feats
            )
        paired_ranking_logits = paired_ranking_logits.unsqueeze(-1)
        # As the objective function is cross entropy loss, we need to pad the logits. 
        paired_ranking_logits = th.cat([
            paired_ranking_logits, th.zeros_like(paired_ranking_logits)
        ], dim=-1)
        
        return {
            'paired_ranking_logits': paired_ranking_logits,
        }
    
    def compute_scalar_ranking_logits(
        self, 
        lft_frame_feats: th.Tensor, 
        rgt_frame_feats: th.Tensor, 
        ce_latent: Optional[th.Tensor] = None, 
    ) -> th.Tensor:
        '''
        In scalar ranking method, we can compute logits for each frame. 
        args:
            lft_frame_feats: (B, C), the features of the left frames.
            rgt_frame_feats: (B, C), the features of the right frames.
            ce_latent: (B, C) the features of the condition, can be None. 
        return:
            paired bilinear ranking logits: (B, 2). 
            ranking_score: (B, ). 
        '''
        if self.enable_condition:
            ce_latent = self.lz(ce_latent)
            lft_logits = th.einsum("Ba,ab,Bb->B", lft_frame_feats, self.W, ce_latent).unsqueeze(-1)
            rgt_logits = th.einsum("Ba,ab,Bb->B", rgt_frame_feats, self.W, ce_latent).unsqueeze(-1)
        else:
            lft_logits = self.W(lft_frame_feats)
            rgt_logits = self.W(rgt_frame_feats)

        paired_ranking_logits = th.cat([lft_logits, rgt_logits], dim=-1) # B, 2
        return {
            'paired_ranking_logits': paired_ranking_logits, 
        }
    
    def rollout_ranking_logits(
        self,
        vi_latent: th.Tensor,
        ce_latent: Optional[th.Tensor] = None, 
    ) -> Dict:
        '''
        This is just used to visualize the ranking logit of each frame. 
        '''
        if self.enable_condition:
            ce_latent = self.lz(ce_latent)
            logits = th.einsum("BTa,ab,BTb->BT", vi_latent, self.W, ce_latent)
        else:
            logits = self.W(vi_latent).squeeze(-1)
        return logits
    
    def forward(self, latents, stage='train', **kwargs) -> Dict:
        '''
        Compute the temporal rank between sampled frames. 
        args:
            vi_latent: (B, T, C) / av_latent for auxiliary vision feature. 
                For example, it can be CLIP's vision feature. 
            ce_latent: (B, T, C)
        '''
        
        vi_latent = latents[self.vision_feature]
        ce_latent = latents['ce_latent']
        
        B, T, C = vi_latent.shape
        
        if self.enable_diff_traj:
            assert ce_latent is not None, "Condition is required when enable_diff_traj is True"
            neg_traj_idxs = self.sample_neg_traj_idxs(ce_latent)
        else:
            neg_traj_idxs = [[] for _ in range(B)]

        lucky_pairs = {'ce': [], 'lft': [], 'rgt': [], 'label': []}
        for i in range(B):
            # Sample the paired frames from the same trajectory. 
            for l, r in zip(*self.sample_pair_idxs(T, self.num_intra_samples, diff_traj=False)):
                lucky_pairs['ce'] += [(i, r)]
                lucky_pairs['lft'] += [(i, l)]
                lucky_pairs['rgt'] += [(i, r)]
                lucky_pairs['label'] += [l < r]
            # Sample the paired frames from different trajectories. 
            for neg_traj_idx in neg_traj_idxs[i]:
                for l, r in zip(*self.sample_pair_idxs(T, self.num_inter_samples, diff_traj=True)):
                    label = random.choice([True, False])
                    # print('neg:', neg_traj_idx, 'l,r: ', l, r, 'label:', label)
                    if label: 
                        lucky_pairs['ce'] += [(i, l)]
                        lucky_pairs['lft'] += [(i, l)]
                        lucky_pairs['rgt'] += [(neg_traj_idx, r)]
                    else:
                        lucky_pairs['ce'] += [(i, r)]
                        lucky_pairs['lft'] += [(neg_traj_idx, l)]
                        lucky_pairs['rgt'] += [(i, r)]
                    lucky_pairs['label'] += [label]
        
        lucky_pairs['ce'] = th.LongTensor(lucky_pairs['ce']).to(vi_latent.device)
        lucky_pairs['lft'] = th.LongTensor(lucky_pairs['lft']).to(vi_latent.device)
        lucky_pairs['rgt'] = th.LongTensor(lucky_pairs['rgt']).to(vi_latent.device)
        lucky_pairs['label'] = th.LongTensor(lucky_pairs['label']).to(vi_latent.device)

        lft_frame_feats = vi_latent[lucky_pairs['lft'][:, 0], lucky_pairs['lft'][:, 1], :]
        rgt_frame_feats = vi_latent[lucky_pairs['rgt'][:, 0], lucky_pairs['rgt'][:, 1], :]
        lucky_ce_latent = ce_latent[lucky_pairs['ce'][:, 0], lucky_pairs['ce'][:, 1], :]
        
        if self.compare_method == 'bilinear':
            ranking_info = self.compute_bilinear_ranking_logits(
                lft_frame_feats=lft_frame_feats, 
                rgt_frame_feats=rgt_frame_feats, 
                ce_latent=lucky_ce_latent, 
            )
        elif self.compare_method == 'scalar':
            ranking_info = self.compute_scalar_ranking_logits(
                lft_frame_feats=lft_frame_feats,
                rgt_frame_feats=rgt_frame_feats,
                ce_latent=lucky_ce_latent,
            )
            feat_logits = self.rollout_ranking_logits(
                vi_latent=vi_latent,
                ce_latent=ce_latent,
            )
            ranking_info['feat_logits'] = feat_logits
        
        ranking_info['pair_idxs'] = lucky_pairs
        
        return ranking_info
    
    def loss(self, obs, pred, mask=None, **kwargs):
        '''
        ranking_info (alais `pred`): Dict
            feat_logits: (B, T), Optional
            paired_ranking_logits: (B, 2)
            pair_idxs:
                lft: (B, 2)
                rgt: (B, 2)
                label: (B, )
        mask: (B, T) 
        '''
        ranking_info = pred
        # compute the ranking loss
        paired_ranking_logits = ranking_info['paired_ranking_logits']
        paired_ranking_gt = (
            ranking_info['pair_idxs']['label']
            .to(device=paired_ranking_logits.device)
            .long()
        ) 
        # compute binary classification loss
        if random.randint(0, 1000) % 100 == 0:
            if 'feat_logits' in ranking_info:
                print(ranking_info['feat_logits'])
        
        ranking_loss = self.criterion(
            paired_ranking_logits, paired_ranking_gt
        )
        
        ranking_acc = (
            (paired_ranking_logits.argmax(-1) == paired_ranking_gt)
            .float()
        ) 
        
        if mask is not None:
            lft = ranking_info['pair_idxs']['lft']
            rgt = ranking_info['pair_idxs']['rgt']
            rank_specific_mask = (
                mask[
                    th.stack([lft[:, 0], rgt[:, 0]], dim=1),
                    th.stack([lft[:, 1], rgt[:, 1]], dim=1)
                ]
                .all(dim=-1)
                .float()
            )
            ranking_loss = ranking_loss.reshape(-1)[rank_specific_mask.reshape(-1) > 0]
            ranking_acc = ranking_acc.reshape(-1)[rank_specific_mask.reshape(-1) > 0]
        
        ranking_loss = ranking_loss.mean()
        ranking_acc = ranking_acc.mean()
        # print(ranking_loss, ranking_acc)
        return {
            'loss_ranking': ranking_loss, 
            'acc_ranking': ranking_acc, 
        }

def make_temp_rank_head(**kwargs) -> nn.Module:
    return TempRankHead(**kwargs)

class CtrTempRankHead(BaseHead):
    pass

class CursorHead(BaseHead):
    
    def __init__(
        self, 
        hidsize: int, 
        nb_bins: Dict[str, int], 
        **kwargs, 
    ) -> None: 
        super().__init__()
        self.nb_bins = nb_bins
        self.hidsize = hidsize
        
        self.heads = nn.ModuleDict({
            'cursor_x': nn.Linear(hidsize, nb_bins['cursor_x']),
            'cursor_y': nn.Linear(hidsize, nb_bins['cursor_y']),
        })
    
    def forward(self, latents, **kwargs) -> Dict[str, th.Tensor]:
        feat = latents['tf_latent']
        return {k: self.heads[k](feat) for k in self.heads}
    
    def loss(self, targets: th.Tensor, logits: th.Tensor) -> th.Tensor:
        raise NotImplementedError
        return F.cross_entropy(logits.view(-1, self.nb_bins), targets.view(-1), reduction="none").view_as(targets)

def make_cursor_head(**kwargs) -> nn.Module:
    return CursorHead(**kwargs)

class HorizonHead(BaseHead):
    
    def __init__(
        self, 
        hidsize: int, 
        **kwargs, 
    ) -> None:
        super().__init__()
        self.hidsize = hidsize
        self.mlp = nn.Sequential(
            nn.Linear(hidsize, hidsize // 2),
            nn.ReLU(),
            nn.Linear(hidsize // 2, 1),
        )
    
    def forward(self, latents, **kwargs) -> th.Tensor:
        #predict the temporal distance in log space: return = log(1 + x)
        feat = latents['tf_latent']
        return self.mlp(feat).squeeze(-1)
    
    def loss(self, obs, pred, mask=None, **kwargs):
        # targets: B x T
        # logits: B x T
        logits = pred
        targets = obs['log_horizon']
        horizon_loss = F.mse_loss(logits, targets, reduction="none")
        if mask is not None:
            horizon_loss = horizon_loss.reshape(-1)[mask.reshape(-1) > 0]
        horizon_loss = horizon_loss.mean()
        return {
            'loss_horizon': horizon_loss,
        }

def make_horizon_head(**kwargs) -> nn.Module:
    return HorizonHead(**kwargs)

class SubgoalHead(BaseHead):
    
    def __init__(
        self, 
        hidsize: int, 
        hierarchical_flag: bool = False, 
        **kwargs, 
    ) -> None:
        super().__init__()
        self.hidsize = hidsize
        self.hierarchical_flag = hierarchical_flag
        self.mlp = nn.Sequential(
            nn.Linear(hidsize, hidsize // 2),
            nn.ReLU(),
            nn.Linear(hidsize // 2, hidsize // 4),
        )
        if self.hierarchical_flag:
            num_output = 3 * 3
            self.composed_head = nn.Linear(hidsize // 4, num_output)
        else:
            self.recipe_head = nn.Linear(hidsize // 4, 3)
            self.table_head = nn.Linear(hidsize // 4, 3)
            self.layout_heads = nn.ModuleList([
                nn.Linear(hidsize // 4, 3) for _ in range(9)
            ])
            
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, latents, **kwargs) -> Dict[str, th.Tensor]:
        feat = latents['tf_latent']
        feat = self.mlp(feat)
        result = {
            'recipe': self.recipe_head(feat), 
            'table': self.table_head(feat), 
            'layout': th.stack([
                head(feat) for i, head in enumerate(self.layout_heads)
            ], dim=2)
        }
        return result
    
    def loss(self, obs, pred, mask=None, **kwargs) -> th.Tensor:
        targets = obs['subgoal']
        logits = pred 
        subgoal_loss = 0
        recipe_loss = self.criterion(
            logits['recipe'].view(-1, 3) , targets['recipe'].view(-1), 
        ).view_as(targets['recipe'])
        table_loss = self.criterion(
            logits['table'].view(-1, 3), targets['table'].view(-1), 
        ).view_as(targets['table'])
        layout_loss = self.criterion(
            logits['layout'].view(-1, 3), targets['layout'].view(-1), 
        ).view_as(targets['layout']).mean(-1)
        subgoal_loss += recipe_loss + table_loss + layout_loss
        if mask is not None:
            subgoal_loss = subgoal_loss.reshape(-1)[mask.reshape(-1) > 0]
        subgoal_loss = subgoal_loss.mean()
        
        return {
            'loss_subgoal': subgoal_loss
        }

def make_subgoal_head(**kwargs) -> nn.Module:
    return SubgoalHead(**kwargs)

register_heads = {
    'temp_rank_head': make_temp_rank_head, 
    'cursor_head': make_cursor_head, 
    'horizon_head': make_horizon_head, 
    'subgoal_head': make_subgoal_head, 
}

def build_auxiliary_heads(auxiliary_head_kwargs, **parent_kwargs) -> Dict[str, nn.Module]:
    
    auxilary_heads_dict = {}
    
    for head, head_kwargs in auxiliary_head_kwargs.items():
        assert head in register_heads, \
            f"Unknown auxiliary head {head}, available: {register_heads.keys()}"
        if not head_kwargs['enable']:
            continue
        auxilary_heads_dict[head] = register_heads[head](**head_kwargs, **parent_kwargs)
    
    return auxilary_heads_dict


ACTION_KEY_DIM = OrderedDict({
    'forward': {'type': 'one-hot', 'dim': 2}, 
    'back': {'type': 'one-hot', 'dim': 2}, 
    'left': {'type': 'one-hot', 'dim': 2}, 
    'right': {'type': 'one-hot', 'dim': 2}, 
    'jump': {'type': 'one-hot', 'dim': 2}, 
    'sneak': {'type': 'one-hot', 'dim': 2}, 
    'sprint': {'type': 'one-hot', 'dim': 2}, 
    'attack': {'type': 'one-hot', 'dim': 2},
    'use': {'type': 'one-hot', 'dim': 2}, 
    'drop': {'type': 'one-hot', 'dim': 2},
    'inventory': {'type': 'one-hot', 'dim': 2}, 
    'camera': {'type': 'real', 'dim': 2}, 
    'hotbar.1': {'type': 'one-hot', 'dim': 2}, 
    'hotbar.2': {'type': 'one-hot', 'dim': 2}, 
    'hotbar.3': {'type': 'one-hot', 'dim': 2}, 
    'hotbar.4': {'type': 'one-hot', 'dim': 2}, 
    'hotbar.5': {'type': 'one-hot', 'dim': 2}, 
    'hotbar.6': {'type': 'one-hot', 'dim': 2}, 
    'hotbar.7': {'type': 'one-hot', 'dim': 2}, 
    'hotbar.8': {'type': 'one-hot', 'dim': 2}, 
    'hotbar.9': {'type': 'one-hot', 'dim': 2}, 
})

class ActionEmbedding(nn.Module):
    
    def __init__(
        self, 
        num_channels: int = 512,
        intermediate_dim: int = 64,
        action_type: Union['decomposed', 'composed'] = 'decomposed', 
        action_space: Optional[spaces.Space] = None, 
    ) -> None:
        super().__init__()
        self.action_type = action_type
        self.action_space = action_space
        if self.action_type == 'decomposed': 
            module_dict = dict()
            for key, conf in ACTION_KEY_DIM.items():
                key = 'act_' + key.replace('.', '_')
                if conf['type'] == 'one-hot':
                    module_dict[key] = nn.Embedding(conf['dim'], intermediate_dim)
                elif conf['type'] == 'real':
                    module_dict[key] = nn.Linear(conf['dim'], intermediate_dim)
            self.embedding_layer = nn.ModuleDict(module_dict)
            
        elif self.action_type == 'composed':
            module_dict = dict()
            for key, space in action_space.items():
                module_dict[key] = nn.Embedding(space.nvec.item(), num_channels)
            self.embedding_layer = nn.ModuleDict(module_dict)
        
        else:
            raise NotImplementedError
        self.final_layer = nn.Linear(len(self.embedding_layer) * intermediate_dim, num_channels)
    
    def forward_key_act(self, key: str, act: th.Tensor) -> th.Tensor:
        key_embedding_layer = self.embedding_layer['act_'+key.replace('.', '_')]
        if isinstance(key_embedding_layer, nn.Embedding):
            return key_embedding_layer(act.long())
        elif isinstance(key_embedding_layer, nn.Linear):
            return key_embedding_layer(act.float())
    
    def forward(self, action: Dict[str, th.Tensor]) -> th.Tensor:
        
        if self.action_type == 'decomposed':
            if len(action) != len(ACTION_KEY_DIM):
                # convert to decomposed action and launch to device
                npy_act = MinecraftWrapper.agent_action_to_env(action)
                device = next(self.parameters()).device
                action = {key: th.from_numpy(act).to(device) for key, act in npy_act.items()}
            return self.final_layer(th.cat([
                self.forward_key_act(key, action[key]) for key in ACTION_KEY_DIM.keys()
            ], dim=-1))
        elif self.action_type == 'composed':
            return self.final_layer(th.cat([
                self.forward_key_act(key, action[key]) for key in self.action_space.keys()
            ], dim=-1))

class TextEmbedding(nn.Module):
    
    def __init__(self, condition_dim: int = 512, **kwargs):
        super().__init__()
        # first load into cpu, then move to cuda by the lightning
        self.clip_model, _ = clip.load("ViT-B/32", device='cpu')
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.ffn = nn.Linear(512, condition_dim)
    
    @th.no_grad()
    def forward(self, texts, device="cuda", **kwargs):
        self.clip_model.eval()
        text_inputs = clip.tokenize(texts).to(device)
        embeddings = self.clip_model.encode_text(text_inputs)
        if hasattr(self, 'ffn'):
            embeddings = self.ffn(embeddings.to(self.ffn.weight.dtype))
        return embeddings

def layout_to_idx(layout: str) -> List[int]:
    mapping = {'o': 1, '#': 2}
    result = [mapping.get(c, 0) for c in layout]
    if len(result) < 9:
        result += [0] * (9 - len(result))
    return result

class SubGoalEmbedding(nn.Module): 
    
    def __init__(self, condition_dim: int = 512, **kwargs):
        super().__init__()
        self.text_embedding = TextEmbedding(condition_dim=condition_dim//2)
        self.recipe_embedding = nn.Embedding(3, 128)
        self.table_embedding = nn.Embedding(3, 128)
        self.resource_embedding = nn.ModuleList([
            nn.Embedding(3, 128) for _ in range(9)
        ]) # 0 for empty, 1 for place, 2 for mask
        self.final_layer = nn.Linear(condition_dim//2+2*128+9*128, condition_dim)
        self.time_step = 0
    
    def forward(self, texts, subgoal=None, _T=0, device="cuda", **kwargs):
        obs_text_embeddings = self.text_embedding(texts, device=device)
        try:
            _B, _T = subgoal['table'].shape[:2]
        except:
            _B = obs_text_embeddings.shape[0]
        obs_text_embeddings = obs_text_embeddings.unsqueeze(1).expand(-1, _T, -1)
        
        # if subgoals is None:
        #     print('_T: ', _T, 'time: ', self.time_step)
        #     subgoals = [
        #         # (
        #         #     [OrderedDict(recipe=0, table=0, layout="oooo") for _ in range(_T)] if self.time_step < 100 else \
        #         #     [OrderedDict(recipe=1, table=0, layout="oooo") for _ in range(_T)] if self.time_step < 200 else \
        #         #     [OrderedDict(recipe=1, table=0, layout="####") for _ in range(_T)]
        #         # )
        #         (
        #             [OrderedDict(recipe=0, table=0, layout="oooo") for _ in range(_T)] if self.time_step < 50 else \
        #             [OrderedDict(recipe=0, table=0, layout="#ooo") for _ in range(_T)] if self.time_step < 100 else \
        #             [OrderedDict(recipe=0, table=0, layout="#o#o") for _ in range(_T)] if self.time_step < 200 else \
        #             [OrderedDict(recipe=0, table=0, layout="#o##") for _ in range(_T)] if self.time_step < 300 else \
        #             [OrderedDict(recipe=0, table=0, layout="####") for _ in range(_T)] 
        #         )
        #             for _ in range(_B)
        #     ]
        #     self.time_step += 1
        
        # recipe_tensor = th.tensor(
        #     [ [ subgoals[i][j].get('recipe', -1) + 1  for j in range(_T) ] for i in range(_B)]
        # ).to(device)
        # table_tensor = th.tensor(
        #     [ [ subgoals[i][j].get('table', -1) + 1  for j in range(_T) ] for i in range(_B)]
        # ).to(device)
        # resource_tensor = th.tensor(
        #     [ [ layout_to_idx(subgoals[i][j].get('layout', []))  for j in range(_T) ] for i in range(_B)]
        # ).to(device)
        recipe_tensor = subgoal['recipe']
        table_tensor = subgoal['table']
        resource_tensor = subgoal['layout'] 
        
        # table_tensor = th.zeros_like(table_tensor)
        # recipe_tensor = th.ones_like(recipe_tensor)
        # resource_tensor[:, :, :4] = 1 #! debug!!!
        
        recipe_embedding = self.recipe_embedding(recipe_tensor)
        table_embedding = self.table_embedding(table_tensor)
        
        resource_embedding = th.cat([
            self.resource_embedding[i](resource_tensor[:,:,i]) for i in range(9)
        ], dim=2)
        result = th.cat([
            obs_text_embeddings, recipe_embedding, table_embedding, resource_embedding
        ], dim=-1)
        result = self.final_layer(result)
        return result


class TrajectoryEmbedding(nn.Module):
    
    def __init__(
        self,
        hidsize: int, 
        num_heads: int = 8,
        **kwargs
    ) -> None:
        super().__init__()
        self.sp_xattn = CrossAttentionLayer(input_size=hidsize, num_heads=num_heads)
        self.sp_slot = nn.Parameter(th.randn(1, hidsize) * 1e-3)
        self.tp_slot = nn.Parameter(th.randn(1, hidsize) * 1e-3)
        bert_config = GPT.get_default_config()
        bert_config.model_type = None
        bert_config.block_size = 256
        bert_config.n_layer = 8
        bert_config.n_head = num_heads
        bert_config.n_embd = hidsize
        self.tp_bert = GPT(bert_config)

    def forward(self, vision_feats, **kwargs):
        '''
        Encode trajectory into a fixed-length (num_slots) vector. 
        args:
            vision_feats: (B, T, C) or (B, T, C, W, W)
        '''
        B, T, C = vision_feats.shape[:3]
        if len(vision_feats.shape) == 5:
            W = vision_feats.shape[-1]
            flat_vision_feats = vision_feats.reshape(B*T, C, W*W).permute(0, 2, 1)
            sp_slot_embedding = self.sp_slot.unsqueeze(0).expand(B*T, -1, -1)
            x = self.sp_xattn(sp_slot_embedding, flat_vision_feats)
            x = x.reshape(B, T, C)
        
        tp_slot_embedding = self.tp_slot.unsqueeze(0).expand(B, -1, -1)
        input = th.cat([tp_slot_embedding, x], dim=1)
        output = self.tp_bert(input)[:, :1, :]
        return output


def build_condition_embedding_layer(
    name: Optional[str] = None, **kwargs, 
) -> Union[nn.Module, None]:
    if name == 'text_embedding':
        return TextEmbedding(**kwargs)
    elif name == 'subgoal_embedding':
        return SubGoalEmbedding(**kwargs)
    elif name == 'trajectory_embedding':
        return TrajectoryEmbedding(**kwargs)
    else:
        return None

class ImgPreprocessing(nn.Module):
    """Normalize incoming images.

    :param img_statistics: remote path to npz file with a mean and std image. If specified
        normalize images using this.
    :param scale_img: If true and img_statistics not specified, scale incoming images by 1/255.
    """

    def __init__(self, img_statistics: Optional[str] = None, scale_img: bool = True):
        super().__init__()
        self.img_mean = None
        if img_statistics is not None:
            img_statistics = dict(**np.load(img_statistics))
            self.img_mean = nn.Parameter(th.Tensor(img_statistics["mean"]), requires_grad=False)
            self.img_std = nn.Parameter(th.Tensor(img_statistics["std"]), requires_grad=False)
        else:
            self.ob_scale = 255.0 if scale_img else 1.0

    def forward(self, img):
        # x = img.to(dtype=th.float32)
        x = img
        if self.img_mean is not None:
            x = (x - self.img_mean) / self.img_std
        else:
            x = x / self.ob_scale
        if x.dim() == 4:
            x = x.unsqueeze(1)
        return x


class ImgObsProcess(nn.Module):
    """ImpalaCNN followed by a linear layer.

    :param cnn_outsize: impala output dimension
    :param output_size: output size of the linear layer.
    :param dense_init_norm_kwargs: kwargs for linear FanInInitReLULayer
    :param init_norm_kwargs: kwargs for 2d and 3d conv FanInInitReLULayer
    """

    def __init__(
        self,
        cnn_outsize: int,
        output_size: int,
        dense_init_norm_kwargs: Dict = {},
        init_norm_kwargs: Dict = {},
        **kwargs,
    ):
        super().__init__()
        self.cnn = ImpalaCNN(
            outsize=cnn_outsize,
            init_norm_kwargs=init_norm_kwargs,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            **kwargs,
        )
        self.linear = FanInInitReLULayer(
            cnn_outsize,
            output_size,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )

    def forward(self, img, cond=None, **kwargs):
        return self.linear(self.cnn(img, cond=cond))


def general_preprocessor(
    image_tensor: th.Tensor, 
    imsize:int = 224, 
    scale_input:float = 255.0, 
    normalize:bool = True, 
    channel_last:bool = False,
    **kwargs,
) -> th.Tensor:

    if image_tensor.dim() == 4:
        image_tensor = image_tensor.unsqueeze(1)
    
    # shape is (B, T, C, H, W) or (B, T, H, W, C)
    if image_tensor.shape[-1] == 3:
        image_tensor = image_tensor.permute(0, 1, 4, 2, 3).contiguous()
    # shape is (B, T, C, H, W)
    
    transform_list = [
        T.Resize((imsize, imsize)),
        T.Lambda(lambda x: x / scale_input), 
    ]
    
    if normalize:
        transform_list.append(T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    
    transform = T.Compose(transform_list)
    
    if image_tensor.dim() == 5:
        processed_images = th.stack([transform(image_tensor[:, t]) for t in range(image_tensor.size(1))], dim=1)
    else:
        processed_images = transform(image_tensor)
    
    if channel_last:
        processed_images = processed_images.permute(0, 1, 3, 4, 2).contiguous()
    
    return processed_images


class SpatialSoftmax(nn.Module):
    
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(th.ones(1)*temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width)
                )
        pos_x = th.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = th.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
        else:
            feature = feature.view(-1, self.height*self.width)

        softmax_attention = F.softmax(feature/self.temperature, dim=-1) # NxCxHW
        expected_x = th.sum(self.pos_x*softmax_attention, dim=1, keepdim=True) # NCxHW
        expected_y = th.sum(self.pos_y*softmax_attention, dim=1, keepdim=True) # NCxHW
        expected_xy = th.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel*2)

        return feature_keypoints

class CustomEfficientNet(nn.Module):
    
    def __init__(
        self, 
        version: str, 
        resolution: int = 224, 
        out_dim: int = 1024, 
        pooling: bool = False, 
        atten: str = 'default',
        **kwargs, 
    ) -> None:
        super().__init__()
        self.version = version
        self.resoulution = resolution
        self.out_dim = out_dim
        
        self.model = EfficientNet.from_pretrained(version)
        
        if 'b0' in version:
            self.mid_dim = 1280
        elif 'b4' in version:
            self.mid_dim = 1792
        
        if resolution == 360:
            self.feat_reso = (11, 11)
        elif resolution == 224:
            self.feat_reso = (7, 7)
        elif resolution == 128:
            self.feat_reso = (4, 4)

        self.final_layer = nn.Conv2d(self.mid_dim, out_dim, 1)
        if atten == 'default':
            print('>>> --- Factory.py[CustomEfficientNet]: atten default: simple')
            self.atten = 'simple'
        else:
            self.atten = atten
        # assert not (pooling and cross_attention), "pooling and cross_attention cannot be True at the same time"
        self.pooling_layer = nn.AdaptiveMaxPool2d(1)
        
        if pooling:
            # self.pooling_layer = nn.AdaptiveMaxPool2d(1)
            self.atten = None
            print('>>> --- Factory.py[CustomEfficientNet]: init with pooling')
            pass
        else: 
            if self.atten == 'xatten':
                print('>>> --- Factory.py[CustomEfficientNet]: init with xatten')
                self.xattn_layer = CrossAttentionLayer(out_dim, 4)
                self.cond_layer = nn.Linear(512, out_dim)
            elif self.atten == 'simple':
                print('>>> --- Factory.py[CustomEfficientNet]: init with simple atten')
                self.simple_atten_layer = SimpleCrossAttention(out_dim, 512, 4)
            else:
                print('>>> --- Factory.py[CustomEfficientNet]: error init')
            self.last_layer = nn.Linear(2*out_dim, out_dim)

    def cond_pool(self, imgs, cond=None):
        B, T = imgs.shape[:2]
        x = imgs.reshape(B * T, *imgs.shape[2:])
        x = self.model.extract_features(x, cond=cond)
        x = self.final_layer(x)

        x_pool = self.pooling_layer(x).squeeze(-1).squeeze(-1)
        x_pool = x_pool.reshape((B, T) + x_pool.shape[1:])
        return x_pool
    
    def cross_atten(self, imgs, cond=None):
        assert cond is not None, f'Factory.py[CustomEfficientNet]: condition should not be none in cross attention'
        B, T = imgs.shape[:2]
        x = imgs.reshape(B * T, *imgs.shape[2:])
        x = self.model.extract_features(x)
        x = self.final_layer(x)  # [B*T, C, H, W]

        x_pool = self.pooling_layer(x).squeeze(-1).squeeze(-1)
        x_pool = x_pool.reshape((B, T) + x_pool.shape[1:])

        flatten_feat = x.view((B, T, self.out_dim, np.prod(self.feat_reso)))
        flatten_feat = flatten_feat.permute(0, 1, 3, 2).contiguous()
        x_query, attn = self.simple_atten_layer(flatten_feat, cond)
        # attn = attn.reshape(attn.shape[:-1] + self.feat_reso)  # attn feat <-- [B, T, 1, H, W]

        x_cat = th.cat((x_pool, x_query), dim=-1)
        feat = self.last_layer(x_cat)
        return feat
    
    def xatten(self, imgs, cond=None):
        assert cond is not None, "<cond> embedding cannot be None when cross_attention is True"
        B, T = imgs.shape[:2]
        x = imgs.reshape(B * T, *imgs.shape[2:])
        
        x = self.model.extract_features(x)
        x = self.final_layer(x)
        
        x_pool = self.pooling_layer(x).squeeze(-1).squeeze(-1)
        x_pool = x_pool.reshape((B, T) + x_pool.shape[1:])
              
        flatten_feat = x.view(B*T, self.out_dim, np.prod(self.feat_reso))
        flatten_feat = flatten_feat.permute(0, 2, 1).contiguous()
        cond_feat = self.cond_layer(cond).view(B*T, 1, self.out_dim)
        x_query = self.xattn_layer(cond_feat, flatten_feat).view(B, T, self.out_dim)
        
        x_cat = th.cat((x_pool, x_query), dim=-1)
        feat = self.last_layer(x_cat)
        return feat
        
    def forward(self, imgs, cond=None, **kwargs): 
        
        if self.atten is not None:
            if self.atten == 'simple':
                return self.cross_atten(imgs, cond=cond)
            elif self.atten == 'xatten':
                return self.xatten(imgs, cond=cond)
            else:
                raise f'Factory.py[CustomEfficientNet]: unknown atten type {self.atten}.'
        else:
            return self.cond_pool(imgs, cond=cond)

# <- bingo
class CursorImgEfficientNet(nn.Module):
    
    def __init__(
        self, 
        version: str, 
        resolution: int = 224, 
        out_dim: int = 1280, 
        pooling: bool = False, 
        # cross_attention: bool = False,
        **kwargs, 
    ) -> None:
        super().__init__()
        self.version = version
        self.resoulution = resolution
        self.out_dim = out_dim
        
        self.model = EfficientNet.from_pretrained(
            version, image_size=(resolution, resolution))
        # self.model = EfficientNet.from_name(version)
        
        if 'b0' in version:
            self.mid_dim = 1280
        elif 'b4' in version:
            self.mid_dim = 1792

        self.final_layer = nn.Conv2d(self.mid_dim, out_dim, 1)
        
        # assert not (pooling and cross_attention), "pooling and cross_attention cannot be True at the same time"
        if pooling:
            self.pooling_layer = nn.AdaptiveMaxPool2d(1)
        
    def forward(self, imgs, cond=None, **kwargs): 
        B, T = imgs.shape[:2]
        x = imgs.reshape(B * T, *imgs.shape[2:])
        
        if not hasattr(self, 'xattn_layer'):
            x = self.model.extract_features(x, cond=cond)
        else:
            x = self.model.extract_features(x)
            
        x = self.final_layer(x)
        
        x = x.reshape((B, T) + x.shape[1:])
        
        if hasattr(self, 'pooling_layer'):
            x = self.pooling_layer(x).squeeze(-1).squeeze(-1)
        
        return x


class CustomResNet(nn.Module):
    
    def __init__(self, version: str = '50', out_dim: int = 1024, **kwargs):
        super().__init__()
        if version == '18':
            self.model = torchvision.models.resnet18(pretrained=True)
        elif version == '50':
            self.model = torchvision.models.resnet50(pretrained=True)
        elif version == '101':
            self.model = torchvision.models.resnet101(pretrained=True)
        
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.final_layer = nn.Linear(2048, out_dim)
    
    def forward(self, imgs, **kwargs):
        B, T = imgs.shape[:2]
        x = imgs.reshape(B * T, *imgs.shape[2:])
        x = self.model(x)
        x = x.view(B * T, -1)
        x = self.final_layer(x)
        return x.reshape(B, T, -1)


class CustomCLIPv(nn.Module):
    
    def __init__(self, version: str = "ViT-B/32", out_dim: int = 1024, **kwargs):
        super().__init__()
        # first load into cpu, then move to cuda by the lightning
        clip_model, preprocess = clip.load(version, device='cpu')
        self.preprocess = preprocess
        self.vision_encoder = clip_model.visual
        self.vision_encoder.eval()
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        self.final_layer = nn.Linear(512, out_dim)
    
    @th.no_grad()
    def forward(self, imgs, **kwargs):
        B, T = imgs.shape[:2]
        x = imgs.reshape(B * T, *imgs.shape[2:])
        x = self.vision_encoder(x)
        x = self.final_layer(x)
        return x.reshape(B, T, -1)


def build_backbone(name: str = 'IMPALA', **kwargs) -> Dict:
    
    result_modules = {}
    if name == 'IMPALA':
        first_conv_norm = False
        impala_kwargs = kwargs.get('impala_kwargs', {})
        init_norm_kwargs = kwargs.get('init_norm_kwargs', {})
        dense_init_norm_kwargs = kwargs.get('dense_init_norm_kwargs', {})
        result_modules['preprocessing'] = partial(
            general_preprocessor, 
            imsize=128, 
            scale_input=255.0, 
            normalize=False, 
            channel_last=True,
            **kwargs,
        )
        result_modules['obsprocessing'] = ImgObsProcess(
            cnn_outsize=256,
            output_size=kwargs['hidsize'],
            inshape=kwargs['img_shape'],
            chans=tuple(int(kwargs['impala_width'] * c) for c in kwargs['impala_chans']),
            nblock=2,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            init_norm_kwargs=init_norm_kwargs,
            first_conv_norm=first_conv_norm,
            require_goal_embedding=kwargs.get('require_goal_embedding', False), 
            **impala_kwargs, 
        )
        
    elif name == 'CLIPv':
        model = CustomCLIPv(
            out_dim=kwargs['hidsize'],
            **kwargs,
        )
        result_modules['preprocessing'] = partial(
            general_preprocessor, 
            imsize=224, 
            scale_input=255.0, 
            normalize=True, 
            channel_last=False
        )
        result_modules['obsprocessing'] = model

    elif name == 'SAM':
        pass
    
    elif name == 'EFFICIENTNET':
        model = CustomEfficientNet(
            out_dim=kwargs['hidsize'],
            **kwargs, 
        )
        result_modules['preprocessing'] = partial(
            general_preprocessor, 
            imsize=kwargs['resolution'], 
            scale_input=255.0, 
            normalize=True, 
            channel_last=False
        )
        result_modules['obsprocessing'] = model
        
    elif name == 'RESNET':
        result_modules['preprocessing'] = partial(
            general_preprocessor, 
            imsize=224, 
            scale_input=255.0, 
            normalize=True, 
            channel_last=False, 
            out_dim=kwargs['hidsize']
        )
        result_modules['obsprocessing'] = CustomResNet(
            version=kwargs['version'], 
            out_dim=kwargs['hidsize'],
            **kwargs, 
        )
    return result_modules

class BaseConditioningLayer(nn.Module):
    
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, vision_feats: th.Tensor, cond_feats: th.Tensor, **kwargs):
        raise NotImplementedError
    

class SpXattnCondLayer(BaseConditioningLayer):
    
    def __init__(
        self, 
        hidsize: int, 
        num_heads: int = 4, 
        residual_condition: bool = False,
        **kwargs, 
    ) -> None:
        
        super().__init__(**kwargs)
        self.xattn_layer = CrossAttentionLayer(hidsize, num_heads=num_heads)
        self.residual_condition = residual_condition

    def forward(self, vision_feats: th.Tensor, cond_feats: th.Tensor, **kwargs):
        '''
        Inject the condition information into vision feats via cross-attention. 
        args: 
            vision_feats: (B, T, C, W, W)
            cond_feats: (B, T, C) or (B, C)
        return:
            output_feats: (B, T, C)
        '''
        assert len(vision_feats.shape) == 5, \
            "SpatialCrossAttentionConditioningLayer requires spatial dimension."

        B, T, C, W, _ = vision_feats.shape
        if len(cond_feats.shape) == 2:
            cond_feats = cond_feats.unsqueeze(1)
        
        if cond_feats.shape[1] == 1:
            cond_feats = cond_feats.repeat(1, T, 1)

        if cond_feats.shape[1] != T:
            padding = (
                th.zeros(B, T-cond_feats.shape[1], C)
                .to(device=cond_feats.device, dtype=cond_feats.dtype)
            )
            cond_feats = th.cat([padding, cond_feats], dim=1)

        flat_vision_feats = vision_feats.reshape(B*T, C, -1).permute(0, 2, 1)
        flat_cond_feats = cond_feats.reshape(B*T, C).unsqueeze(1)
        output_feats = self.xattn_layer(flat_cond_feats, flat_vision_feats)
        output_feats = output_feats.reshape(B, T, C)
        
        if self.residual_condition:
            output_feats = output_feats + cond_feats
            
        return output_feats


class AddCondLayer(BaseConditioningLayer):
    
    def forward(self, vision_feats: th.Tensor, cond_feats: th.Tensor, **kwargs):
        '''
        Inject the condition information into vision feats via cross-attention. 
        args: 
            vision_feats: (B, T, C)
            cond_feats: (B, T, C) or (B, C)
        return:
            output_feats: (B, T, C)
        '''
        assert len(vision_feats.shape) == 3, \
            "AddConditioningLayer requires no additional dimension than B, T, C."

        B, T, C = vision_feats.shape
        
        if len(cond_feats.shape) == 2:
            cond_feats = cond_feats.unsqueeze(1).repeat(1, vision_feats.shape[1], 1)
        
        output_feats = vision_feats + cond_feats
        
        return output_feats


def build_conditioning_fusion_layer(
    name: Optional[str] = None, **kwargs
):
    if name == 'spatial_xattn':
        return SpXattnCondLayer(**kwargs)
    elif name == 'add':
        return AddCondLayer(**kwargs)
    else:
        raise None

if __name__ == '__main__':
    print("enable debug mode")
    input_tensor = th.randn(1, 3, 224, 224)
    features = model(input_tensor)
    print(features.shape) 
