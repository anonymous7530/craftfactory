import os
import io
import av
import cv2
import yaml 
import hydra
from rich.console import Console
import numpy as np
import torch
import logging
from copy import deepcopy
from pprint import pprint
from pathlib import Path
from typing import Dict, List, Union, Sequence, Mapping, Any, Optional
from omegaconf import DictConfig, OmegaConf
from aligo import Aligo 

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.algorithms.pg.pg_torch_policy import PGTorchPolicy
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.policy.torch_policy_template import build_torch_policy

import jarvis
from jarvis.arm.src.minecraft_agent import (
    AgentModule, 
    HierarchicalActionDist, 
)
from jarvis.stark_tech.ray_bridge import MinecraftWrapper
from jarvis.gallary.dataset.manager import draw_subgoal_layout
from jarvis.gallary.utils.rollout import Recorder
from jarvis import JARVISBASE_TMP

def to_dict(kwargs: DictConfig):
    result = dict()
    for k, v in kwargs.items():
        if type(v) == DictConfig:
            result[k] = to_dict(v)
        else:
            result[k] = v
    return result

class RayEvalPolicy(PGTorchPolicy):
    
    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        res = super().extra_action_out(input_dict, state_batches, model, action_dist)
        if 'subgoal' in model.extra_out:
            res['subgoal'] = model.extra_out['subgoal']
        if 'horizon' in model.extra_out:
            res['horizon'] = model.extra_out['horizon']
        if 'ranking_score' in model.extra_out:
            res['ranking_score'] = model.extra_out['ranking_score']
        return res
        

class Evaluator(object):

    def __init__(
        self, 
        env: Union[str, Dict, DictConfig], 
        model: Dict, 
        num_gpus: int = 1,
        num_workers: int = 1, 
        num_envs_per_worker: int = 1,
        num_cpus_per_worker: int = 1,
        num_gpus_per_worker: float = 1.0,
        remote_worker_envs: bool = False,
        remote_env_batch_wait_ms: int = 0,
        rollout_fragment_length: int = 100,
        batch_mode: str = 'complete_episodes',
        num_episodes_per_worker: int = 1,
        checkpoint_path: str = None,
        # below are arguments for generating videos
        aliyun: Dict = {},
        # below are for collecting data
        root: str = 'trajectories', 
        height: int = 360, 
        width: int = 640,
        fps: int = 20,
        **kwargs, 
    ) -> None:
        
        self.aliyun = aliyun
        self.process_other_arguments()
        
        self.num_episodes_per_worker = num_episodes_per_worker
        self.register_minecraft_models_and_dists()
        
        self.num_workers = num_workers
        self.worker_set = WorkerSet(
            env_creator=lambda _: MinecraftWrapper(env),
            default_policy_class=RayEvalPolicy,
            config=dict(
                model=model, 
                num_envs_per_worker=num_envs_per_worker,
                num_cpus_per_worker=num_cpus_per_worker,
                num_gpus_per_worker=num_gpus_per_worker,
                rollout_fragment_length=rollout_fragment_length, 
                remote_worker_envs=remote_worker_envs,
                remote_env_batch_wait_ms=remote_env_batch_wait_ms,
                num_gpus=num_gpus,
                batch_mode=batch_mode,
                sample_async=True,
                _disable_initialize_loss_from_dummy_batch=True,
                recreate_failed_workers=True,
                restart_failed_sub_environments=True,
                num_consecutive_worker_failures_tolerance=10,
            ),
            num_workers=num_workers,
            local_worker=False,
        )
        # self.actors = []
        # for worker in self.worker_set.remote_workers():
        #     self.actors.append(worker)
        self.height = height
        self.width = width
        self.recorder = Recorder(
            root=root, 
            height=height,
            width=width,
            fps=fps,
        )
        
        
    def process_other_arguments(self):
        if self.aliyun.get('enable', False):
            self.ali = Aligo(self.aliyun['name'], level=logging.ERROR)
            info = self.ali.get_personal_info()
            total_size = info.personal_space_info.total_size
            used_size = info.personal_space_info.used_size
            Console().log(f'Aligo free space left: {(1-used_size/total_size)*100:.2f}%')
            folder_list = self.ali.get_file_list()
            Console().log(f'Aligo folder_list: {folder_list}')
            self.remote_folder = self.ali.get_folder_by_path(self.aliyun['remote_folder'])
            

    def register_minecraft_models_and_dists(self):
        ModelCatalog.register_custom_model("minecraft_agent", AgentModule)
        ModelCatalog.register_custom_action_dist("hierarchical_action_dist", HierarchicalActionDist)

    def close(self):
        self.worker_set.stop()
    
    def stat_monitor(curr_obs: Dict, prev_obs: Dict) -> Dict:
        '''
        Monitor the change of stats (inventory, pickup, break_item, craft_item, mine_block, kill_entity)
        '''
        stat_names = ['pickup', 'break_item', 'craft_item', 'mine_block', 'kill_entity']
        results = {k:[] for k in stat_names}
        for stat in stat_names:
            for meta in curr_obs[stat]:
                delta = curr_obs[stat].get(meta, 0) - prev_obs[stat].get(meta, 0)
                if delta > 0:
                    results[stat].append((meta, delta))
        return results

    @ray.remote(num_cpus=10, num_gpus=0)
    def _save_episode(eps_id: int, episode: Dict[str, List]):
        print(f"eps: {eps_id}, len: {len(episode['infos'])}")

        infos = episode['infos']
        actions = episode['actions']
        rewards = episode['rewards']
        
        video_name = f"video_{eps_id}.mp4"
        outStream = io.BytesIO()
        container = av.open(outStream, mode='w', format='mp4')
        stream = container.add_stream('h264', rate=20)
        stream.width = 640 
        stream.height = 360
        stream.pix_fmt = 'yuv420p'
        
        accumulated_accomp = {}
        accumulated_reward = 0.
        message_time = -10000

        for frame_id, info in enumerate(infos):
            try:
                frame = info['pov'].copy()
            except:
                frame = info[0]['agent0']['pov'].copy()
            # <- bingo >> visualize img_cursor
            index = 10
            if 'index' in info:
                index = info['index']
            if 'img_cursor' in info:
                img_cursor = info['img_cursor']
                box = info['box']
            
            reward = rewards[frame_id]
            if 'text' in info: 
                text_condition = info['text']
            else:
                text_condition = 'none' 
            accumulated_reward += reward

            cv2.putText(frame, f"Task: {text_condition}", (150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (234, 53, 70), 2)
            cv2.putText(frame, f"Step: {frame_id}, Reward: {accumulated_reward}, P: {index}", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (248, 102, 36), 2)
            
            if 'horizon' in episode:
                log_horizon = episode['horizon'][frame_id]
                ori_horizon = np.exp(log_horizon) - 1
                cv2.putText(frame, f"Log-horizon: {log_horizon:.1f}, Ori-horizon: {ori_horizon:.1f}", (150, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (248, 102, 36), 2)
            
            elif 'ranking_score' in episode:
                ranking_score = episode['ranking_score'][frame_id]
                cv2.putText(frame, f"Ranking score: {ranking_score[0]:.1f}", (150, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (248, 102, 36), 2)
            
            # Draw action
            action = (actions[frame_id][0], actions[frame_id][1])
            minerl_action = MinecraftWrapper.agent_action_to_env(action)

            for row, (k, v) in enumerate(minerl_action.items()):
                color = (234, 53, 70) if (v != 0).any() else (249, 200, 14) 
                if k == 'camera':
                    v = "[{:.2f}, {:.2f}]".format(v[0], v[1])
                cv2.putText(frame, f"{k}: {v}", (10, 25 + row*15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw stats
            if frame_id > 1:
                stat_change = Evaluator.stat_monitor(curr_obs=info, prev_obs=prev_info)
                for stat in stat_change:
                    for meta, val in stat_change[stat]:
                        message = f"{frame_id} {stat}: {meta} x {int(val)}"
                        message_time = frame_id
                        accumulated_accomp[f"{stat}:{meta}"] = accumulated_accomp.get(f"{stat}:{meta}", 0) + val

            if frame_id - message_time <= 20:
                cv2.putText(frame, message, (150, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (67, 188, 205), 2)
            
            if 'subgoal' in episode:
                draw_subgoal_layout(frame, episode['subgoal'], tidx=frame_id)
            
            frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet) 
            
            prev_info = info

        for packet in stream.encode():
            container.mux(packet)
        container.close()
        bytes = outStream.getvalue()
        return {
            'episode_id': eps_id,
            'video_name': video_name,
            'video_bytes': bytes,
        }


    def save_video(self, episodes: List):
        Console().log('Saving videos to disk...')
        futures = []
        for eps_id, episode in enumerate(episodes):
            futures += [Evaluator._save_episode.remote(eps_id=eps_id, episode=episode)]
        
        while len(futures) > 0:
            done_id, futures = ray.wait(futures, num_returns=1)
            done_id = done_id[0]
            result = ray.get(done_id)
            video_name = result['video_name']
            video_bytes = result['video_bytes']
            with open(video_name, 'wb') as f:
                f.write(video_bytes)
            Console().log(f"Saved video {video_name} to disk.")
    
    def compute_reward(self, episodes):
        rewards = []
        for eps_id, episode in enumerate(episodes):
            rewards.append(sum(episode['rewards']))
        return rewards

    def sample(self):
        episodes = {}
        
        remain = self.num_episodes_per_worker * self.num_workers
        
        for i in range(1, self.num_workers + 1):
            self.worker_set.foreach_worker_async(lambda w: w.sample(), remote_worker_ids=[i])
        
        while remain > 0:
            result = self.worker_set.fetch_ready_async_reqs()
            if result is None or len(result) == 0:
                continue
            
            for i in range(1, self.num_workers + 1):
                self.worker_set.foreach_worker_async(lambda w: w.sample(), remote_worker_ids=[i])
            
            for worker_id, com_episode in result:
                self.worker_set.foreach_worker_async(lambda w: w.sample(), remote_worker_ids=[worker_id])
                remain -= 1
                print('remain: ', remain)
                
                truncated_eps = com_episode['default_policy'].split_by_episode()
                for eps in truncated_eps:
                    eps_id = eps['eps_id'][0]
                    if eps_id not in episodes:
                        episodes[eps_id] = {
                            'actions': [],
                            'rewards': [],
                            'infos': [],
                            'terminateds': [],
                        }
                        if 'subgoal' in eps:
                            episodes[eps_id]['subgoal'] = {'recipe': [], 'table': [], 'layout': []}
                        if 'horizon' in eps:
                            episodes[eps_id]['horizon'] = []
                        if 'ranking_score' in eps:
                            episodes[eps_id]['ranking_score'] = []
                    
                    episodes[eps_id]['actions'] += list(deepcopy(eps['actions']))
                    episodes[eps_id]['rewards'] += list(deepcopy(eps['rewards']))
                    episodes[eps_id]['infos'] += list(deepcopy(eps['infos']))
                    episodes[eps_id]['terminateds'] += list(deepcopy(eps['terminateds']))
                    if 'subgoal' in eps:
                        episodes[eps_id]['subgoal']['recipe'] += list(deepcopy(eps['subgoal']['recipe']))
                        episodes[eps_id]['subgoal']['table'] += list(deepcopy(eps['subgoal']['table']))
                        episodes[eps_id]['subgoal']['layout'] += list(deepcopy(eps['subgoal']['layout']))
                    if 'horizon' in eps:
                        episodes[eps_id]['horizon'] += list(deepcopy(eps['horizon']))
                    if 'ranking_score' in eps:
                        episodes[eps_id]['ranking_score'] += list(deepcopy(eps['ranking_score']))
            
            removed_eps = []
            for eps_id, eps_dict in episodes.items():
                # print('eps_id: ', eps_id, len(eps_dict['terminateds']))
                if any(eps_dict['terminateds']):
                    # print("Finished episode: ", eps_id)
                    yield eps_dict
                    removed_eps.append(eps_id)
            
            for eps_id in removed_eps:
                del episodes[eps_id]
            
    
    def benchmark(self, save_video=True):
        episodes = [e for e in self.sample()]
        if save_video:
            self.save_video(episodes)
        rewards = self.compute_reward(episodes)
        metrics = {
            'rewards': rewards, 
            'mean_reward': np.mean(rewards),
        }
        return metrics
    
    
    def collect_data(self):
        
        for eps in self.sample():
            
            # info['pov'] for info in eps['infos']
            frames = []
            for idx, info in enumerate(eps['infos']):
                try:
                    frame = info['pov']
                except:
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    Console().log('arning: {idx} frame, no pov in info.')
                frames.append(frame)
            
            infos = [] # infos[1].keys()
            for info in eps['infos']:
                info = {k: v for k, v in info.items() if k != 'pov'}
                infos.append(info)
            # import ray; ray.util.pdb.set_trace()
            actions = []
            for action in eps['actions']:
                action = (action[0], action[1])
                minerl_action = MinecraftWrapper.agent_action_to_env(action)
                actions.append(minerl_action)
            
            Console().log('Saving one trajectory to disk.')
            self.recorder.save_trajectory(
                video=frames,
                actions=actions,
                infos=infos,
            )
    

@hydra.main(config_path="configs", config_name="evaluation")
def main(cfg):
    ray.init(_temp_dir=f'{JARVISBASE_TMP}/ray')
    evaluator = Evaluator(
        **to_dict(cfg.evaluation), 
        **to_dict(cfg.others), 
        **to_dict(cfg.collect), 
    )
    
    if cfg.mode == 'benchmark':
        metrics = evaluator.benchmark(save_video=True)  # True
        print(metrics)
    elif cfg.mode == 'collect':
        evaluator.collect_data()
    
    evaluator.close()

if __name__ == "__main__":
    print("#######################################init inventory######################################")
    main()
