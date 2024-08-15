import hydra
from hydra import compose, initialize
from pathlib import Path
from typing import (
    Dict, List, Tuple, Union, Sequence, Mapping, Any, Optional
)

from omegaconf import DictConfig, OmegaConf
from collections import OrderedDict

import av
import cv2
import os
import torch
import yaml
import numpy as np
import random
import sys
import pdb

from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.pg.pg_torch_policy import PGTorchPolicy
from ray.rllib.algorithms.pg import PGConfig, PG
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

import jarvis
from jarvis.arm.src.minecraft_agent import (
    AgentModule, 
    HierarchicalActionDist, 
)

from jarvis import JARVISBASE_ROOT, JARVISBASE_PRETRAINED, JARVISBASE_TMP
from jarvis.gallary.craft_script import CraftScript, SmeltScript

from jarvis.stark_tech.ray_bridge import MinecraftWrapper
from jarvis.arm.src.minecraft_agent import AgentModule, to_dict
from jarvis.gallary.craft_script.collect.collect_data.collect_crafting import (
    RECIPES_INGREDIENTS
)
from jarvis.assembly import version

import json
item_name_list = json.load(open(f'{os.getenv("JARVISBASE_ROOT")}/jarvis/assets/all_items/all_items.json'))
item_name_list.append('none')

ModelCatalog.register_custom_model("minecraft_agent", AgentModule)
ModelCatalog.register_custom_action_dist("hierarchical_action_dist", HierarchicalActionDist)
register_env("dummy_env", lambda x: MinecraftWrapper('test'))


def slot_mapping(id: int):
    if id < 36:
        return 'inventory', id
    elif id < 45:
        return 'container_slots', id - 36 + 1
    else:
        return 'container_slots', 0

class MarkBase:
    
    def __init__(self, **kwargs):
        self.colormap = cv2.COLORMAP_VIRIDIS
        pass
        
    def reset(self):
        raise NotImplementedError
    
    def do(self):
        raise NotImplementedError
    
    def record_step(self):
        record_frames = getattr(self, 'record_frames', [])
        record_infos = getattr(self, 'record_infos', [])
        self.record_frames = record_frames + [self.info['pov']]
        if 'cursor_map' in self.info:
            cursor_map = self.info['cursor_map']
            cursor_map = (cursor_map - cursor_map.min()) / \
                (cursor_map.max() - cursor_map.min()) * 255
            cursor_map = cursor_map.astype(np.uint8)
            cursor_map = cv2.resize(cursor_map, (640, 360))
            cursor_map = cv2.applyColorMap(cursor_map, self.colormap)
            self.record_infos = record_infos + [cursor_map]

    def caption_frame(self,  **kwargs):
        one = '' if 'one' not in kwargs else kwargs['one']
        all = [''] * len(self.record_frames) if 'all' not in\
            kwargs else kwargs['all']
        
        assert len(all) == len(self.record_frames), \
            f'len all: {len(all)} not eq to len frames: {len(self.record_frames)}'
        
        def gen_frame(frame_list):
            # print('frame shape: ', frame_list[0].shape)  # [360, 640, 3]
            for frame, single in zip(frame_list, all):
                frame = cv2.resize(frame, dsize=(640, 360), interpolation=cv2.INTER_LINEAR)
                cv2.putText(frame, one, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.7, color=(255,255,255), thickness=1)
                cv2.putText(frame, single, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.7, color=(255,255,255), thickness=1)
                yield frame
        self.record_frames = gen_frame(self.record_frames.copy())
    
    def make_traj_video(self, frame_list, file_output="mark_test.mp4", fps=20):
        if getattr(self, 'record_frames', None) is None:
            return
        container = av.open(file_output, mode='w', format='mp4')
        stream = container.add_stream('h264', rate=fps)
        stream.width = 640 
        stream.height = 360
        stream.pix_fmt = 'yuv420p'
        for frame in frame_list:
            frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()


class MarkMiner(MarkBase): 
    '''
    The MarkMiner is implemented based on the Ray RLlib. 
    It is a neural network agent that takes the current 
    observation as input, output the action to take. 
    '''
    def __init__(
        self, 
        policy_config: Union[Tuple, Dict, DictConfig], 
        env: Optional[MinecraftWrapper] = None,
        **kwargs, 
    ) -> None: 
        
        super().__init__(**kwargs)
        if isinstance(policy_config, Tuple):
            # agent_config is a path to a yaml file
            hydra.core.global_hydra.GlobalHydra.instance().clear()
            initialize(config_path=policy_config[0], version_base='1.3')
            self.policy_config = compose(config_name=policy_config[1])
            OmegaConf.resolve(self.policy_config)
        elif isinstance(policy_config, Dict) or isinstance(policy_config, DictConfig):
            self.policy_config = policy_config
        else:
            raise ValueError("Agent policy_config must be a path or a dict. ")

        algo_config = (
            PPOConfig()
            .update_from_dict({
                'num_gpus_per_worker': 1,
                'env': 'dummy_env',
                'model': {
                    'custom_model': 'minecraft_agent',
                    'custom_model_config': {'policy_config': self.policy_config},
                    'custom_action_dist': 'hierarchical_action_dist',
                },
                '_disable_initialize_loss_from_dummy_batch': True, 
                '_enable_learner_api': False,
                '_enable_rl_module_api': False,
            })
        )
        
        self.algo = algo_config.build()
        self.env = env

    def reset_env(self, env):
        self.env = env
    
    def reset(self):
        noop_action = self.env.noop_action()
        self.obs = self.env.step(noop_action)[0]
        self.terminated = self.truncated = False
        self.state = self.algo.get_policy().get_initial_state()
        self.prev_action = np.array([0, 0])
        self.time_step = 0
        self.episode_reward = 0
    
    def step(self, condition: str = ''):
        # self.env.manual_set_text(condition)
        action, state_out,  _ = self.algo.compute_single_action(
            observation=self.obs, 
            state=self.state, 
            prev_action=self.prev_action, 
        )
        if len(state_out) == 13:
            state_out, extra = state_out[:-1], state_out[-1]
        else:
            extra = None
            assert len(state_out) == 12
        self.obs, reward, self.terminated, self.truncated, self.info = self.env.step(action)
        self.state = state_out
        self.prev_action = np.concatenate([action['buttons'], action['camera']], axis=0)
        self.time_step += 1
        self.episode_reward += reward
        if extra is not None:
            self.info['cursor_map'] = extra.squeeze(0)
        return self.obs, reward, self.terminated, self.truncated, self.info
    
    def do(self, condition: str = '', max_step=160, **kwargs):
        ''' kwargs: eval_mode,  '''

        self.reset()
        eval_mode = 'index' if 'eval_mode' not in kwargs else kwargs['eval_mode']
        max_reset_num = 2 if 'max_reset_num' not in kwargs else \
            kwargs['max_reset_num']
        max_reset_step = max_step // max_reset_num
        while not self.terminated and not self.truncated:
            if self.time_step % 10 == 0:
                print(f'step count: {self.time_step}')
            obs, _, _, _, info = self.step(condition)
            self.record_step()

            ''' check finishment '''
            end_invt, end_id = slot_mapping(kwargs['end'])
            start_invt, start_id = slot_mapping(kwargs['start'])
            item = info[end_invt][end_id]
            
            if item['quantity'] > 0:
                if eval_mode == 'index':
                    if info[start_invt][start_id]['quantity'] <= 0:
                        print(f'finished with step: {self.time_step} ...')
                        return True, self.time_step
                    else:
                        print(f'pickup wrong with step: {self.time_step} ...')
                        return False, self.time_step
                elif eval_mode == 'item':
                    if item['type'] == kwargs['item']:
                        print(f'finished with step: {self.time_step} ...')
                        return True, self.time_step
                    else:
                        print(f'pickup wrong with item({item["type"]}): {self.time_step} ...')
                        return False, self.time_step
                else:
                    raise f'unrecognized item mode: {eval_mode}'
            
            ''' reset init state '''
            if self.time_step % (max_reset_step) == max_reset_step - 1:
                print('reset init state')
                self.state = self.algo.get_policy().get_initial_state()
                self.prev_action = np.array([0, 0])
            
            ''' max step to test '''
            if self.time_step > max_step:
                print(f'max step {max_step} reached, episode end...')
                break

        return False, self.time_step


''' select a rand sub list from source list '''
def select_rand_num(num, source=[]):
    assert 1 <= num <= len(source), '1 <= num <= len(source)'
    select_box = list(source)
    selected = []
    for i in range(num):
        hit = random.choice(select_box)
        selected.append(hit)
        select_box.remove(hit)
    return selected


def get_random_inventory(env, fixed_items, rand_items, rand_num=8):
    quantity = 1
    commands = []
    for _ in range(rand_num):
        slot = random.randint(1, 35)
        item_name = random.choice(rand_items)
        commands.append(f'/replaceitem entity @p container.{slot} minecraft:{item_name} {quantity}')
    for slot, item_name in fixed_items:
        commands.append(f'/replaceitem entity @p container.{slot} minecraft:{item_name} {quantity}')
    for cmd in commands:
        env._env.env.execute_cmd(cmd)

class Log:

    def __init__(self, log_path):
        self.log_path = log_path

    def write(self, text):
        with open(self.log_path, 'a') as f:
            f.write(text+'\n')

def eval(model_code, eval_id, path_config,
         test_case=[(23, 41) , (12, 42), (3, 37)],
         num_eval_item=2,
         len_eval_item=5,
         title='',
         ):
    start = [1, 8, 9, 17, 27, 35, 22, 2, 10, 26, 23, 29]
    end = [36, 37, 38, 39, 40, 41, 42, 43, 44, 37, 38, 40]
    random.shuffle(start)
    random.shuffle(end)
    test_case = [(s, e) for s, e in zip(start, end)]
    path_ckpt_recipe4 = f'{JARVISBASE_PRETRAINED}/{version}/{model_code}/{model_code}.weights'
    out_dir = f'{JARVISBASE_TMP}/{version}/{model_code}_skill{eval_id}'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    success_num, total = 0, 0

    log = Log(f'{out_dir}/succ_{model_code}_{eval_id}.txt')
    log.write(f'------------{title}--------------')

    eval_mode = 'index'
    
    ''' policy config & load agent '''
    full_config = yaml.load(open(path_config), Loader=yaml.FullLoader)
    policy_config  = full_config['policy'] if 'policy' in full_config else full_config
    policy_config['from']['weights'] = path_ckpt_recipe4
    mark = MarkMiner(policy_config=policy_config, env=None)

    ''' rand select a sub list (no repeat) '''
    item_list = select_rand_num(num_eval_item, RECIPES_INGREDIENTS)
    log.write('Item list: {}'.format(', '.join(item_list)))

    for start, end in test_case:  # (10, 4) , (25, 6), (3, 1), (6, 5), (14, 7), (1, 0)
        item_list = select_rand_num(num_eval_item, RECIPES_INGREDIENTS)
        for item_name in item_list:  # RECIPES_INGREDIENTS[:3]
            log.write(f'Eval with item: {item_name}, start: {start}, end: {end}')
            print(f'fix {item_name} in pos {start}')
            os.system("ps -elf | grep 'mcp' | awk '{print $4}' | xargs kill -9")
            env = MinecraftWrapper('test', fixpos={
                0: {
                    'type': 'crafting_table',
                    'quantity': 1
                },
            })
            fixed_item = [(start, item_name)]
            rand_items = RECIPES_INGREDIENTS
            
            mark.reset_env(env)
 
            status_list, mark_list = [], []
            for i in range(len_eval_item):
                ''' test one trajectory '''
                _, info = env.reset(openinv=True, init_rand_move=True)  # adjust before run
                if not info['is_gui_open']:
                    print('inventory open failed, skip current eval')
                    continue
                get_random_inventory(env, fixed_item, rand_items, rand_num=5)
                env.manual_set_recipe({
                    'index': np.array([start, end]), 
                    'item': np.array(item_name_list.index(item_name)),
                })
                ''' max_reset_num is to reset init state, max_step // max_reset_num for each
                    reset '''
                status, traj_len = mark.do(start=start, end=end, item=item_name,
                    eval_mode=eval_mode, max_step=160, max_reset_num=2)
                status_list.append(status)
                mark_list += [f'{i}: succ' if status else f'{i}: fail'] * traj_len
                if status:
                    success_num += 1
                total += 1
                print(f'item: {item_name}, cur: {sum(status_list)}/{len(status_list)},', 
                    f'total: {success_num}/{total}')
                log.write(f'item: {item_name}, cur: {sum(status_list)}'\
                        f'/{len(status_list)}, total: {success_num}/{total}')
            ''' caption frame and turn frame list into generator '''
            caption_one = f'start:{start}, end:{end}' if eval_mode == 'index' else f'end:{end}, item: {item_name}'
            mark.caption_frame(all=mark_list, one=caption_one)
            mark.make_traj_video(mark.record_frames, file_output=f'{out_dir}/{model_code}_{eval_id}_{start}_{item_name}{i+1}.mp4')
            if getattr(mark, 'record_infos', None) is not None and len(mark.record_infos) > 0:
                mark.make_traj_video(mark.record_infos, file_output=f'{out_dir}/{model_code}_{eval_id}_{start}_{item_name}{i+1}_heat.mp4')
            mark.record_infos, mark.record_frames = [], []
    print('success: ', success_num, ', total: ', total)
    return success_num, total

'''
ps -elf | grep 'ray' | awk '{print $4}' | xargs kill -9
CUDA_VISIBLE_DEVICES='0,1' python ${JARVISBASE_ROOT}/jarvis/assembly/mark.py rcp4_0812v11
ls ~/ray | xargs -I xxx rm ~/ray/xxx -r

'''
if __name__ == '__main__':

    # pdb.set_trace()

    model_code, eval_id = sys.argv[1], '00'
    path_policy = f'{JARVISBASE_PRETRAINED}/{version}/{model_code}/config.yaml'
    if not os.path.exists(path_policy):
        print('<warning> --- using default config --- ')
        path_policy = f'{os.getenv("JARVISBASE_ROOT")}/jarvis/arm/configs'\
            '/policy/vpt_cursor.yaml'
    try:
        test_comp = False
        for filename in os.listdir(f'{JARVISBASE_PRETRAINED}/{version}/{model_code}'):
            if '.json' in filename:
                # test_comp = True
                break
        if test_comp:
            all_conn = [f'{i}_{j}' for i in range(1, 36) for j in range(36, 45)]
            connected = json.load(open(f'{JARVISBASE_PRETRAINED}/{version}/{model_code}/{filename}'))
            out_conn = list(set(all_conn) - set(connected))
            if len(out_conn) < 10:
                eval(model_code, eval_id, path_policy)
            else:
                test_case = [ (int(item.split('_')[0]), int(item.split('_')[1])) for item in random.choices(out_conn, k=3)]
                eval(model_code, '06', path_policy, test_case=test_case, title='out of composition case')
                os.system("ps -elf | grep 'mcp' | awk '{print $4}' | xargs kill -9")
                test_case = []
                while len(test_case) < 3:
                    item = random.choice(connected)
                    start, end = item.split('_')
                    if 0 < int(start) < 36:
                        test_case.append((int(start), int(end)))
                eval(model_code, '08', path_policy, test_case=test_case, title='in distribution case')
        else:
            eval(model_code, eval_id, path_policy)
    except Exception as e:
        raise e
    finally:
        os.system("ps -elf | grep 'mcp' | awk '{print $4}' | xargs kill -9")
    
    


    
