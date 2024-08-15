'''
author:        caishaofei-MUS2 <1744260356@qq.com>
date:          2023-05-05 15:44:33
Copyright © Team CraftJarvis All rights reserved
'''
import av
import os
import time
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from collections import OrderedDict
from typing import Dict, List, Tuple, Union, Sequence, Mapping, Any, Optional

import cv2
import json
import torch
import numpy as np
import random
import math

from jarvis.stark_tech.entry import env_generator
from jarvis.arm.src.utils.vpt_lib.actions import ActionTransformer
from jarvis.arm.src.utils.vpt_lib.action_mapping import CameraHierarchicalMapping
from jarvis.gallary.utils.craft_utils import (
    SLOT_POS_TABLE_WO_RECIPE,
    gaussian_sample,
)
from jarvis.assembly.recipe_model import RecipeWrapper

CAMERA_SCALER = 360.0 / 2400.0


ENV_CONFIG_DIR = Path(__file__).parent.parent / "global_configs" / "envs"
RELATIVE_ENV_CONFIG_DIR = "../global_configs/envs"

centet_szie = 40
SLOT_NUM = 47
WIDTH, HEIGHT = 640, 360

ACTION_TRANSFORMER_KWARGS = dict(
    camera_binsize=2,
    camera_maxval=10,
    camera_mu=10,
    camera_quantization_scheme="mu_law",
)


KEYS_TO_INFO = ['pov', 'inventory', 'equipped_items', 'life_stats', 'location_stats', 'use_item', 'drop', 'pickup', 'break_item', 'craft_item', 'mine_block', 'damage_dealt', 'entity_killed_by', 'kill_entity', 'full_stats', 'player_pos', 'is_gui_open']

item_index_mapping = json.load(open(f'{os.getenv("JARVISBASE_ROOT")}/jarvis/assets'\
                                    '/all_items/items_mapping.json'))
# item_list = json.load(open(f'{os.getenv("JARVISBASE_ROOT")}/jarvis/assets'\
#                                     '/all_items/items_mapping.json'))


def resize_image(img, target_resolution = (224, 224)):
    return cv2.resize(img, dsize=target_resolution, interpolation=cv2.INTER_LINEAR)


class MinecraftWrapper(gym.Env):
    
    ACTION_SPACE_TYPE = 'Dict'
    action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
    action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)
    

    MAX_LENGTH = 50
    CHARSET = """~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|} """

    @classmethod
    def text_to_tensor(cls, text: Union[str, List]) -> np.ndarray:
        if isinstance(text, str):
            padding_length = cls.MAX_LENGTH - len(text)
            return np.pad(np.array( [cls.CHARSET.index(c) for c in text] ), (0, padding_length), 'constant')
        elif isinstance(text, list):
            return np.array( [cls.text_to_tensor(t) for t in text] )
        
    @classmethod
    def tensor_to_text(cls, tensor: Union[np.ndarray, torch.Tensor]) -> Union[str, List]:
        if tensor.ndim == 1:
            remove_padding_str = ''.join([cls.CHARSET[c] for c in tensor if c != 0])
            return remove_padding_str
        else:
            result = []
            for i in range(tensor.shape[0]):
                result.append( cls.tensor_to_text(tensor[i]) )
            return result
                

    @classmethod
    def get_obs_space(cls, width=640, height=360):
        return spaces.Dict({
            'img': spaces.Box(low=0, high=255, shape=(height, width, 3), 
                              dtype=np.uint8),
            'text': spaces.Box(low=0, high=len(cls.CHARSET)-1, 
                               shape=(cls.MAX_LENGTH,), dtype=np.uint8), 
            # <- bingo
            'img_cursor': spaces.Box(low=0, high=255, shape=(height, width, 3), 
                              dtype=np.uint8),
            'box': spaces.Box(low=0, high=255, shape=(20, 20*SLOT_NUM, 3), 
                              dtype=np.uint8),
            'index_1': spaces.Box(low=0, high=46, shape=(), dtype=np.uint8),
            'index_2': spaces.Box(low=0, high=46, shape=(), dtype=np.uint8),
            'item': spaces.Box(low=0, high=630, shape=(), dtype=np.int64),
        })
    
    @classmethod
    def get_action_space(cls):
        '''
        Convert the action space to the type of 'spaces.Tuple', 
        since spaces.Dict is not supported by ray.rllib. 
        '''
        if MinecraftWrapper.ACTION_SPACE_TYPE == 'Dict':
            return spaces.Dict(cls.action_mapper.get_action_space_update())
        elif MinecraftWrapper.ACTION_SPACE_TYPE == 'Tuple':
            original_action_space = cls.action_mapper.get_action_space_update()
            return spaces.Tuple((original_action_space['buttons'], original_action_space['camera']))
        else:
            raise ValueError(f'Unsupported action space type: {MinecraftWrapper.ACTION_SPACE_TYPE}')

    @classmethod
    def get_dummy_action(cls, B: int, T: int, device="cpu"):
        '''
        Get a dummy action for the environment.
        '''
        ac_space = cls.get_action_space()
        action = ac_space.sample()
        
        dummy_action = {}
        if isinstance(action, OrderedDict):
            for key, val in action.items():
                dummy_action[key] = (
                    torch.from_numpy(val)
                    .reshape(1, 1, -1)
                    .repeat(B, T, 1)
                    .to(device)
                )
        elif isinstance(action, tuple):
            dummy_action = (
                torch.from_numpy(action)
                .reshape(1, 1, -1)
                .repeat(B, T, 1)
                .to(device)
            )
        else:
            raise NotImplementedError
        
        return dummy_action

    @classmethod
    def agent_action_to_env(cls, agent_action):
        """Turn output from policy into action for MineRL"""
        # This is quite important step (for some reason).
        # For the sake of your sanity, remember to do this step (manual conversion to numpy)
        # before proceeding. Otherwise, your agent might be a little derp.
        action = agent_action
        # First, convert the action to the type of dict
        if isinstance(action, tuple):
            action = {
                'buttons': action[0], 
                'camera': action[1], 
            }
        # Second, convert the action to the type of numpy
        if isinstance(action["buttons"], torch.Tensor):
            action = {
                "buttons": action["buttons"].cpu().numpy(),
                "camera": action["camera"].cpu().numpy()
            }
        # Here, the action is the type of dict, and the value is the type of numpy
        minerl_action = cls.action_mapper.to_factored(action)
        minerl_action_transformed = cls.action_transformer.policy2env(minerl_action)
        return minerl_action_transformed

    @classmethod
    def env_action_to_agent(cls, minerl_action_transformed, to_torch=True, check_if_null=False, device="cuda"):
        """
        Turn action from MineRL to model's action.

        Note that this will add batch dimensions to the action.
        Returns numpy arrays, unless `to_torch` is True, in which case it returns torch tensors.

        If `check_if_null` is True, check if the action is null (no action) after the initial
        transformation. This matches the behaviour done in OpenAI's VPT work.
        If action is null, return "None" instead
        """
        if isinstance(minerl_action_transformed["attack"], torch.Tensor):
            minerl_action_transformed = {key: val.cpu().numpy() for key, val in minerl_action_transformed.items()}

        minerl_action = cls.action_transformer.env2policy(minerl_action_transformed)
        if check_if_null:
            if np.all(minerl_action["buttons"] == 0) and np.all(minerl_action["camera"] == cls.action_transformer.camera_zero_bin):
                return None

        # Add batch dims if not existant
        if minerl_action["camera"].ndim == 1:
            minerl_action = {k: v[None] for k, v in minerl_action.items()}
        
        # Merge temporal and batch dimension
        if minerl_action["camera"].ndim == 3:
            B, T = minerl_action["camera"].shape[:2]
            minerl_action = {k: v.reshape(B*T, -1) for k, v in minerl_action.items()}
            action = cls.action_mapper.from_factored(minerl_action)
            action = {key: val.reshape(B, T, -1) for key, val in action.items()}
        else:
            action = cls.action_mapper.from_factored(minerl_action)
            
        if to_torch:
            action = {k: torch.from_numpy(v).to(device) for k, v in action.items()}

        return action


    def __init__(self, env_config: Union[str, Dict, DictConfig], **kwargs) -> None:
        super().__init__()
        if isinstance(env_config, str):
            hydra.core.global_hydra.GlobalHydra.instance().clear()
            initialize(config_path=RELATIVE_ENV_CONFIG_DIR, version_base='1.3')
            self.env_config = compose(config_name=env_config)
        elif isinstance(env_config, Dict) or isinstance(env_config, DictConfig):
            self.env_config = env_config
        else:
            raise ValueError("env_config must be a string or a dict")
        
        if 'fixpos' in kwargs:
            self.env_config.init_inventory = kwargs['fixpos']
        self._env, self.additional_info = env_generator(self.env_config)
        
        width, height = self.env_config['resize_resolution'] # 224x224
        self.resize_resolution = (width, height)
        self.action_space = MinecraftWrapper.get_action_space()
        self.observation_space = MinecraftWrapper.get_obs_space(width=width, height=height)
        # the coordinate x, y is correlated to y(yaw), p(pitch) & w(width), h(height)
        # the camera is dy, dx or d_pitch, d_yaw or d_h, d_w
        self.cursor_pos = [180., 320.]  # pitch, yaw or (h, w)
        self.is_gui_open = False

        self.manual_set_recipe()

    def get_cursor_image(self, obs_pov):
        half_size = 10
        p, y = self.cursor_pos
        p, y = int(p), int(y)
        
        H, W, C = obs_pov.shape
        assert 0 <= p <= H and 0 <= y <= W, f'pitch: {p}, yaw: {y}, output of range'
        img_cursor = np.zeros_like(obs_pov, dtype=np.uint8)
        st_im = (max(p-half_size, 0), max(y-half_size, 0))
        ed_im = (min(p+half_size, H), min(y+half_size, W))
        target_im = obs_pov[st_im[0]: ed_im[0], st_im[1]: ed_im[1], :]
        img_cursor[st_im[0]: ed_im[0], st_im[1]: ed_im[1], :] = target_im.copy()

        ts_boxes = []
        HF_SH, HF_SW = 10, 10  # half slot height, half slot width
        for slot_name, (cx, cy) in SLOT_POS_TABLE_WO_RECIPE.items():
            assert cy-HF_SH >=0 and cy+HF_SH <=360 and cx-HF_SW >=0 and \
                cx+HF_SW <=640
            ts_boxes.append(obs_pov[cy-HF_SH:cy+HF_SH,
                                    cx-HF_SW:cx+HF_SW, :].copy())
            pass
        assert len(ts_boxes) == SLOT_NUM
        box = np.concatenate(ts_boxes, axis=-2)
        return img_cursor, box

    def manual_set_text(self, text: str):
        '''Set the text to be returned by the environment. '''
        self.manual_text = text   

    def manual_set_recipe(self, recipe={
            'index': np.array([0, 0]), 'item': np.array(630)}):  # 
        self.recipe = recipe
        # print('set recipe: {}'.format(self.recipe))

    def _build_obs(self, input_obs: Dict, info: Dict) -> Dict:
        
        # return_text = getattr(self, 'manual_text', info['text'])
        return_text = 'raw'
        recipe = getattr(self, 'recipe', {
            'index': np.array([0, 0]), 'item': np.array(630),
        })
        img = input_obs['pov']
        img_cursor, box = self.get_cursor_image(input_obs['pov'])
        
        output_obs = {
            'img': resize_image(img, self.resize_resolution),
            'text': MinecraftWrapper.text_to_tensor( return_text ),
            'img_cursor': resize_image(img_cursor, self.resize_resolution),  # <- bingo
            'box': box,
            'index_1': recipe['index'][0],  # action index put here
            'index_2': recipe['index'][1],
            'item': recipe['item'],
        }
        return output_obs

    def step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        CAMERA_SCALER = 360.0 / 2400.0
        '''Takes three kinds of actions as environment inputs. '''

        if isinstance(action, dict) and 'attack' in action.keys():
            minerl_action = action
        else:
            # Hierarchical action space to factored action space
            minerl_action = MinecraftWrapper.agent_action_to_env(action)

        # <- bingo |<
        if self.is_gui_open:
            tp_p = self.cursor_pos[0] + minerl_action['camera'][0] / CAMERA_SCALER
            tp_y = self.cursor_pos[1] + minerl_action['camera'][1] / CAMERA_SCALER
            if (tp_p < 5. or tp_p > 355.) or (tp_y < 5. or tp_y > 635. ):
                minerl_action['camera'] = np.array([0., 0.])
            else:
                self.cursor_pos = np.array([tp_p, tp_y])

            for key in minerl_action:  # mask action that not valid in crafting inv
                if key not in ['attack', 'use', 'camera']:
                    minerl_action[key] = np.array(0)
        
        obs, reward, terminated, info = self._env.step(minerl_action)
        trauncated = terminated
        self.is_gui_open = info['is_gui_open']

        if 'event_info' in info and len(info['event_info']) > 0:
            print("env info:", info['event_info'])

        # output img_cursor in evaluation
        info['img_cursor'], info['box'] = self.get_cursor_image(info['pov'])
        info['recipe'] = self.recipe
        
        return (
            self._build_obs(obs, info), 
            reward, 
            terminated, 
            trauncated, 
            info,
        )

    def open_crafting_inv(self,):
        # look down
        action = self.noop_action()
        for i in range(2):
            action['camera'] = np.array([88, 0])
            obs, reward, terminated, trauncated, inf = self.step(action)
        # jump    
        action = self.noop_action() 
        action['jump'] = 1
        for i in range(1):
            obs, reward, terminated, trauncated, inf = self.step(action)
        action['jump'] = 0
        for i in range(5):
            obs, reward, terminated, trauncated, inf = self.step(action)
        # use 
        action = self.noop_action()
        for i in range(2):
            action['use'] = 1
            for i in range(1):
                obs, reward, terminated, trauncated, inf = self.step(action)
            action['use'] = 0
            for i in range(5):
                obs, reward, terminated, trauncated, inf = self.step(action)
        return obs, inf
    
    def action_once(self, x: float, y: float):
        # the coordinate x, y is correlated to y(yaw), p(pitch) & w(width), h(height)
        if not (abs(x) <= 180. and abs(y) <= 90.):
            print(f'step is two big dx: {x}, dy: {y}, move invalid')
            return 
        action = self.noop_action() 
        px, py = self.cursor_pos[1] + x, self.cursor_pos[0] + y
        if (px < 0. or px > 640.) or (py < 0. or py > 360.):
            return # do nothing
        action['camera'] = np.array([y * CAMERA_SCALER, x * CAMERA_SCALER])
        self.obs, _, _, _, self.info = self.step(action) 
        self.cursor_pos = [py, px]

    def move_to_pos_noisy(self, x: float, y: float, threshold: float=0.5, pos_variance=3.0, a=1):
        stay = False
        x, y = gaussian_sample(x, y, pos_variance, pos_variance)  # rand sample point around
        camera_x = x - self.cursor_pos[0]
        camera_y = y - self.cursor_pos[1]
        while not (abs(camera_x) < threshold and abs(camera_y) < threshold):
            r = math.sqrt(camera_x**2 + camera_y**2)
            mv_x, mv_y = gaussian_sample(x, y, r/2, (0, WIDTH, 0, HEIGHT))
            mv_cam_x, mv_cam_y = mv_x - self.cursor_pos[0], mv_y - self.cursor_pos[1]
            mv_cam_r = math.sqrt(mv_cam_x**2 + mv_cam_y**2)
            v_x = math.sqrt(2*a*abs(mv_cam_x)**2/mv_cam_r) 
            v_y = math.sqrt(2*a*abs(mv_cam_y)**2/mv_cam_r)
            # Move step should be within remain length, when close to target
            # it will violate. When close to target, sample gaussian velocity
            v_xl, v_yl = abs(mv_cam_x), abs(mv_cam_y)
            if stay:
                v_xl, v_yl = gaussian_sample(v_x, v_y, (abs(mv_cam_x), abs(mv_cam_y)),
                                         (0, 2*abs(mv_cam_x), 0, 2*abs(mv_cam_y)))
            v_x = v_xl if abs(mv_cam_x) < v_x else v_x
            v_y = v_yl if abs(mv_cam_y) < v_y else v_y
            v_x = v_x if mv_cam_x >= 0 else -v_x
            v_y = v_y if mv_cam_y >= 0 else -v_y
            self.action_once(v_x, v_y)
            camera_x = x - self.cursor_pos[0]
            camera_y = y - self.cursor_pos[1]
    

    def init_rand_move(self):
        # the coordinate x, y is correlated to y(yaw), p(pitch) & w(width), h(height)
        nor_x, nor_y = gaussian_sample(320, 180, (160,90), (300, 160))
        if self.is_gui_open:
            dx, dy = nor_x - self.cursor_pos[1], nor_y - self.cursor_pos[0]
            [self.action_once(dx/2, dy/2) for _ in range(2)]


    def get_item_back(self, start, end):
        cor_start = SLOT_POS_TABLE_WO_RECIPE[f'resource_{start}']
        cor_end = SLOT_POS_TABLE_WO_RECIPE[f'inventory_{end}']
        
        ''' move to start point '''
        self.move_to_pos_noisy(cor_start)

        ''' get item (attack) '''
        action = self.noop_action()
        action['attack'] = 1
        for i in range(1):
            self.obs, _, _, _, self.info = self.step(action)
        action['attack'] = 0
        for i in range(5):
            self.obs, _, _, _, self.info = self.step(action)
        
        ''' move to end point '''
        self.move_to_pos_noisy(cor_end)

        ''' put item (use) '''
        action = self.noop_action()
        action['use'] = 1
        for i in range(1):
            self.obs, _, _, _, self.info = self.step(action)
        action['use'] = 0
        for i in range(5):
            self.obs, _, _, _, self.info = self.step(action)

    
    def reset(self, *, seed=None, options=None, openinv=True, 
              init_rand_move=False) -> Tuple[Dict, Dict]:
        obs, info = self._env.reset()
        self.manual_set_recipe()
        # <- bingo
        self.cursor_pos = [180., 320.]
        ''' open crafting inventory table '''
        if openinv:
            obs, info = self.open_crafting_inv()
            if init_rand_move:
                self.init_rand_move()
        
        info['img_cursor'], info['box'] = self.get_cursor_image(info['pov'])
        info['recipe'] = self.recipe
        return obs, info

    def noop_action(self):
        return self._env.noop_action()

    def seed(self, seed: int) -> None:
        self._env.seed(seed)

    def render(self):
        return self._env.render()


if __name__ == "__main__":
    env = MinecraftWrapper("test", fixpos={
        0: {
            'type': 'crafting_table',
            'quantity': 1
        },
        3: {
            'type': 'oak_planks',
            'quantity': 1,
        }
    })
    obs, info = env.reset(openinv=False)
    env.manual_set_recipe({
        'index': np.array([0, 0]), 'item': np.array(630)
    })
    tmpdir = f'{os.getenv("JARVISBASE_ROOT")}/tmp'
    container = av.open(f"{tmpdir}/env_test2.mp4", mode='w', format='mp4')
    stream = container.add_stream('h264', rate=1)
    stream.width = 640 
    stream.height = 360
    stream.pix_fmt = 'yuv420p'
    # import queue 
    from queue import Queue
    fps_queue = Queue()
    for i in range(10):
        time_start = time.time()
        action = env.action_space.sample()
        obs, reward, terminated, trauncated, info = env.step(action)
        time_end = time.time()
        curr_fps = 1/(time_end-time_start)

        fps_queue.put(curr_fps)
        if fps_queue.qsize() > 200:
            fps_queue.get()
        average_fps = sum(list(fps_queue.queue))/fps_queue.qsize()
        text = f"frame: {i}, fps: {curr_fps:.2f}, avg_fps: {average_fps:.2f}"
        if i % 20 == 0:
            print(text)
        if i % 30 == 29:
            obs, info = env.reset(openinv=False)
        frame = resize_image(info['pov'], (640, 360))
        
        for i, item in enumerate(info['container_slots']):
            if item['quantity'] > 0:
                # cv2.putText(frame, f'{item["type"]}: {item["slot_id"]}', (150, 25 + i*15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                pass
        if 'recipe' in info:
            # cv2.putText(frame, 'rcp: {},{}, item: {}'.format(
            #     info['recipe']['index'][0], info['recipe']['index'][1], info['recipe']['item']
            # ), (150, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            pass
        
        frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)
    
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    # print(info['container_slots'])

    # from collections import OrderedDict
    # act = OrderedDict()
    # act['buttons'] = action['buttons'].reshape((1, 1)).repeat(5, 0)
    # act['camera'] = action['camera'].reshape((1, 1)).repeat(5, 0)
    # torch_act = OrderedDict({
    #     'buttons': torch.from_numpy(act['buttons']),
    #     'camera': torch.from_numpy(act['camera']),
    # })
    # import ipdb; ipdb.set_trace()
    

    # from tqdm import tqdm
    # for i in tqdm(range(100000000)):
    #     action = env.action_space.sample()
    #     factored_action = MinecraftWrapper.agent_action_to_env(action)
    #     horbars = [(x, y.item()) for x, y in factored_action.items() if 'hotbar' in x]
    
    #     r = sum( y for x, y in horbars)
    #     if r > 1:
    #         print(i, r)
    #     # print(factored_action)

