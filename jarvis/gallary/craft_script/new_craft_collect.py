import time
import random
import math
import os 
import cv2
import re
import torch
import av
import ray
import rich
from rich.console import Console
from tqdm import tqdm
import uuid
import json
import hydra
import random
import numpy as np
from pathlib import Path
from typing import (
    Sequence, List, Mapping, Dict, 
    Callable, Any, Tuple, Optional, Union
)

from jarvis import (
    JARVISBASE_ROOT,
    JARVISBASE_DATASET,
    JARVISBASE_ROOT,
    JARVISBASE_TMP,
)
from jarvis.stark_tech.ray_bridge import MinecraftWrapper
from jarvis.gallary.utils.rollout import Recorder
from jarvis.gallary.craft_script import (
    KEY_POS_INVENTORY_W_RECIPE,
    KEY_POS_INVENTORY_WO_RECIPE,
    KEY_POS_TABLE_W_RECIPE,
    KEY_POS_TABLE_WO_RECIPE,
    CAMERA_SCALER,
    WIDTH, HEIGHT,
)

RECIPES_INGREDIENTS = [
    'oak_planks', 'stick', 'diamond', 'cobblestone', 'leather',
    'iron_ingot', 'gold_ingot', 'paper', 'white_wool', 'string',
    'glass', 'coal', 'feather', 'egg', 'obsidian', 'blue_ice', 'clay',
    'redstone', 'wheat', 'apple']

TAG_ITEMS_JSON_PATH = f'{JARVISBASE_ROOT}/jarvis/assets/tag_items.json'

with open(f'{JARVISBASE_ROOT}/jarvis/assets/all_items/all_items.json', 'r') as file:
    hash_item = json.load(file)
hash_item.append('none')

tag_item_path = f'{JARVISBASE_ROOT}/jarvis/assets/tag_items.json'
with open(tag_item_path, 'r') as file:
        tag_item = json.load(file)

# ----------------------- max len ------------------------------------------- #

with open(TAG_ITEMS_JSON_PATH, 'r') as f:
    TAG_ITEMS = json.load(f)

''' select a rand sub list from source list '''
def select_rand_num(num, source=[]):
    if num == 0:
        return []
    assert 1 <= num <= len(source), '1 <= num <= len(source)'
    select_box = list(source)
    selected = []
    for i in range(num):
        hit = random.choice(select_box)
        selected.append(hit)
        select_box.remove(hit)
    return selected

''' checkout position is valid (in default original image size). '''
def position_valid(x: Union[float, int], y: Union[float, int]):
    return 0 <= x <= WIDTH and 0 <= y <= HEIGHT

''' sample 2-D Gaussian point from origin point.
    var is gaussian variance, var_x=var_y=var or var_x=var[0] & var_y=var[1],
    we made point range restrict that point should in rg '''
def gaussian_sample(x: Union[float, int], y: Union[float, int], 
    var: Union[float, int, tuple, list], rg: Union[list, tuple, int, float]): 
    if type(rg) == int or type(rg) == float:
        st_x, ed_x, st_y, ed_y = x - rg, x + rg, y - rg, y + rg
    else:
        if len(rg) == 2:  # half rang of x & y
            st_x, ed_x, st_y, ed_y = x - rg[0], x + rg[0], y - rg[1], y + rg[1]
        elif len(rg) == 4:  # x from rg[0] to rg[1] & y: rg[2] - rg[3]
            st_x, ed_x, st_y, ed_y = x - rg[0], x + rg[1], y - rg[2], y + rg[3]
        else:
            raise 'range len must be 2 or 4, cur: {}'.format(len(rg))
    
    nor_x, nor_y = np.random.normal((x, y), var)
    if (st_x <= nor_x <= ed_x) and (st_y <= nor_y <= ed_y):
        return nor_x, nor_y
    return x, y

''' shuffle and copy a new dict '''
def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic

''' write frames to video files '''
def write_video(
    filename: str,
    frames: List[np.ndarray],
    width: int=640,
    height: int=360,
    fps: Union[int, float]=20
):
    with av.open(filename, mode='w', format='mp4') as container:
        stream = container.add_stream('h264', rate=fps)
        stream.width = width
        stream.height = height
        for frame in frames:
            frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

''' get each slot's center position '''
def COMPUTE_SLOT_POS(KEY_POS):
    result = {}
    for k, v in KEY_POS.items():
        left_top = v['left-top']
        right_bottom = v['right-bottom']
        row = v['row']
        col = v['col']
        prefix = v['prefix']
        start_id = v['start_id']
        width = right_bottom[0] - left_top[0]
        height = right_bottom[1] - left_top[1]
        slot_width = width // col  # 20
        slot_height = height // row  # 18
        slot_id = 0
        for i in range(row):
            for j in range(col):
                result[f'{prefix}_{slot_id + start_id}'] = (
                    left_top[0] + j * slot_width + (slot_width // 2), 
                    left_top[1] + i * slot_height + (slot_height // 2),
                )
                slot_id += 1
    return result

SLOT_POS_INVENTORY_WO_RECIPE = COMPUTE_SLOT_POS(KEY_POS_INVENTORY_WO_RECIPE)
SLOT_POS_INVENTORY_W_RECIPE = COMPUTE_SLOT_POS(KEY_POS_INVENTORY_W_RECIPE)
SLOT_POS_TABLE_WO_RECIPE = COMPUTE_SLOT_POS(KEY_POS_TABLE_WO_RECIPE)
SLOT_POS_TABLE_W_RECIPE = COMPUTE_SLOT_POS(KEY_POS_TABLE_W_RECIPE)

# <- bingo
SLOT_POS_MAPPING = {
    'inventory_w_recipe': SLOT_POS_INVENTORY_W_RECIPE,
    'inventory_wo_recipe': SLOT_POS_INVENTORY_WO_RECIPE,
    'crating_table_w_recipe': SLOT_POS_TABLE_W_RECIPE,
    'crating_table_wo_recipe': SLOT_POS_TABLE_WO_RECIPE,
}

def exception_exit(func):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print('Error: ', e)
            exit()
    return wrapper

''' agent that play in environment '''
# @ray.remote(max_restarts=-1, max_task_retries=-1)
class Worker_collect_new(object):
    
    def __init__(
        self, 
        env: Union[str, MinecraftWrapper],
        sample_ratio: float = 0.5,
        inventory_slot_range: Tuple[int, int] = (0, 36), 
        **kwargs, 
    )-> None:
        print("Initializing worker...")
        self.sample_ratio = sample_ratio
        self.inventory_slot_range = inventory_slot_range
    
        fill_list = kwargs['fill_list'] if 'fill_list' in kwargs else []
        if isinstance(env, str):
            self.env = MinecraftWrapper(env)
            self.reset(fill_list=fill_list)
        else:
            self.env = env
            self.reset(fake_reset=True, fill_list=fill_list)
        
    def reset(self, fake_reset: bool = False, fill_list=[]):
        self.outframes = []
        self.outactions = []
        self.plan_overall = []
        self.resource_record = {
            f'resource_{x}': {'type': 'none', 'quantity': 0} for x in range(9)
        }

        self.plan_phase = {'index_1': -1, 'index_2': -1, 'item': 'none'}

        self.cursor = [WIDTH // 2, HEIGHT // 2]

        self.cursor_seq = []
        
        if not fake_reset:
            self.obs, self.info = self.env.reset()
            while not self.info['is_gui_open']:
                print('Gui open false, reset again...')
                self.obs, self.info = self.env.reset()
            self.outframes.append(self.info['pov'].astype(np.uint8))
            fill_names = self.rand_init_inventory(fill_list)

        ''' if need recipe book, move to recipe button and open it '''
        if hasattr(self, 'current_gui_type') and self.current_gui_type:
            if self.current_gui_type == 'inventory_w_recipe':
                self.move_to_slot(SLOT_POS_INVENTORY_W_RECIPE, 'recipe_0')
                self._select_item()
            elif self.current_gui_type == 'crating_table_w_recipe':
                self.move_to_slot(SLOT_POS_TABLE_W_RECIPE, 'recipe_0')
                self._select_item()
                
            # if self.info['is_gui_open']:
            #     self._call_func('inventory')
        
        self.current_gui_type = None
        self.cursor_slot = 'none'
        self.crafting_slotpos = 'none'
        if not fake_reset:
            return fill_names
        return []

    def _assert(self, condition, message=None):
        if not condition:
            raise AssertionError(message)
        
    def _step(self, action):
        self.obs, _, _, _, self.info = self.env.step(action)

        self.cursor_seq.append(self.cursor)

        '''add resource info'''
        action['index_1'] = self.plan_phase['index_1']
        action['index_2'] = self.plan_phase['index_2']
        action['item'] = self.plan_phase['item']

        self.outframes.append(self.info['pov'].astype(np.uint8))
        self.outactions.append(dict(action))
        return self.obs, _, _, _, self.info


    def _call_func(self, func_name: str):
        action = self.env.noop_action()
        action[func_name] = 1
        for i in range(1):
            self.obs, _, _, _, self.info = self._step(action)
        action[func_name] = 0
        for i in range(5):
            self.obs, _, _, _, self.info = self._step(action)

    def _use_item(self):
        self._call_func('use')
    
    def _select_item(self):
        self._call_func('attack')
    
    def _palce_continue(self, times=1):
        action = self.env.noop_action()
        action['use'] = 1
        for i in range(1):
            self.obs, _, _, _, self.info = self._step(action)
        action['use'] = 0
        for i in range(1):
            self.obs, _, _, _, self.info = self._step(action)

    def move_with_order3(self, dl: float, dt: int):
        ''' the velocity is a curve, acceleraty come from a0 to -a0 '''
        assert dt > 0, f'dt: {dt} can not be zero or negitive'
        assert dl < 60. * dt, f'too fast with len: {dl}, t: {dt}, mx=60'
        a0 = 6*dl / dt**2
        s = 0.
        for t in range(1, dt + 1):
            x = a0 * t**2 * (3 - 2*t/dt) / 6
            yield x - s
            s = x

    def move_with_order2(self, dl: float, dt: int):
        ''' the velocity is a broken line, acceleraty = a0 then turn to -a0 '''
        assert dt > 0, f'dt: {dt} can not be zero or negitive'
        assert dl < 45. * dt, f'too fast with len: {dl}, t: {dt}, mx=45'
        a0 = 4*dl / dt**2
        s = 0.
        for t in range(1, dt + 1):
            x = a0 * t**2 / 2 if t <= dt/2 else \
                a0*dt*t - dl - a0 * t**2 / 2
            yield x - s
            s = x

    def move_to_pos(self, x: float, y: float, step_rg=(10, 20)):
        camera_x = x - self.cursor[0]
        camera_y = y - self.cursor[1]

        step_s =  max(math.ceil(abs(camera_x / 60.)), 
                     math.ceil(abs(camera_y / 60.)))
        if step_rg[0] < step_s:
            step_rg[0] = step_s
        if step_rg[1] <= step_s:
            step_rg[1] = step_s + 1
        num_steps= np.random.randint(*step_rg)

        for dx, dy in zip(self.move_with_order3(camera_x, num_steps), 
                          self.move_with_order3(camera_y, num_steps)):
            self.action_once(dx, dy)

    def move_to_pos_noisy(self, x: float, y: float, threshold: float=0.5, pos_variance=3.0, a=1):
        stay = True
        x, y = gaussian_sample(x, y, pos_variance, pos_variance)  # rand sample point around
        camera_x = x - self.cursor[0]
        camera_y = y - self.cursor[1]
        while not (abs(camera_x) < threshold and abs(camera_y) < threshold):
            r = math.sqrt(camera_x**2 + camera_y**2)
            mv_x, mv_y = gaussian_sample(x, y, r/2, (0, WIDTH, 0, HEIGHT))
            mv_cam_x, mv_cam_y = mv_x - self.cursor[0], mv_y - self.cursor[1]
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
            camera_x = x - self.cursor[0]
            camera_y = y - self.cursor[1]
        
    def init_move_or_stay(self, var=(WIDTH/8, HEIGHT/8)):
        nor_x, nor_y = gaussian_sample(WIDTH/2, HEIGHT/2, var, (WIDTH/2, HEIGHT/2))
        # 0.6 random move
        if random.random() > 0.01:
            dx, dy = nor_x - self.cursor[0], nor_y - self.cursor[1]
        # done twice, since one step maybe too long
            [self.action_once(dx/2, dy/2) for _ in range(2)]
        else:
            [self.action_once(0, 0) for _ in range(2)]
  
    def action_once(self, x: float, y: float):
        if not (abs(x) <= 180. and abs(y) <= 90.):
            print(f'step is too big dx: {x}, dy: {y}, move invalid')
            return 
        action = self.env.noop_action() 
        px, py = self.cursor[0] + x, self.cursor[1] + y
        if (px < 0. or px > 640.) or (py < 0. or py > 360.):
            return # do nothing
        action['camera'] = np.array([y * CAMERA_SCALER, x * CAMERA_SCALER])
        self.obs, _, _, _, self.info = self._step(action) 
        self.cursor = [px, py]

    def find_empty_box(self, inventory):
        empty_ids, empty_ids_bar = [], []
        for k, v in inventory.items():
            if v['type'] == 'none':
                empty_ids.append(k)
                if k < 9:
                    empty_ids_bar.append(k)
        return empty_ids, empty_ids_bar

    def move_to_slot(self, SLOT_POS: Dict, slot: str):
        self._assert(slot in SLOT_POS, f'Error: slot: {slot}')
        x, y = SLOT_POS[slot]
        self.move_to_pos_noisy(x, y)
        self.cursor_slot = slot

    def pull_item(self, 
        SLOT_POS: Dict, 
        item_from: str, 
        item_to: str
    ) -> None:
        if 'resource' in item_to:
            item = self.info['inventory'][int(item_from.split('_')[-1])]
            self.resource_record[item_to] = item
        self.move_to_slot(SLOT_POS, item_from)

        # self._select_item()
        self._use_item()

        self.move_to_slot(SLOT_POS, item_to)
        self._use_item()

    def find_in_inventory(self, labels: Dict, item: str, item_type: str='item', path=None):

        if path == None:
            path = []
        for key, value in labels.items():
            current_path = path + [key]
            if item_type == "item":
                if re.match(item, str(value)):
                    return current_path
                elif isinstance(value, dict):
                    result = self.find_in_inventory(value, item, item_type, current_path)
                    if result is not None:
                        return result[0]
            elif item_type == "tag":
                relative_path = os.path.join("jarvis/assets/tag_items.json")
                tag_json_path = os.path.join(JARVISBASE_ROOT, relative_path)
                with open(tag_json_path) as file:
                    tag_info = json.load(file)

                item_list = tag_info['minecraft:'+item]
                for i in range(len(item_list)):
                    if re.match(item_list[i][10:], str(value)):
                        return current_path
                    elif isinstance(value, dict):
                        result = self.find_in_inventory(value, item, item_type, current_path)
                        if result is not None:
                            return result[0]
        return None
    
    ''' generate resource recording item labels '''
    def get_labels(self):
        result = {}
        if 'table' in self.current_gui_type:
            num_slot = 9
        else:
            num_slot = 4
        for i in range(num_slot):  # 0-8
            slot = f'resource_{i}'
            item = self.resource_record[slot]
            result[slot] = item
        
        # generate inventory item labels 0-35
        for slot, item in self.info['inventory'].items():
            result[f'inventory_{slot}'] = item
        
        result['cursor_slot'] = self.cursor_slot
        result['gui_type'] = self.current_gui_type
        result['equipped_items'] = { k: v['type'] for k, v in self.info['equipped_items'].items()}
        
        return result
    
    ''' rand fill the inventory '''
    def rand_init_inventory(self, from_list, extra_fill_num):
        fixed_list = []
        for index in from_list:
            fixed_list.append(index)
        ''' in fixed_list position there must be items '''
        quantity = 1   # fix item number = 1
        extra_set = set(range(1, 36)) - set(fixed_list)
        assert extra_fill_num <= len(extra_set), \
            f'extra_fill_num {extra_fill_num} is larger than extra_set {len(extra_set)}'
        extra_fill = select_rand_num(extra_fill_num, list(extra_set))
        for slot in extra_fill + fixed_list:
            item_name = random.choice(RECIPES_INGREDIENTS)
            cmd = f'/replaceitem entity @p container.{slot} minecraft:{item_name} {quantity}'
            self.env._env.env.execute_cmd(cmd)

    def rand_init_inventory(self, fill_list):
        quantity = 1   # fix item number = 1
        fill_names = []
        for slot in fill_list:
            item_name = random.choice(RECIPES_INGREDIENTS)
            fill_names.append(item_name)
            cmd = f'/replaceitem entity @p container.{slot} minecraft:{item_name} {quantity}'
            self.env._env.env.execute_cmd(cmd)
        return fill_names
    
    def crafting(self, item_from: list=[], item_to: list=[], fill_names=[]):
        assert len(fill_names) == len(item_from), f'item_list: {len(fill_names)}, from: {len(item_from)}'
        item = fill_names

        try: 
            # initialize state 
            self.cursor = [WIDTH // 2, HEIGHT // 2]
            self.start, self.end = [], []
            self.crafting_slotpos = SLOT_POS_TABLE_WO_RECIPE
            self.current_gui_type = 'crating_table_wo_recipe'

            for i in range(len(item_from)):

                self.plan_phase['index_1'] = item_from[i]
                self.plan_phase['index_2'] = item_to[i]
                self.plan_phase['item'] = fill_names[i]
                self.init_move_or_stay()

                item_from_i = item_from[i] % 36
                item_to_i = item_to[i] % 36
                item_i = item[i]

                labels = self.get_labels()
                if item_from_i == -1:
                    item_from_i = self.find_in_inventory(labels, item)
                    self.plan_phase['item'] = item_i
                else:
                    if item_from[i] < 36:
                        labels=self.get_labels()
                        self.plan_phase['item'] = labels['inventory_'+str(item_from_i)]['type']
                    else:
                        labels=self.get_labels()
                        self.plan_phase['item'] = labels['resource_'+str(item_from_i)]['type']  

                # crafting...
                if item_from[i] > 35:
                    x = item_from[i] - 35
                else:
                    if item_from[i] < 9:
                        x = item_from[i] + 36
                    else:
                        x = item_from[i]

                if item_to[i] > 35:
                    y = item_to[i] - 35
                else:
                    if item_to[i] < 9:
                        y = item_to[i] + 36
                    else:
                        y = item_to[i]

                self.plan_phase['index_1'] = x + 1
                self.plan_phase['index_2'] = y + 1
                
                if item_from[i] < 36:
                    self.pull_item(self.crafting_slotpos, \
                                'inventory_' + str(item_from_i), 'resource_' + str(item_to_i))
                
                    self._assert(self.info['container_slots'][item_to_i+1]['quantity'], \
                                f'failure')  
                else:
                    if item_to[i] < 36:
                        self.pull_item(self.crafting_slotpos, \
                                    'resource_' + str(item_from_i), 'inventory_' + str(item_to_i))
                        
                        if item_to_i < 9:
                            item_to_i = item_to_i + 36
                        self._assert(self.info['container_slots'][item_to_i+1]['quantity'], \
                                    f'failure')  
                    else:
                        self.pull_item(self.crafting_slotpos, \
                                    'resource_' + str(item_from_i), 'resource_' + str(item_to_i))                        
                        self._assert(self.info['container_slots'][item_to_i+1]['quantity'], \
                                    f'failure')  
            # adjust slice  
            start, end = [], []
            tmp = []
            for i in range(len(self.outactions)-1):
                if self.outactions[i]['index_1'] != self.outactions[i+1]['index_1']:
                    if self.outactions[i]['index_1'] == -1:
                        tmp.append(i+1)
                    else:
                        tmp.append(i)
            
            tmp.append(len(self.outactions)-2)
            
            for i in range(len(tmp)):
                if i%2:
                    end.append(tmp[i])
                else:
                    start.append(tmp[i])

            self.start = start
            self.end = end
                    
        except AssertionError as e:
            return False, str(e) 
        
        return True, None

# 0 - 35
# 36 - 44
if __name__ == '__main__':
    worker = Worker_collect_new('collect', 
            fill_list=[15, 17, 5, 13, 32, 10, 14, 30, 2, 33, 34, 3, 6, 7, 21])
    done, info = worker.crafting(
        item_from=[15, 41, 17, 5, 13, 32, 10, 14, 43, 30, 2, 38, 44, 39, 42, 41, 40, 43, 37, 38, 44, 36], 
        item_to=[41, 36, 39, 44, 42, 43, 41, 37, 40, 43, 38, 15, 23, 44, 2, 31, 10, 38, 30, 19, 12, 29], 
        fill_names=[0, 0, 1, 2, 3, 4, 5, 6, 4, 7, 8, 8, 2, 1, 3, 5, 4, 7, 6, 7, 1, 0],
    )
    print(len(worker.outactions), len(worker.outframes), len(worker.cursor_seq))
    print(worker.start, worker.end)
    for i in range(len(worker.outactions)):
        print(worker.outactions[i]['index_1'], worker.outactions[i]['index_2'], 
              worker.outactions[i]['item'], worker.cursor_seq[i])
    # print(worker.cursor_seq)
    # print(done, info)
    # write video
    write_video(f'{JARVISBASE_TMP}/test_rand_fill.mp4', worker.outframes)