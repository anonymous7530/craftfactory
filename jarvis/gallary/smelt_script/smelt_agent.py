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
from typing import Sequence, List, Mapping, Dict, Callable, Any, Tuple, Optional
from jarvis.stark_tech.ray_bridge import MinecraftWrapper
from jarvis.assets import RECIPES_INGREDIENTS
from jarvis.gallary.utils.rollout import Recorder
from jarvis.gallary.craft_script.craft_agent import *


CAMERA_SCALER = 360.0 / 2400.0
WIDTH, HEIGHT = 640, 360

'''
KEY_POS_FU_WO_RECIPE
KEY_POS_INVENTORY_WO_RECIPE
'''
KEY_POS_FURNACE_WO_RECIPE = {
    'resource_slot': {
        'left-top': (287, 113), 
        'right-bottom': (303, 164), 
        'row': 2, 
        'col': 1,
        'prefix': 'resource', 
        'start_id': 0, 
    },
    'result_slot': {
        'left-top': (345, 127), 
        'right-bottom': (368, 152),
        'row': 1, 
        'col': 1,
        'prefix': 'result', 
        'start_id': 0, 
    },
    'hotbar_slot': {
        'left-top': (242, 236), 
        'right-bottom': (401, 256),
        'row': 1, 
        'col': 9, 
        'prefix': 'inventory', 
        'start_id': 0, 
    }, 
    'inventory_slot': {
        'left-top': (242, 178), 
        'right-bottom': (401, 234), 
        'row': 3, 
        'col': 9,
        'prefix': 'inventory',
        'start_id': 9,
    }, 
    'recipe_slot': {
        'left-top': (254, 132),
        'right-bottom': (272, 147),
        'row': 1, 
        'col': 1,
        'prefix': 'recipe', 
        'start_id': 0,
    }
}

KEY_POS_FURNACE_W_RECIPE = {
    'resource_slot': {
        'left-top': (361, 114), 
        'right-bottom': (383, 164), 
        'row': 2, 
        'col': 1,
        'prefix': 'resource', 
        'start_id': 0, 
    },
    'result_slot': {
        'left-top': (419, 127), 
        'right-bottom': (443, 152),
        'row': 1, 
        'col': 1,
        'prefix': 'result', 
        'start_id': 0, 
    },
    'hotbar_slot': {
        'left-top': (317, 240), 
        'right-bottom': (476, 256),
        'row': 1, 
        'col': 9, 
        'prefix': 'inventory', 
        'start_id': 0, 
    }, 
    'inventory_slot': {
        'left-top': (317, 178), 
        'right-bottom': (475, 234), 
        'row': 3, 
        'col': 9,
        'prefix': 'inventory',
        'start_id': 9,
    }, 
    'recipe_slot': {
        'left-top': (331, 131),
        'right-bottom': (346, 145),
        'row': 1, 
        'col': 1,
        'prefix': 'recipe', 
        'start_id': 0,
    }
}

SLOT_POS_FURNACE_WO_RECIPE = COMPUTE_SLOT_POS(KEY_POS_FURNACE_WO_RECIPE)
SLOT_POS_FURNACE_W_RECIPE = COMPUTE_SLOT_POS(KEY_POS_FURNACE_W_RECIPE)


class Worker_smelting(Worker):

    @exception_exit
    def reset(self, fake_reset: bool = False):
        if hasattr(self, 'current_gui_type') and self.current_gui_type:
            if self.current_gui_type == 'Furnace_w_recipe':
                self.move_to_slot(SLOT_POS_FURNACE_WO_RECIPE, 'recipe_0')
                self._select_item()
            elif self.current_gui_type == 'Furnace_wo_recipe':
                self.move_to_slot(SLOT_POS_FURNACE_WO_RECIPE, 'recipe_0')
                self._select_item()
                
            if self.info['is_gui_open']:
                self._call_func('inventory')
        
        if not fake_reset:
            self.obs, self.info = self.env.reset()
        
        self.current_gui_type = None
        self.resource_record = {f'resource_{x}': {'type': 'none', 'quantity': 0} for x in range(2)}
        self.cursor_slot = 'none'
        self.crafting_slotpos = 'none'
        self._null_action(1)

    def open_furnace_wo_recipe(self):
        self._null_action()
        labels = self.get_labels()
        inventory_id = self.find_in_inventory(labels, 'furnace')
        self._assert(inventory_id, 'no furnace')
        if inventory_id != 'inventory_0':
            self.open_inventory_wo_recipe()
            self.current_gui_type = 'inventory_wo_recipe'
            if labels['inventory_0']['type'] != 'none':
                for i in range(2):
                    del labels["resource_"+str(i)]
                inventory_id_none = self.find_in_inventory(labels, 'none')
                self.pull_item(SLOT_POS_INVENTORY_WO_RECIPE, 'inventory_0', inventory_id_none)
                self._null_action(2)
            self.pull_item(SLOT_POS_INVENTORY_WO_RECIPE, inventory_id, 'inventory_0')
        self._call_func('inventory')
        self._place_down()
        self.cursor = [WIDTH // 2, HEIGHT // 2]
        self._call_func('use')
        self.current_gui_type = 'Furnace_wo_recipe'
    
    def open_furnace_w_recipe(self):
        self.open_furnace_wo_recipe()
        self.move_to_slot(SLOT_POS_FURNACE_WO_RECIPE, 'recipe_0')
        self._select_item()
        self.current_gui_type = 'Furnace_w_recipe'

    def get_labels(self):
        result = {}
        # generate resource recording item labels
        for i in range(2):
            slot = f'resource_{i}'
            item = self.resource_record[slot]
            result[slot] = item
        
        # generate inventory item labels
        for slot, item in self.info['inventory'].items():
            result[f'inventory_{slot}'] = item
        
        result['cursor_slot'] = self.cursor_slot
        result['gui_type'] = self.current_gui_type
        result['equipped_items'] = { k: v['type'] for k, v in self.info['equipped_items'].items()}
        
        return result
    
    def _assert(self, condition, message=None):
        if not condition:
            # write_video(f'error/{message}.mp4', self.outframes)
            self._null_action(5)
            raise AssertionError(message)
        
    # smelting main
    def smelting(self, target: str, is_recipe=False, close_inventory=False):
        try:
            self.outframes = []
            self.outactions = []
            self.outinfos = []
            # recipe info
            cur_path = os.path.abspath(os.path.dirname(__file__))
            root_path = cur_path[:cur_path.find('jarvis')]
            relative_path = os.path.join("jarvis/assets/recipes", target + '.json')
            recipe_json_path = os.path.join(root_path, relative_path)
            with open(recipe_json_path) as file:
                recipe_info = json.load(file)
            if not self.current_gui_type:
                if is_recipe:
                    self.open_furnace_w_recipe()
                    self.crafting_slotpos = SLOT_POS_FURNACE_W_RECIPE
                else:
                    self.open_furnace_wo_recipe()
                    self.crafting_slotpos = SLOT_POS_FURNACE_WO_RECIPE

            # smelting
            print(f"need 1 smelting to get 1 {target}")
            self.smelting_once(target, recipe_info)

            # close inventory
            if close_inventory:
                labels = self.get_labels()
                inventory_id = self.find_in_inventory(labels, 'wooden_pickaxe')
                self._assert(inventory_id, f"no wooden_pickaxe to return furnace")
                if inventory_id != 'inventory_0':
                    if labels['inventory_0']['type'] != 'none':
                        for i in range(2):
                            del labels["resource_"+str(i)]
                        inventory_id_none = self.find_in_inventory(labels, 'none')
                        self.pull_item(self.crafting_slotpos, 'inventory_0', inventory_id_none)
                        self._null_action(2)
                    self.pull_item(self.crafting_slotpos, inventory_id, 'inventory_0')
                self._call_func('inventory')
                self.return_furnace()
                self.current_gui_type = None

        except AssertionError as e:
            return False, str(e) 
        return True, None
    
    def return_furnace(self):
            for i in range(20):
                self._attack_continue(10)
                labels = self.get_labels()
                if self.find_in_inventory(labels, 'furnace'):
                    break
            labels = self.get_labels()
            self._assert(self.find_in_inventory(labels, 'furnace'), f'return furnace') 


   # smelting once 
    def smelting_once(self, target: str,  recipe_info: Dict):
        slot_pos = self.crafting_slotpos 
        ingredient = recipe_info.get('ingredient')
        cook_time = recipe_info.get('cookingtime')
        items = dict()
        items_type = dict()
        # clculate the amount needed and store <item, quantity> in items
        if ingredient.get('item'):
            item = ingredient.get('item')[10:]
            item_type = 'item'
        else:
            item = ingredient.get('tag')[10:]
            item_type = 'tag'
        items_type[item] = item_type
        if items.get(item):
            items[item] += 1
        else:
            items[item] = 1
        items['fuels'] = 1
        items_type['fuels'] = 'tag'
        
        # place each item in order
        resource_idx = 0
        for item, _ in items.items():
            labels = self.get_labels()
            item_type = items_type[item]
            inventory_id = self.find_in_inventory(labels, item, item_type)
            self._assert(inventory_id, f"not enough {item}")
            inventory_num = labels.get(inventory_id).get('quantity')

            # place 
            self.pull_item(slot_pos, inventory_id, 'resource_' + str(resource_idx))
            resource_idx += 1

            # return the remaining items
            if inventory_num > 1:
                self.pull_item_return(slot_pos, inventory_id)

        self._null_action(int(cook_time))
        # get result
        # Do not put the result in resource_0-4/9 & resource_0-4   
        labels = self.get_labels()
        for i in range(2):
            del labels["resource_"+str(i)]
        for i in range(9):
            del labels["inventory_"+str(i)]
        # if tagret exists, stack it
        result_inventory_id_1 = self.find_in_inventory(labels, target)

        if result_inventory_id_1:
            item_num = labels.get(result_inventory_id_1).get('quantity')
            self.pull_items(self.crafting_slotpos, 1, 'result_0', result_inventory_id_1)
            labels_after = self.get_labels()
            item_num_after = labels_after.get(result_inventory_id_1).get('quantity')
            if item_num == item_num_after:
                result_inventory_id_2 = self.find_in_inventory(labels, 'none')
                self._assert(result_inventory_id_2, f"no space to place result")
                self.pull_item_continue(self.crafting_slotpos, result_inventory_id_2, target)
                # check result
                self._assert(self.get_labels().get(result_inventory_id_2).get('type') == target, f"fail for unkown reason") 
        else:
            result_inventory_id_2 = self.find_in_inventory(labels, 'none')
            self._assert(result_inventory_id_2, f"no space to place result")
            self.pull_items(self.crafting_slotpos, 1, 'result_0', result_inventory_id_2)
            # check result
            self._assert(self.get_labels().get(result_inventory_id_2).get('type') == target, f"fail for unkown reason")

        # clear resource          
        self.resource_record =  {f'resource_{x}': {'type': 'none', 'quantity': 0} for x in range(2)}

if __name__ == '__main__':
    pass
