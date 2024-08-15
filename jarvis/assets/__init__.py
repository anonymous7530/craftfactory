import json
from pathlib import Path
from typing import (
    Optional, Sequence, List, Tuple, Dict, Union, Callable
)

FILE_DIR = Path(__file__).parent

SPAWN_FILE = FILE_DIR / "spawn.json"
MC_CONSTANTS_FILE = FILE_DIR / "mc_constants.1.16.json"
COMMON_LABELS_FILE = FILE_DIR / "common_labels.json"
CARED_ITEMS_FILE = FILE_DIR / "cared_items.json"
RECIPES_DIR = FILE_DIR / "recipes"
COLOR_FILE = FILE_DIR / "colors.json"

RECIPES_BOOKS = {}
RECIPES_INGREDIENTS = []

for recipe in Path(RECIPES_DIR).glob("*.json"):
    with recipe.open("r") as f:
        RECIPES_BOOKS[recipe.stem] = json.load(f)
        if RECIPES_BOOKS[recipe.stem]["type"] == "minecraft:crafting_shapeless":
            for ingredient in RECIPES_BOOKS[recipe.stem]["ingredients"]:
                if 'item' in ingredient:
                    RECIPES_INGREDIENTS.append(ingredient['item'].replace("minecraft:", ""))
                elif 'tag' in ingredient:
                    RECIPES_INGREDIENTS.append(ingredient['tag'].replace("minecraft:", ""))
        elif RECIPES_BOOKS[recipe.stem]["type"] == "minecraft:crafting_shaped":
            for key, ingredient in RECIPES_BOOKS[recipe.stem]["key"].items():
                if 'item' in ingredient:
                    RECIPES_INGREDIENTS.append(ingredient['item'].replace("minecraft:", ""))
                elif 'tag' in ingredient:
                    RECIPES_INGREDIENTS.append(ingredient['tag'].replace("minecraft:", ""))

RECIPES_INGREDIENTS = sorted(list(set(RECIPES_INGREDIENTS)))
# print('Len of ingredients:', len(RECIPES_INGREDIENTS))
# print(RECIPES_INGREDIENTS)

with open(MC_CONSTANTS_FILE, "r") as f:
    MC_CONSTANTS = json.load(f)

with open(COMMON_LABELS_FILE, "r") as f:
    COMMON_ITEMS_IDX_TO_NAME = json.load(f)

with open(CARED_ITEMS_FILE, "r") as f:
    CARED_ITEMS = json.load(f)

ALL_BIOMES = ['forest', 'plains']
ALL_WEATHERS = ['clear', 'rain', 'thunder']
EQUIP_SLOTS = ['head', 'feet', 'chest', 'legs', 'offhand']

ALL_ITEMS = {}
EQUIPABLE_ITEMS = {}
for item in MC_CONSTANTS["items"]:
    ALL_ITEMS[item['type']] = item
    if item['bestEquipmentSlot'] in EQUIP_SLOTS:
        equip_slot = item['bestEquipmentSlot']
        if equip_slot not in EQUIPABLE_ITEMS:
            EQUIPABLE_ITEMS[equip_slot] = []
        EQUIPABLE_ITEMS[equip_slot].append(item['type'])
    
ALL_ITEMS_IDX_TO_NAME = sorted(list(ALL_ITEMS.keys()))
ALL_ITEMS_NAME_TO_IDX = {item: idx for idx, item in enumerate(ALL_ITEMS_IDX_TO_NAME)}


KEYS_TO_INFO = [
    'pov', 
    'inventory', 
    'equipped_items', 
    'life_stats', 
    'location_stats', 
    'use_item', 
    'drop', 
    'pickup', 
    'break_item', 
    'craft_item', 
    'mine_block', 
    'damage_dealt', 
    'entity_killed_by', 
    'kill_entity', 
    'full_stats', 
    'player_pos', 
    'is_gui_open',
    'container_slots',
]