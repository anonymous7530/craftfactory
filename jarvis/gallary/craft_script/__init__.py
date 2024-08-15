from jarvis.gallary.craft_script.craft_agent import Worker as CraftScript
from jarvis.gallary.smelt_script.smelt_agent import Worker_smelting as SmeltScript


CAMERA_SCALER = 360.0 / 2400.0
WIDTH, HEIGHT = 640, 360

'''
KEY_POS_INVENTORY_W_RECIPE
KEY_POS_INVENTORY_WO_RECIPE
KEY_POS_TABLE_WO_RECIPE
KEY_POS_TABLE_W_RECIPE
'''
KEY_POS_INVENTORY_WO_RECIPE = {
    'resource_slot': {
        'left-top': (329, 114), 
        'right-bottom': (365, 150), 
        'row': 2, 
        'col': 2,
        'prefix': 'resource', 
        'start_id': 0, 
    },
    'result_slot': {
        'left-top': (385, 124), 
        'right-bottom': (403, 142),
        'row': 1, 
        'col': 1,
        'prefix': 'result', 
        'start_id': 0, 
    },
    'hotbar_slot': {
        'left-top': (239, 238), 
        'right-bottom': (401, 256),
        'row': 1, 
        'col': 9, 
        'prefix': 'inventory', 
        'start_id': 0, 
    }, 
    'inventory_slot': {
        'left-top': (239, 180), 
        'right-bottom': (401, 234), 
        'row': 3, 
        'col': 9,
        'prefix': 'inventory',
        'start_id': 9,
    }, 
    'recipe_slot': {
        'left-top': (336, 158),
        'right-bottom': (356, 176),
        'row': 1, 
        'col': 1,
        'prefix': 'recipe', 
        'start_id': 0,
    }
}

KEY_POS_INVENTORY_W_RECIPE = {
    'resource_slot': {
        'left-top': (406, 114), 
        'right-bottom': (442, 150), 
        'row': 2, 
        'col': 2,
        'prefix': 'resource', 
        'start_id': 0, 
    },
    'result_slot': {
        'left-top': (462, 124), 
        'right-bottom': (480, 142),
        'row': 1, 
        'col': 1,
        'prefix': 'result', 
        'start_id': 0, 
    },
    'hotbar_slot': {
        'left-top': (316, 238), 
        'right-bottom': (478, 256),
        'row': 1, 
        'col': 9, 
        'prefix': 'inventory', 
        'start_id': 0, 
    }, 
    'inventory_slot': {
        'left-top': (316, 180), 
        'right-bottom': (478, 234), 
        'row': 3, 
        'col': 9,
        'prefix': 'inventory',
        'start_id': 9, 
    }, 
    'recipe_slot': {
        'left-top': (413, 158),
        'right-bottom': (433, 176),
        'row': 1, 
        'col': 1,
        'prefix': 'recipe', 
        'start_id': 0,
    }
}

KEY_POS_TABLE_WO_RECIPE = {
    'resource_slot': {
        'left-top': (261, 113), 
        'right-bottom': (315, 167), 
        'row': 3, 
        'col': 3,
        'prefix': 'resource', 
        'start_id': 0, 
    },
    'result_slot': {
        'left-top': (351, 127), 
        'right-bottom': (377, 153),
        'row': 1, 
        'col': 1,
        'prefix': 'result', 
        'start_id': 0, 
    },
    'hotbar_slot': {
        'left-top': (239, 238), 
        'right-bottom': (401, 256),
        'row': 1, 
        'col': 9, 
        'prefix': 'inventory', 
        'start_id': 0, 
    }, 
    'inventory_slot': {
        'left-top': (239, 180), 
        'right-bottom': (401, 234), 
        'row': 3, 
        'col': 9,
        'prefix': 'inventory',
        'start_id': 9,
    }, 
    'recipe_slot': {
        'left-top': (237, 131),
        'right-bottom': (257, 149),
        'row': 1, 
        'col': 1,
        'prefix': 'recipe', 
        'start_id': 0,
    }
}

KEY_POS_TABLE_W_RECIPE = {
    'resource_slot': {
        'left-top': (338, 113), 
        'right-bottom': (392, 167), 
        'row': 3, 
        'col': 3,
        'prefix': 'resource', 
        'start_id': 0, 
    },
    'result_slot': {
        'left-top': (428, 127), 
        'right-bottom': (454, 153),
        'row': 1, 
        'col': 1,
        'prefix': 'result', 
        'start_id': 0, 
    },
    'hotbar_slot': {
        'left-top': (316, 238), 
        'right-bottom': (478, 256),
        'row': 1, 
        'col': 9, 
        'prefix': 'inventory', 
        'start_id': 0, 
    }, 
    'inventory_slot': {
        'left-top': (316, 180), 
        'right-bottom': (478, 234), 
        'row': 3, 
        'col': 9,
        'prefix': 'inventory',
        'start_id': 9,
    }, 
    'recipe_slot': {
        'left-top': (314, 131),
        'right-bottom': (334, 149),
        'row': 1, 
        'col': 1,
        'prefix': 'recipe', 
        'start_id': 0,
    }
}