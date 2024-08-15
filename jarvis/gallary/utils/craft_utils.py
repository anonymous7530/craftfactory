import numpy as np
from typing import Union

''' constant '''

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

SLOT_POS_TABLE_WO_RECIPE = COMPUTE_SLOT_POS(KEY_POS_TABLE_WO_RECIPE)


''' functional utils  '''

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