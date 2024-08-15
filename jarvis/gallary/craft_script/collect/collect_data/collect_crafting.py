
import yaml
import sys
import os
import random
import json
import ray
import string

from jarvis import HOME, JARVISBASE_TMP
from jarvis.gallary.utils.rollout import Recorder
from jarvis.gallary.craft_script.new_craft_collect import Worker_collect_new as Worker

JARVISBASE_ROOT = os.getenv('JARVISBASE_ROOT')
JARVISBASE_DATASET = os.getenv('JARVISBASE_DATASET')

TAG_ITEMS_JSON_PATH = f'{JARVISBASE_ROOT}/jarvis/assets/tag_items.json'
ENVS_TEMPLATE_PATH = f'{JARVISBASE_ROOT}/jarvis/global_configs/envs'\
    '/template.yaml'
ENVS_COLLECT_PATH = f'{JARVISBASE_ROOT}/jarvis/global_configs/envs'\
    '/collect'

RECIPES_INGREDIENTS = [
    'oak_planks', 'stick', 'diamond', 'cobblestone', 'leather',
    'iron_ingot', 'gold_ingot', 'paper', 'white_wool', 'string',
    'glass', 'coal', 'feather', 'egg', 'obsidian', 'blue_ice', 'clay',
    'redstone', 'wheat', 'apple']


# ----------------------- max len ------------------------------------------- #

with open(TAG_ITEMS_JSON_PATH, 'r') as f:
    TAG_ITEMS = json.load(f)

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

def init_inventory(target: list = [], uid: str = '__') :

    requires = []
    random_integers = [
        random.randint(0, len(RECIPES_INGREDIENTS)-1) for _ in range(9)
    ]
    for i in random_integers:
        requires.append({
            'type': RECIPES_INGREDIENTS[i], 
            'quantity_range': [0, 1]
        })

    
    with open(ENVS_TEMPLATE_PATH, 'r') as file:
        env_config_template = yaml.load(file, Loader=yaml.FullLoader)

    env_config_template['init_inventory'] = {
        0: {'type': 'crafting_table', 'quantity': 1}}
    random_integers = [
        random.randint(1, len(RECIPES_INGREDIENTS)-1) for _ in range(len(target))
    ]
    for i in range(len(target)):
        env_config_template['init_inventory'][target[i]] = {
        'type': RECIPES_INGREDIENTS[random_integers[i]],
        'quantity': 1
    }
    
    env_config_template['random_fill_inventory']['requires'] = requires

    with open(ENVS_COLLECT_PATH + '_' + uid + '.yaml', 'w') as file:
        yaml.dump(env_config_template, file, default_flow_style=False)



@ray.remote
def collect_crafting_overall(item_from, item_to, fill_list, 
                             from_item_index_list, num_craft_per_worker=5):
    worker = Worker("collect")

    output_all = []
    for i in range(num_craft_per_worker):
        fill_names = worker.reset(fill_list=fill_list)
        from_items = [fill_names[idx] for idx in from_item_index_list]
        done, info = worker.crafting(item_from, item_to, from_items)
        if done:
            output = [worker.outframes[:-1], 
                      worker.outactions,
                      worker.cursor_seq,
                      worker.start,
                      worker.end]
            output_all.append(output)
            # print(f"get a trajectory of crafting {target} {i+1}")
        else:
            pass
            # print(f"error of crafting {target} {i+1}, {info}")
    return output_all


def generate_from_to_list(max_extra_fill=6):
    from_list = []
    from_item_index_list = []
    to_list = []
    
    ''' init random move item number and position '''
    n_list = []
    for i in range(1, 10):
        n_list += [i] * i
    n = random.choice(n_list)

    init_item_from = select_rand_num(n, list(range(1, 36)))

    ''' get extra fill item position, random extra fill number '''
    extra_set = set(range(1, 36)) - set(init_item_from)
    extra_fill_num = random.randint(1, max_extra_fill)
    extra_fill = set(select_rand_num(extra_fill_num, list(extra_set)))

    fill_list = init_item_from + list(extra_fill)
      # number with item_index 0-len(fill_list)
    fill_item_index = fill_list.copy()

    def update_from_to(beg, end):
        from_list.append(beg)
        to_list.append(end)

        ''' trace item index for each item in fill list '''
        index_item = fill_item_index.index(beg)
        from_item_index_list.append(index_item)
        fill_item_index[index_item] = end  # update the new state

    init_item_to = set()
    ''' move a random number <n> items to resoure slots and adjust
        item position '''
    for index in init_item_from:
        ''' get rand end index '''
        rest_end = set(range(36, 45)) - init_item_to
        rand_end = random.choice(list(rest_end))
        init_item_to.add(rand_end)

        update_from_to(index, rand_end)

        if random.random() < 9./71 and 0 < len(init_item_to) < 9:
            adj_id = random.choice(list(init_item_to))
            rest_end = set(range(36, 45)) - init_item_to
            rand_end = random.choice(list(rest_end))

            init_item_to.remove(adj_id)
            init_item_to.add(rand_end)

            
            update_from_to(adj_id, rand_end)

    ''' move back items and adjust item position '''
    while len(init_item_to) > 0:
        mov_id = random.choice(list(init_item_to))
        rest_end = set(range(0, 36)) - extra_fill
        rand_end = random.choice(list(rest_end))

        init_item_to.remove(mov_id)
        extra_fill.add(rand_end)

        update_from_to(mov_id, rand_end)

        if random.random() < 9./71 and 0 < len(init_item_to) < 9:
            adj_id = random.choice(list(init_item_to))
            rest_end = set(range(36, 45)) - init_item_to
            rand_end = random.choice(list(rest_end))

            init_item_to.remove(adj_id)
            init_item_to.add(rand_end)

            update_from_to(adj_id, rand_end)
    return from_list, to_list, fill_list, from_item_index_list


if __name__ == '__main__':       
    num_workers, num_craft_per_worker = sys.argv[1:]
    num_workers = int(num_workers)
    num_craft_per_worker = int(num_craft_per_worker)

    ''' inventory to resource: 35 * 9
        resource to resource: 9 * 8, happen prob = 8/71
        resource to inventory: 9 * 36 
        strategy: 
            move a random number <n> items to resoure slots
            move items back into inventory 
            during above process, 8/71 prob to adjust the item in
            resource slots '''
    
    from_list, to_list, fill_list, from_item_index_list = generate_from_to_list()
    # code = 'rcp5_triple_item'
    code = 'rcp5_antelope'

    # print(f"item_from: {from_list}")
    # print(f"item_to: {to_list}")

    dataset_dir = f'{JARVISBASE_DATASET}'
    ray.init(num_cpus=int(num_workers), _temp_dir=f'{HOME}/ray') 

    result_ids = []
    for _ in range(num_workers):
        result_ids.append(collect_crafting_overall.remote(
            from_list, to_list, fill_list, 
            from_item_index_list, num_craft_per_worker))
    recorder = []
    n = len(from_list)
    for i in range(n):
        target_2 = to_list[i]
        target_1 = from_list[i]
        path = f'{dataset_dir}/{code}/{target_1}_{target_2}'
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        recorder.append(Recorder(path, enable_info=False))


    while result_ids:
        completed_ids, remaining_ids = ray.wait(result_ids, num_returns=1)
        completed_result = ray.get(completed_ids)

        for i in range(len(completed_result)):
            print(f'get {len(completed_result[i])} trajectories')
            for j in range(len(completed_result[i])):

                video = completed_result[i][j][0]
                action = completed_result[i][j][1]
                cursor = completed_result[i][j][2]
                start = completed_result[i][j][3]
                end = completed_result[i][j][4]

                if len(start) == n and len(end) == n:
                    for k in range(len(start)):
                        video_sub = video[start[k]:end[k]] 
                        action_sub = action[start[k]:end[k]]
                        cursor_sub = cursor[start[k]:end[k]]

                        target_1 = action_sub[0]['index_1'] 
                        target_2 = action_sub[0]['index_2']

                        recorder[k].save_trajectory(
                            video_sub,
                            action_sub,
                            cursor_sub,
                            [],
                        )

        result_ids = remaining_ids