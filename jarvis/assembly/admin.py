''' this is the admin program that control the overall process of experiment
    (hope so), including training & evaluating '''
import os
import time
import re
import torch
import GPUtil
import socket
import numpy as np
import json
import multiprocessing

from jarvis import (
    JARVISBASE_ROOT, JARVISBASE_PRETRAINED, 
    JARVISBASE_OUTPUT, JARVISBASE_TMP,
)
from jarvis.assembly import (
    host, hostname, version, socket_port,
)
from jarvis.assembly.mark import eval

def check_gpu_memory(needg=12, device_num=4):
    ''' check whether there is 12g x 4gpu '''
    gpus = GPUtil.getGPUs()
    remains = np.array([gpu.memoryFree for gpu in gpus])
    needg = needg * 1000
    ct = 0
    while (remains >= needg).sum() < device_num:  # MB
        if ct == 0:
            print(f'res mem(MB): {remains}, needg: {needg}, devices: {device_num}')
        gpus = GPUtil.getGPUs()
        remains = np.array([gpu.memoryFree for gpu in gpus])
        ct = (ct + 1) % (1 << 10)
    print('mem enough, start program...')
    return np.arange(len(remains))[remains >= needg][:device_num].tolist()


def get_closest_start_time(start_time, format, dir_parent):
    if type(start_time) == str:
        str_start = start_time
    else:
        str_start = time.strftime(format, start_time)

    pattern = format.replace('%Y', r'\d{4}')
    for sign in 'mdHMS':
        pattern = pattern.replace(f'%{sign}', r'\d{2}')

    closest_subd = ''
    for sub in os.listdir(dir_parent):
        res = re.search(pattern, sub)
        if res is None:
            continue
        if res.group(0) >= str_start:
            ''' the closest time that after start time '''
            if closest_subd == '':
                closest_subd = sub
            elif closest_subd > sub:
                closest_subd = sub
    return closest_subd

def locate_output_dir(start_time, dir_result=f'{JARVISBASE_OUTPUT}'):
    ''' we check the directory that is the closet to the start time '''
    subd1 = get_closest_start_time(start_time, '%Y-%m-%d', dir_result)
    if subd1 == '':
        return ''
    subd2 = get_closest_start_time(start_time, '%H-%M-%S', f'{dir_result}/{subd1}')
    if subd2 == '':
        return ''
    return f'{subd1}/{subd2}'


def check_new_moel(train_dir, set_ckpt):
    pattern = r'checkpoint_\d{6}'
    for subd in os.listdir(train_dir):
        if re.search(pattern, subd) is None:
            continue
        if not os.path.exists(f'{train_dir}/{subd}/model'):
            continue
        if subd not in set_ckpt:
            return subd
    return None


def convert_ckpt(path_ckpt_in, target='recipe.weights'):
    print(''' wait for write model... ''')
    time.sleep(10)
    path_ckpt_out = f'{JARVISBASE_TMP}/weights/{target}'
    model = torch.load(f'{path_ckpt_in}/model')
    para = model['state_dict']
    remove_keys = [
        'clip_model',
    ]
    for key in list(para.keys()):
        rms = [item in key for item in remove_keys]
        if sum(rms) > 0:
            para.pop(key)
        else:
            para[key.replace('policy.', '')] = para.pop(key)
    torch.save(para, path_ckpt_out)

def worker(commands):
    os.system(' '.join(commands))

def convert_model_and_test(mac=hostname):  # 0.5
    
    ''' ########## important settings ########### '''
    percent_datause=1.0
    composition_dropout=0.6
    num_gpu = 3

    start_time = time.localtime()
    datanum=f'{int(60*percent_datause + 0.5)}k'
    ckpt_name = time.strftime(f'rcp5_%m%d%H{mac}_{datanum}.weights', 
                              start_time)
    dir_result = f'{JARVISBASE_OUTPUT}/ray_results/{version}'
    if not os.path.exists(dir_result):
        os.mkdir(dir_result)
    dir_output = f'{JARVISBASE_OUTPUT}/outputs'
    if not os.path.exists(dir_result):
        os.mkdir(dir_result)

    ''' -------------- start training program '''
    devices = check_gpu_memory(needg=12, device_num=num_gpu)
    str_devices = ','.join([str(id) for id in devices])
    print(f'-------------- selected devices: {str_devices}')
    commands = [
        f'CUDA_VISIBLE_DEVICES="{str_devices}"',
        f'python {JARVISBASE_ROOT}/jarvis/arm/offline.py',
        'policy=vpt_cursor',
        'data.mode=raw',
        'data.enable_cursor=True',
        f'data.percent_datause={percent_datause}',
        f'data.composition_dropout={composition_dropout}',
        'optimize.support=ray-lightning',
        'optimize.logger=wandb',
        f'optimize.project_name={version}',
        f'optimize.devices={num_gpu}',
        'optimize.num_workers=4',
        'optimize.learning_rate=0.000181',
        'optimize.experiment_name={}'.format(ckpt_name.split('.')[0]),
    ]
    # commands.append(f'policy.from.weights={JARVISBASE_TMP}')
    p = multiprocessing.Process(target=worker, args=(commands,))
    p.start()

    print('''--------------  wait for create train dir... ''')
    ct = 0
    while True:
        subd = get_closest_start_time(start_time, '%Y-%m-%d_%H-%M-%S', dir_result)
        if ct % 30 == 0:
            print(f'----- wait for create train dir... '\
                  f'start time: {time.strftime("%Y-%m-%d_%H-%M-%S", start_time)}')
        if subd != '':
            print('create train directory...')
            train_dir = f'{dir_result}/{subd}'
            break
        time.sleep(1)
        ct += 1

    ''' make log file '''
    log = open(f'{train_dir}/train_eval.log', 'a')
    log.write(f'------------- begin {ckpt_name} ------------\n')
    log.close()

    print('''--------------  wait for create output dir... ''')
    while True:
        subd = locate_output_dir(start_time, dir_output)
        path_config = f'{dir_output}/{subd}/.hydra/config.yaml'
        if subd != '' and os.path.exists(path_config):
            dir_config = f'{dir_output}/{subd}'
            break
        time.sleep(1)

    print('''--------------  copy config file ''')
    os.system('cp {} {}'.format(path_config, train_dir))

    print('''--------------  check model exist, convert, and eval with socket ''')
    eval_on = False
    if eval_on:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host = '0.0.0.0'
        port = socket_port
        server_socket.bind((host, port))
        server_socket.listen(5)

    set_ckpt, succ_result, succ_best = set(), {}, 0.
    while True:
        subd_ckpt = check_new_moel(train_dir, set_ckpt)
        if subd_ckpt is None:
            time.sleep(5)
            continue

        dir_ckpt = f'{train_dir}/{subd_ckpt}'

        print('''--------------  convert model... ''')
        

        if not eval_on:
            # convert_ckpt(dir_ckpt, ckpt_name.split('.')[0] + f'_{subd_ckpt}.weights')
            set_ckpt.add(subd_ckpt)
            continue
        else:
            convert_ckpt(dir_ckpt, ckpt_name)

        ''' check gpu memory '''
        devices = check_gpu_memory(
            needg=9, device_num=2
        )
        str_devices = ','.join([str(id) for id in devices])
        commands = [
            f'CUDA_VISIBLE_DEVICES="{str_devices}"',
            f'python {JARVISBASE_ROOT}/jarvis/assembly/admin.py',
            'client',
            f'{port}',
        ]
        print(f'''--------------  start client on devices: {devices} ''')
        p = multiprocessing.Process(target=worker, args=(commands,))
        p.start()

        print('''--------------  evaluate model... ''')
        client_socket, addr = server_socket.accept()
        
        message = {
            'ckpt_name': ckpt_name,
            'subd_ckpt': subd_ckpt,
            'path_config': path_config,
        }
        client_socket.send(json.dumps(message).encode())

        data = client_socket.recv(1024)
        data = json.loads(data.decode())
        client_socket.close()

        succ = data['succ']
        total = data['total']
        state_finish = data['state']
        if not state_finish and total == 0:
            log = open(f'{train_dir}/train_eval.log', 'a')
            log.write(f'ckpt: {subd_ckpt} failed to eval\n')
            log.close()
            continue

        succ_rate = 1.0 * succ / total if total > 0 else 0.
        set_ckpt.add(subd_ckpt)
        succ_result[subd_ckpt] = succ_rate

        log = open(f'{train_dir}/train_eval.log', 'a')
        log.write(f'ckpt: {subd_ckpt}, succ_rate: {succ_rate}, '\
                    f'succ: {succ}, total: {total}\n')

        print(''' -------------- compare and save best ckpt... ''')
        if succ_rate > succ_best:
            os.system('cp {} {}'.format(
                f'{JARVISBASE_PRETRAINED}/{ckpt_name}',
                f'{JARVISBASE_PRETRAINED}/{version}/{ckpt_name.split(".")[0]}_be.weights',
            ))
            succ_best = succ_rate
            log.write(f'ckpt {subd_ckpt} rank best saved ckpt\n')
        log.close()


def recv_eval(port):
    time.sleep(2)  # wait for server to recv message
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, int(port)))
    
    message = client_socket.recv(1024)
    message = json.loads(message)
    print(f"client receive：{message}")
    ckpt_name = message['ckpt_name']
    subd_ckpt = message['subd_ckpt']
    path_config = message['path_config']
    
    succ, total = 0, 0
    try:
        succ, total = eval(model_code=ckpt_name.split('.')[0], 
            eval_id=f'{subd_ckpt[-2:]}_01',
            path_config=path_config,
            test_case=[(23, 41) , (12, 42), (3, 37)],
            num_eval_item=2,
            len_eval_item=5,
        )
        message_to_send = {
            'succ': succ,
            'total': total,
            'state': True,
        }
    except Exception as e:
        print(f'evaluate ckpt {subd_ckpt} failed')
        message_to_send = {
            'succ': succ,
            'total': total,
            'state': False,
        }
    finally:
        client_socket.send(json.dumps(message_to_send).encode())
        client_socket.close()
        os.system("ps -elf | grep 'mcp' | awk '{print $4}' | xargs kill -9")
        os.system(f"ls {JARVISBASE_TMP}/ray | xargs -I xxx rm {JARVISBASE_TMP}/ray/xxx -r")


if __name__ == '__main__':
    import sys
    para = sys.argv[1]
    if para == 'main':
        convert_model_and_test()  # your machine
    elif para == 'client':
        recv_eval(sys.argv[2])

