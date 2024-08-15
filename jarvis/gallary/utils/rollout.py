import os
import uuid
import json

import string
import secrets
from pathlib import Path
from  datetime import datetime

import av
import numpy as np
from rich.console import Console

from typing import (
    Dict, List, Union, Sequence, Mapping, Any, Optional
)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def write_video(
    file_name: str, 
    frames: Union[List[np.ndarray], bytes], 
    width: int = 640, 
    height: int = 360, 
    fps: int = 20
) -> None:
    """Write video frames to video files. """
    with av.open(file_name, mode="w", format='mp4') as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        for frame in frames:
            frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(frame):
                    container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)



class AuxilaryFuncs(object):
    
    def accomplishment_by_inventory(obs_seq: Sequence[Dict]) -> Sequence[List[str]]:
        '''inventories: Sequence[Mapping[int, Dict[str, int]]]'''
        inventories = [ obs['inventory'] for obs in obs_seq ]
        def _sumup(inventory: Mapping[int, Dict[str, int]]) -> Dict[str, int]:
            result = dict()
            for slot, item in inventory.items():
                name = item['type']
                quantity = item['quantity']
                if name == 'none':
                    continue
                result[name] = result.get(name, 0) + quantity
            return result
            
        accomplishments = []
        last_inventory = None
        for i, inventory in enumerate(inventories):
            this_accomp = []
            if last_inventory: 
                last_sumup = _sumup(last_inventory)
                this_sumup = _sumup(inventory)
                for name in this_sumup.keys():
                    delta = this_sumup.get(name, 0) - last_sumup.get(name, 0)
                    if delta > 0:
                        this_accomp.append(name)
            accomplishments.append(this_accomp)
            last_inventory = inventory
        return accomplishments

    
    def accomplishment_by_kill_entity(obs_seq: Sequence[Dict]) -> Sequence[List[str]]:
        '''kill_entities: Sequence[Mapping[str, int]]'''
        kill_entities = [ obs['kill_entity'] for obs in obs_seq ]
        accomplishments = []
        last_kill_entity = []
        for i, kill_entity in enumerate(kill_entities):
            this_accomp = []
            if last_kill_entity:
                for name in kill_entity.keys():
                    delta = kill_entity[name] - last_kill_entity[name]
                    if delta > 0:
                        this_accomp.append(name)
            accomplishments.append(this_accomp)
            last_kill_entity = kill_entity
        return accomplishments
    
    
    def accomplishment_by_craft_item(obs_seq: Sequence[Dict]) -> Sequence[List[str]]:
        '''craft_items: Sequence[Mapping[str, int]]'''
        craft_items = [ obs['craft_item'] for obs in obs_seq ]
        accomplishments = []
        last_craft_item = []
        for i, craft_item in enumerate(craft_items):
            this_accomp = []
            if last_craft_item:
                for name in craft_item.keys():
                    delta = craft_item[name] - last_craft_item[name]
                    if delta > 0:
                        this_accomp.append(name)
            accomplishments.append(this_accomp)
            last_craft_item = craft_item
        return accomplishments
    
    
    def accomplishment_by_break_item(obs_seq: Sequence[Dict]) -> Sequence[List[str]]:
        '''break_items: Sequence[Mapping[str, int]]'''
        break_items = [ obs['break_item'] for obs in obs_seq ]
        accomplishments = []
        last_break_item = []
        for i, break_item in enumerate(break_items):
            this_accomp = []
            if last_break_item:
                for name in break_item.keys():
                    delta = break_item[name] - last_break_item[name]
                    if delta > 0:
                        this_accomp.append(name)
            accomplishments.append(this_accomp)
            last_break_item = break_item
        return accomplishments


ACCOMPLISHMENT_FN_MAPPING = {
    'inventory': AuxilaryFuncs.accomplishment_by_inventory,
    'kill': AuxilaryFuncs.accomplishment_by_kill_entity,
    'craft': AuxilaryFuncs.accomplishment_by_craft_item,
    'break': AuxilaryFuncs.accomplishment_by_break_item,
}


class Recorder:
    '''
    Recorder is used to record the trajectory of the agent. 
    '''
    def __init__(
        self, 
        root: dir,
        height: int = 360, 
        width: int = 640, 
        fps: int = 20, 
        enable_info: bool = True,
        enable_cursor: bool = True,
        enable_accomplishment: bool = False, 
        accomplishment_fns: List[str] = ['kill', 'craft', 'break'], 
        **kwargs, 
    ):
        self.height = height
        self.width = width
        self.fps = fps 
        self.enable_info = enable_info
        self.enable_accomplishment = enable_accomplishment
        self.enable_cursor = enable_cursor
        
        self.root = Path(root)
        if self.root.exists():
            # Console().log(f"Directory {self.root} already exists, remove it? ")
            # if input() == 'y':
            #     self.root.rmdir()
            # Console().log("Remove successfully")
            pass
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            Console().log(f"Create directory {self.root} successfully")

        self.video_dir = self.root / 'video'
        self.actions_dir = self.root / 'actions'
        self.infos_dir = self.root / 'infos'
        self.cursor_dir = self.root / 'cursor'
        self.accomplishments_dir = self.root / 'accomplishments'
        
        for dir in [
            self.video_dir, 
            self.actions_dir,
            self.infos_dir,  
            self.cursor_dir,
            self.accomplishments_dir
        ]:
            dir.mkdir(parents=True, exist_ok=True)

        self.accomplishment_fns = [
            ACCOMPLISHMENT_FN_MAPPING[fn] for fn in accomplishment_fns 
        ]
        

    def save_trajectory(
        self, 
        video: Union[List[np.ndarray], bytes], 
        actions: List, 
        cursors: List, 
        infos: List[Dict[str, Any]],
    ) -> None:
        '''
        Record trajectory with video, actions and infos. 
        Args:
            video: list of frames or bytes of video (encoded by pyav). 
            actions: list of actions. 
            infos: list of infos.
        Result:
            Generate a file name and save trajectories into `root/video`, 
            `root/actions`, and `root/infos` directories.
            For example, if the name is "abc1234_xx", then the video file 
            will be saved as `root/video/abc1234_xx.mp4`, the actions will
            be saved as `root/actions/abc1234_xx.json`. 
        '''
        
        # generate file name: {hour}_{minute}_{uuid}
        file_name = f"{datetime.now().hour}_{datetime.now().minute}_{uuid.uuid4().hex[:11]}"
        
        # save video
        video_file = self.video_dir / f"{file_name}.mp4"
        if isinstance(video, bytes):
            with open(video_file, "wb") as f:
                f.write(video)
        elif isinstance(video, list):
            write_video(str(video_file), video, self.width, self.height, self.fps)

        # save actions
        actions_dict = dict()
        for i in range(len(actions)):
            actions_dict[i] = actions[i]
        
        actions_json = json.dumps(actions_dict, cls=NumpyEncoder)
        actions_file = self.actions_dir / f"{file_name}.json"
        with actions_file.open("w") as f:
            f.write(actions_json)
        
        # save cursors
        cursors_dict = dict()
        for i in range(len(cursors)):
            cursors_dict[i] = cursors[i] 

        cursors_json = json.dumps(cursors_dict, cls=NumpyEncoder)
        cursors_file = self.cursor_dir / f"{file_name}.json"
        with cursors_file.open("w") as f:
            f.write(cursors_json)       
        
        # save infos
        if self.enable_info:
            infos_dict = dict()
            for i in range(len(infos)):
                infos_dict[i] = infos[i]
                if 'pov' in infos_dict[i]:
                    del infos_dict[i]['pov']  
            infos_json = json.dumps(infos_dict, cls=NumpyEncoder)
            infos_file = self.infos_dir / f"{file_name}.json"
            with infos_file.open("w") as f:
                f.write(infos_json)
        
        # save accomplishments
        if self.enable_accomplishment:
            length = len(infos)
            multiple_accomplishments = [ fn(infos) for fn in self.accomplishment_fns ]
            accomplishments = { i: [] for i in range(length) }
            for foo in multiple_accomplishments:
                for i, bar in enumerate(foo):
                    accomplishments[i].extend(bar)
            
            accomplishments_json = json.dumps(accomplishments, cls=NumpyEncoder)
            accomplishments_file = self.accomplishments_dir / f"{file_name}.json"
            with accomplishments_file.open("w") as f:
                f.write(accomplishments_json)