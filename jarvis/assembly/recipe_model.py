import torch
import torch.nn as nn
import numpy as np

from jarvis.arm.src.utils.hogepoge import (
    PositionalBox,
    PositionalRecipe,
)
from jarvis.arm.src.utils.vpt_lib.util import FanInInitReLULayer

class MinecraftStagePredictionModel(nn.Module):
    
    def __init__(self, box_num=9, num_stages=11, dim=1024, switch={
        'action': False,
        'camera': False,
    }):
        super(MinecraftStagePredictionModel, self).__init__()
        self.switch = switch
        self.num_stages = num_stages
        self.box_num = box_num
        self.dim = dim
        # convlution and position embedding
        self.pos_emb = PositionalBox(slot_num=9, dim_out=self.dim)
        self.pos_recipe = PositionalRecipe(recipe_len=9, recipe_dim=10,
                                           dim_out=self.dim)

        # Linear layer for action features
        if self.switch['action']:
            action_input_dim = 4 if self.switch['camera'] else 2
            self.action_linear = FanInInitReLULayer(action_input_dim, 
                                                    self.dim,
                                                    layer_type='linear',
                                                    use_activation=True)
        
        self.mid0 = FanInInitReLULayer(self.dim, self.dim,
                                                layer_type='linear',
                                                use_activation=True)
        
        self.mid1 = FanInInitReLULayer(self.dim, self.dim,
                                                layer_type='linear',
                                                use_activation=True)
        
        # LSTM to process image sequences
        self.lstm = nn.LSTM(input_size=self.dim, 
                            hidden_size=self.dim, 
                            num_layers=2, batch_first=True)


        # Combined linear layer for final prediction
        self.final_linear = FanInInitReLULayer(self.dim, self.num_stages,
                                                layer_type='linear',
                                                use_activation=False)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input):
        # Reshape images for LSTM input
        # the first self.box_num box is resources
        boxes = input['box'][..., :20*self.box_num, :] / 255
        boxes = boxes.permute(0, 1, 4, 2, 3).contiguous()  # [B, T, H, W, C]-> [B, T, C, H, W]
        boxes_emb = self.pos_emb(boxes)

        if self.switch['action']:
            camera = input['camera']  # ['action']
            attack = input['attack'].unsqueeze(dim=-1)
            use = input['use'].unsqueeze(dim=-1)
            if self.switch['camera']:
                action = torch.cat((camera, attack, use), dim=-1).float()
            else:
                action = torch.cat((attack, use), dim=-1).float()
            action_emb = self.action_linear(action)
            emb = boxes_emb + action_emb
        else:
            emb = boxes_emb
        ''' fuse recipe overall '''
        recipes = input['plan']  # [B, T, L, C]
        emb_recipe = self.pos_recipe(recipes)

        emb = self.mid0(emb)

        # emb = emb + emb_recipe
        emb = emb * emb_recipe + emb

        emb = self.mid1(emb)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(emb)

        # Select the last time step's output
        # if not self.train:
        #     lstm_out = lstm_out[:, -1, :]

        # Final prediction
        predictions = self.final_linear(lstm_out)
        return self.softmax(predictions)


class RecipeWrapper():

    def __init__(self, model_path:str, device='cpu') -> None:
        self.box_num, self.num_stage = 9, 11
        self.model = MinecraftStagePredictionModel(box_num=self.box_num, 
                        num_stages=self.num_stage, dim=1024, switch={
                            'action': False, 'camera': False,
                        })
        print('torch load recipe model: {}'.format(model_path))
        checkpoint = torch.load(model_path)
        ''' when train with nn.Parallel, some name will begin with "module." '''
        state_dict = {
            k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device)

        self.B, self.T = 1, 20
        self.box_wid = 20

        self.device_type = {'dtype': torch.float32, 'device': device}
        self.reset()

    def reset(self):
        self.history = {
            'box': torch.zeros((self.B, self.T, self.box_wid, 
                self.box_wid * self.box_num, 3), **self.device_type),
            'camera': torch.zeros((self.B, self.T, 2), **self.device_type),
            'attack': torch.zeros((self.B, self.T), **self.device_type),
            'use': torch.zeros((self.B, self.T), **self.device_type),
            'plan': torch.zeros((self.B, self.T, 9, 10), **self.device_type),
        }
    
    def predict(self, boxes, plan, action):
        dpiece = {}
        dpiece['box'] = torch.from_numpy(boxes[:, :self.box_wid * self.box_num, :], 
                ).float().to(self.device_type['device']).unsqueeze(0).unsqueeze(0)
        dpiece['plan'] = torch.from_numpy(plan)\
            .float().to(self.device_type['device']).unsqueeze(0).unsqueeze(0)
        for k in ['camera', 'attack', 'use']:
            if type(action[k]) != np.ndarray:
                action[k] = np.array(action[k])
            dpiece[k] = torch.from_numpy(action[k])\
                .float().to(self.device_type['device']).unsqueeze(0).unsqueeze(0)
        data = {}
        for k, v in self.history.items():
            data[k] = torch.cat((self.history[k], dpiece[k]), dim=1)[:, 1:, ...]
        res = self.model(data)
        _, predict = torch.max(res[:, -1, :], 1)
        self.history = data
        assert 0 <= predict[0] < 11, f'predict: {predict} should be in 0-11'\
            f'res shape: {res.shape}'
        return predict[0]
    
    
