import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import clip
import os

from jarvis.arm.src.utils.vpt_lib.util import FanInInitReLULayer
from jarvis.arm.src.utils.vpt_lib.minecraft_util import store_args

# policy arch
from jarvis.arm.src.utils.selfatten import Attention_Seqtovec as PoliArch
from jarvis.arm.src.utils.selfatten import SoftPositionEmbed as Howm
from jarvis.arm.src.utils.permutation import EquivariantRnn as EquiRNN


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=256) -> None:
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_item = torch.exp(
            -torch.arange(0, d_model, 2) * math.log(10000.) / d_model
        )  # [d_model//2]
        pe[:, 0::2] = torch.sin(position * div_item)  # [max_len, d_model//2]
        pe[:, 1::2] = torch.cos(position * div_item)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return x

class BoxEmbedding(nn.Module):

    def __init__(self, midchan=16, outchan=128, box_width=20) -> None:
        super().__init__()
        self.inchan, self.outchan = 3, outchan
        self.kernel_size = box_width
        self.cov0 = FanInInitReLULayer(
            self.inchan,
            midchan,
            kernel_size=3,
            padding=1,
            stride=1,
            init_scale=1.,
            layer_type='conv',
            batch_norm=True
        )
        self.cov1 = FanInInitReLULayer(
            midchan,
            self.outchan,
            kernel_size=self.kernel_size,
            padding=0,
            stride=self.kernel_size,
            init_scale=1.,
            layer_type='conv',
            batch_norm=True,
        )

    def forward(self, x):
        # x: [B, T, 3, 20, 20*slot_num]
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        x = self.cov0(x)  # [B, T, C, H, W]
        x = self.cov1(x)  # [B, T, C=128, H=1, W=slot_num]
        _, C, H, W = x.shape
        x = x.reshape(B, T, C, H*W)
        x = x.permute(0, 1, 3, 2)  # [B, T, Slot, Embedding]
        return x
    

class PositionalBox(nn.Module):

    def __init__(self, slot_num=47, dim_out=1024):
        super().__init__()
        self.slot_num = slot_num
        self.chan_mid, self.chan_out = 16, 16
        self.dim_mid = self.chan_out * self.slot_num
        self.dim_out = dim_out

        self.pos_enc = PositionalEncoding(self.chan_out, self.slot_num)
        self.box = BoxEmbedding(midchan=self.chan_mid, outchan=self.chan_out, 
                                box_width=20)
        self.mlp0 = FanInInitReLULayer(self.dim_mid, self.dim_mid, layer_type='linear', )
        self.mlp1 = FanInInitReLULayer(self.dim_mid, self.dim_out, layer_type='linear', 
                                       use_activation=True)

    def forward(self, x):
        # x: [B, T, 3, 20, 20*slot_num]
        assert 20 * self.slot_num == x.shape[-1], f'1 tensor shape: {x.shape}'
        x = self.box(x)
        B, T, Slot, D = x.shape
        assert Slot == self.slot_num, f'2 tensor shape: {x.shape}'

        x = x.reshape(B*T, Slot, D)
        x = self.pos_enc(x)
        x = x.reshape(B*T, Slot * D)

        x = self.mlp0(x)
        x = self.mlp1(x)
        return x.reshape(B, T, *x.shape[1:])


class PositionalRecipe(nn.Module):

    def __init__(self, recipe_len=9, recipe_dim=10, dim_out=1024) -> None:
        super().__init__()
        self.pos_enc = PositionalEncoding(recipe_dim, recipe_len)
        self.mlp0 = FanInInitReLULayer(recipe_dim*recipe_len, 
                                       recipe_dim*recipe_len, 
                                       layer_type='linear', 
                                       use_activation=False,)
        self.mlp1 = FanInInitReLULayer(recipe_dim*recipe_len, dim_out, 
                                       layer_type='linear',
                                       use_activation=True)

    def forward(self, input):
        # input shape: B, T, 9, 10
        B, T, L, D = input.shape
        input = input.reshape(B*T, L, D)
        pos_input = self.pos_enc(input)
        pos_input = pos_input.reshape(B, T, L*D)
        emb = self.mlp0(pos_input)
        return self.mlp1(emb)
    

class IndexEmbeddingSimple(nn.Module):

    @store_args
    def __init__(self, output_dim=512, emb_dim=46) -> None:
        super().__init__()
        self.eye_mat = nn.Parameter(
            torch.eye(emb_dim, emb_dim).float(), 
            requires_grad=False
        ) 
        self.ly0 = FanInInitReLULayer(emb_dim, emb_dim, layer_type='linear', 
                                      use_activation=False, init_scale=1.)
        self.ly1 = FanInInitReLULayer(emb_dim, output_dim, layer_type='linear',
                                      init_scale=1.)

    def forward(self, x):
        x = self.eye_mat[x]
        x = self.ly0(x)
        x = self.ly1(x)
        return x
    
class IndexListEmbedding(nn.Module):

    @store_args
    def __init__(self, output_dim=512, emb_dim=47, index_len=9) -> None:
        super().__init__()
        self.eye_mat = nn.Parameter(
            torch.eye(emb_dim, emb_dim).float(), 
            requires_grad=False
        )
        self.adapt_layer = FanInInitReLULayer(emb_dim, emb_dim, layer_type='linear', 
                use_activation=False, init_scale=1.)
        self.layers = nn.ModuleList([
            FanInInitReLULayer(emb_dim, emb_dim, layer_type='linear', 
                use_activation=False, init_scale=1.) for _ in range(index_len)
        ])
        self.ly1 = FanInInitReLULayer(emb_dim * index_len, output_dim, layer_type='linear',
                                      init_scale=1.)
    
    def forward(self, x):  # B, T, L_i
        mask = x < self.emb_dim - 1
        x = self.eye_mat[x]  # B, T, L, D
        x = x * mask.unsqueeze(-1)
        x = self.adapt_layer(x)
        B, T, L, N = x.shape
        x = x.reshape(B, T, -1)
        x = self.ly1(x)
        return x

class Printer:

    def __init__(self, gap=100):
        self.gap = gap
        self.counter = 0

    def p(self, x):
        self.counter = (self.counter + 1) % self.gap
        if self.counter == 0:
            pass

class CondEmbeddingWrapper(nn.Module):

    @store_args
    def __init__(self, output_dim=512, emb_dim=47, index_len=9, cond='self') -> None:
        super().__init__()

        self.cond_method = 'poliarch'  # self, poliarch, howm, equirnn
        if cond == 'self':
            self.cond_embed = IndexListEmbedding(output_dim, emb_dim, index_len)
        elif cond == 'poliarch':
            self.cond_embed = PoliArch(input_dim=output_dim, output_dim=output_dim, num_heads=1,
                                       num_layers=1, dim_feedforward=output_dim, emb_num=emb_dim)
        elif cond == 'howm':
            self.cond_embed = Howm(hidden_size=output_dim, emb_num=emb_dim, 
                                   resolution=[20, index_len])
        elif cond == 'equirnn':
            self.cond_embed = EquiRNN(hidden_size=output_dim, num_emb=emb_dim)
        else:
            raise f'no such cond: {cond}'
        self.printer = Printer(gap=100)
            

    def forward(self, x):  # [B, T, L_i]
        output = self.cond_embed(x)  # [B, T, D]
        self.printer.p(output[0, 0, 100:120])
        return output
            

class ItemEmbedding(nn.Module):

    @store_args
    def __init__(self, output_dim=512, emb_dim=631, mid_dim=512, mode='index') -> None:
        ''' mid_dim should be coinside with clip output dim
            mode: index(embedding onehot), cilp-text(text clip), 
                clip-image(image clip), image(raw pixel) '''
        super().__init__()
        if mode == 'index':
            self.mat = nn.Parameter(
                torch.eye(emb_dim, emb_dim).float(),requires_grad=False)
            self.ly0 = FanInInitReLULayer(emb_dim, mid_dim, layer_type='linear',
                                          use_activation=False)
        elif mode == 'clip-text' or mode == 'clip-image':
            self.mat = nn.Parameter(
                self.load_clip_pt(mode.split('-')[-1]), requires_grad=False)
            self.ly0 = FanInInitReLULayer(mid_dim, mid_dim, layer_type='linear',
                                          use_activation=False)
        elif mode == 'image':
            self.cnn = nn.Conv2d(3, 1, kernel_size=1, stride=1)
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.flat = nn.Flatten(start_dim=-3, end_dim=-1)
            self.ly0 = FanInInitReLULayer(24*24, mid_dim, layer_type='linear',
                                          use_activation=False)
        else:
            raise f'item mode name: {mode} is not in index(embedding onehot), '\
                'cilp-text(text clip), clip-image(image clip), image(raw pixel)'
        self.ly1 = FanInInitReLULayer(mid_dim, output_dim, layer_type='linear',
                                      init_scale=1.)

    def load_clip_pt(self, t='text'):  # 'text' or 'image'
        if t == 'text':
            return torch.load(f'{os.getenv("JARVISBASE_ROOT")}/jarvis/assets/clip_text.pt')\
                .reshape(631, 512)
        elif t == 'image':
            return torch.load(f'{os.getenv("JARVISBASE_ROOT")}/jarvis/assets/clip_image.pt')\
                .reshape(631, 512)
        else:
            raise f'"{t}" should be "text" or "image"'
        
    def forward(self, x):
        if self.mode in ['index', 'clip-text', 'clip-image']:
            if x.dtype != torch.long:
                x = x.long()
            x = self.mat[x]
        elif self.mode == 'image':  # [B, T, 48, 48, 3]
            x = self.cnn(x)
            x = self.maxpool(x)
            x = self.flat(x)
        else:
            raise f'"{self.mode}" should be "index, clip-text/image, or image"'
        x = self.ly0(x)
        x = self.ly1(x)
        return x



''' This is an Embedding about 3x3 resources Embedding '''
class IndexEmbedding(nn.Module):

    @store_args
    def __init__(self, emb_dim=512, emb_num=12, switch={'noise': False, 'relu': True}) -> None:
        super().__init__()
        # self.embed = nn.Embedding(self.emb_num, self.emb_dim)
        self.layer_norm = nn.LayerNorm(self.emb_num)
        self.layer = nn.Linear(self.emb_num, self.emb_dim)
        self.pos_emb = PositionalEncoding(self.emb_dim, self.emb_num)

        # Init Weights (Fan-In)
        init_scale = 1.
        self.layer.weight.data *= init_scale / self.layer.weight.norm(
            dim=tuple(range(1, self.layer.weight.data.ndim)), p=2, keepdim=True
        )
        # Init Bias
        if self.layer.bias is not None:
            self.layer.bias.data *= 0

    def forward(self, x):
        assert x.dtype == torch.int64
        pe0 = self.pos_emb.pe.squeeze(0)
        pos_x = Variable(pe0[x], requires_grad=False)
        eye = torch.eye(self.emb_num, device=x.device)
        x = eye[x].float()  # B, T, D
        if self.switch['noise']:
            noise = torch.randn_like(x) * 0.05
            x = x + noise
        else:
            x = x + 0.05
        x = self.layer_norm(x)
        x = self.layer(x)
        if self.switch['relu']:
            x = torch.relu(x)
        x = x + pos_x
        return x
    

class TextEmbedding(nn.Module):
    
    def __init__(self, device='cuda'):
        super().__init__()
        # first load into cpu, then move to cuda by the lightning
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def forward(self, texts, device="cuda", **kwargs):
        self.clip_model.eval()
        text_inputs = clip.tokenize(texts).to(device)
        embeddings = self.clip_model.encode_text(text_inputs)
        return embeddings
    

class ImageEmbedding(nn.Module):

    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        # first load into cpu, then move to cuda by the lightning
        print('loading clip model...')
        self.clip_model, preprocess = clip.load("ViT-B/32", device=device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.preprocess = preprocess
    
    @torch.no_grad()
    def forward(self, image, device="cuda"):
        self.clip_model.eval()
        # TODO convert image into 512 length embedding
        # return embeddings
        # print('raw: ', image.size)  # 48, 48
        if type(image) == list:
            image_list = [
                self.preprocess(img) for img in image
            ]
            image = torch.stack(image_list)
        else:
            image = self.preprocess(image).unsqueeze(0)
        # print('imagge:', image.dtype, image.shape)  # [1, 3, 224, 224]
        image = image.to(self.device)
        image_features = self.clip_model.encode_image(image)
        # print('image_feature:', image_features.dtype, image_features.shape)
        # torch.float16, [1, 512]
        return image_features.float()

class SwitchNN(nn.Module):

    @store_args    
    def _init__(self, layer_num=3):
        super().__init__()
        self.node_num = 2 ** layer_num
        self.weights = nn.Parameter(torch.normal(
            torch.zeros((self.layer_num, 2*self.node_num), 
                        dtype=torch.float32),
            torch.ones((self.layer_num, 2*self.node_num), 
                       dtype=torch.float32)*0.05))
        self.weights.requires_grad = True

        integer_value = 42

        # 将整数转换为二进制字符串，然后将字符串转换为整数列表
        binary_mat = []
        for i  in range(self.node_num):
            binary_list = [int(bit) for bit in bin(i)[2:]]
            for _ in range(self.layer):
                binary_mat.append(binary_list)

        # 使用torch.tensor创建二进制的tensor
        binary_tensor = torch.tensor(binary_list, dtype=torch.float32)

    def extend2binary(self, i:int):

        pass

    def N(self, i, j, k):

        return 

    def forward(self, x):

        pass


class SimpleCrossAttention(nn.Module):

    def __init__(self, C, D, H):
        ''' N, C, D, H represent dim of
         group, channel, dim, head '''
        super().__init__()
        self.multihead = nn.MultiheadAttention(C, H, batch_first=True)
        self.lly = nn.Linear(D, C)

    def forward(self, input, cond):
        B, T, N, C = input.shape
        ''' reshape cond into same shape as input '''
        cond_x = self.lly(cond).unsqueeze(-2)
        cond_x = cond_x.reshape(B*T, 1, C)
        input = input.reshape(B*T, N, C)
        attn_output, attn = self.multihead(cond_x, input, input)
        attn_output = attn_output.squeeze(-2).reshape(B, T, C)
        return attn_output



def gaussian_heatmap(height, width, center, sigma, device='cpu'):
    """
    Generate a 2D Gaussian heatmap.

    Args:
        height (int): Height of the heatmap.
        width (int): Width of the heatmap.
        center (tuple): Tuple (x, y) specifying the center of the Gaussian.
        sigma (float): Variance of the Gaussian.

    Returns:
        torch.Tensor: 2D Gaussian heatmap.
    """
    center_shape = center.shape
    x = torch.arange(0, width, 1, dtype=torch.float32).unsqueeze(0)  # [1, w]
    y = torch.arange(0, height, 1, dtype=torch.float32).unsqueeze(-1)  # [h, 1]
    x, y = x.to(device), y.to(device)
    for _ in center_shape[:-2]:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
    # center  # [B, 1, 1], [B, 1, 1]
    # Calculate the 2D Gaussian
    dist = ((x - center[..., 0:1].unsqueeze(-2)) ** 2 + (y - center[..., 1:].unsqueeze(-1)) ** 2) / (2 * sigma ** 2)
    heatmap = torch.exp(-dist)

    return heatmap
