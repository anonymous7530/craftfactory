import torch
import torch.nn as nn

from typing import Tuple


# ---------------------- policy architecture -------------------------------- #

class Attention_Seqtovec(nn.Module):

    def __init__(self, input_dim, output_dim, num_heads, num_layers, 
                 dim_feedforward=512, emb_num = 631, dropout=0.):
        super(Attention_Seqtovec, self).__init__()

        self.index_len = 9

        self.eye_mat = nn.Parameter(
            torch.eye(emb_num, emb_num).float(), 
            requires_grad=False
        ) 
        self.adapt_layer = nn.Linear(emb_num, input_dim)
        self.adapt_ly2 = nn.Linear(emb_num * 9, output_dim)
        # Define the Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim, 
                nhead=num_heads, 
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        # Define a Linear layer for the final transformation
        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, x, feat=None):  # B, T, 9
        onehot = self.eye_mat[x]
        B, T, N, _ = onehot.shape
        raw_emb = self.adapt_ly2(onehot.reshape(B, T, -1))

        emb = self.adapt_layer(onehot)   # B, T, 9, D
        B, T, N, D = emb.shape
        emb = emb.reshape((-1, N, D))
        x = emb.permute(1, 0, 2)
        x = self.transformer(x)
        x = x[0]
        output_vector = self.fc(x)
        output_vector = output_vector.reshape(B, T, D)

        return raw_emb * (1 + torch.relu(output_vector))
    
def test():
    att = Attention_Seqtovec(512, 512, 1, 1, 
                             dim_feedforward=512,
                             emb_num=631)
    cond = torch.randint(0, 630, (2, 5, 9))
    vec = att(cond)
    print(vec.shape)



# ------------------ Toward Compositional Generalization ------------------------ #
    

def build_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)
    

class SoftPositionEmbed(nn.Module):

    def __init__(self, hidden_size: int, emb_num=631,
                resolution: Tuple[int, int] = [5, 9],):
        super().__init__()

        self.index_len = 9

        self.eye_mat = nn.Parameter(
            torch.eye(emb_num, emb_num).float(), 
            requires_grad=False,
        )
        
        self.adapt_layer = nn.Linear(emb_num, hidden_size)
        self.adapt_ly2 = nn.Linear(emb_num * 9, hidden_size)

        self.grid = nn.Parameter(build_grid(resolution), requires_grad=False)
        self.dense = nn.Linear(in_features=self.grid.size(-1), out_features=hidden_size)
        self.final_layer = nn.Linear(hidden_size * 9, hidden_size)

        self.hidden_size = hidden_size
        self.resolution = resolution

    def forward(self, inputs):
        onehot = self.eye_mat[inputs]
        B, T, N, _ = onehot.shape
        raw_emb = self.adapt_ly2(onehot.reshape(B, T, -1))
        
        emb = self.adapt_layer(onehot) # [B, T, 9, D]
        B, T, N, D = emb.shape
        assert self.resolution[0] >= T

        emb_proj = self.dense(self.grid) # [B, *resolu, D]
        combine = emb + emb_proj[:, :T, ...]
        result = self.final_layer(combine.reshape(B, T, -1))

        return raw_emb * (1 + torch.relu(result))


def test_grid():
    resolution = [5, 6]
    result = build_grid(resolution)
    print(result)  # [1, 10, 20, 4]
    

def test_wm():
    device = 'cuda'
    B, T, N, D = 2, 5, 9, 512
    cond = torch.randint(0, 630, (B, T, N)).to(device)

    soft_emb = SoftPositionEmbed(hidden_size=D, emb_num=631, 
        resolution=(T, N)).to(device)

    result = soft_emb(cond)
    print(result.shape)


if __name__ == '__main__':
    # test()
    # test_grid()
    test_wm()