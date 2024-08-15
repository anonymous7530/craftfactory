
import numpy as np
import torch
import torch.nn as nn
import math


''' compositional generalization baselines '''

# ---------- baseline one permutation group, RNNENcoder only ---------------- #

# ---------------------------- symmetry_group ------------------------------- #

def get_permutation_matrix(num_letters, permutations, device="cuda"):
    ''' permutations: (i, j) i is raw, j is column id '''
    perm_mat = np.zeros((num_letters, num_letters))
    for perm in permutations:
        perm_mat[perm[1] - 1, perm[0] - 1] = 1.

    row_sums, columns_sums = perm_mat.sum(1), perm_mat.sum(0)
    for j in range(num_letters):
        if row_sums[j] == 0:
            perm_mat[j, j] = 1.
        if columns_sums[j] == 0:
            perm_mat[j, j] = 1.
    assert np.linalg.det(perm_mat) in [1., -1], "Determinant != +-1"
    return torch.tensor(perm_mat).to(device)


class PermutationSymmetry:

    def __init__(self, num_letters, device="cuda"):
        self.num_letters = num_letters
        self.e = torch.eye(self.num_letters, dtype=torch.float64).to(device)
        self.perm_matrices = [self.e]

    def in_group(self, perm_matrix):
        return np.sum([torch.allclose(perm_matrix, mat) 
                       for mat in self.perm_matrices]) > 0

    @property
    def size(self):
        return len(self.perm_matrices)
    
    def mat2index(self, mat):
        if not self.in_group(mat):
            return None
        return [i for i, perm_mat in enumerate(self.perm_matrices) 
                if torch.allclose(mat, perm_mat)][0]
    
    @property
    def learnable(self):
        return False
    

class CircularShift(PermutationSymmetry):

    def __init__(self, num_letters, num_equivariant, first_equivariant=0,
                 device='cuda'):
        super(CircularShift, self).__init__(num_letters, device)
        self.device = device
        self.num_equivariant = num_equivariant
        self.first_equivariant = first_equivariant
        self.last_equivariant = first_equivariant + num_equivariant

        self.init_perm = [(i, i + 1) for i in range(self.first_equivariant,
                                                    self.last_equivariant - 1)]
        self.init_perm += [(self.last_equivariant - 1, self.first_equivariant)]
        self.tau1 = get_permutation_matrix(self.num_letters, self.init_perm, self.device)
        self.perm_matrices.append(self.tau1)
        for _ in range(self.num_equivariant - 2):
            perm_mat = self.perm_matrices[-1] @ self.tau1
            self.perm_matrices.append(perm_mat)

        self.index2mat, self.index2inverse, self.index2inverse_indices = {}, {}, {},
        for idx, mat in enumerate(self.perm_matrices):
            self.index2mat[idx] = mat
            self.index2inverse[idx] = torch.pinverse(mat) 
            inverse_list = []
            for h in self.perm_matrices:
                temp = self.index2inverse[idx] @ h
                temp_idx = self.mat2index(temp)
                inverse_list.append(temp_idx)
            self.index2inverse_indices[idx] = torch.tensor(
                [self.mat2index(self.index2inverse[idx] @ h) for h in self.perm_matrices],
                dtype=torch.long
            ).to(self.device)
        self.to_float32()

    def to_float32(self):
        
        for idx, mat in enumerate(self.perm_matrices):
            self.index2mat[idx] = self.index2mat[idx].float()
            self.index2inverse[idx] = self.index2inverse[idx].float()

# -------------------------------- g_layers --------------------------------- #

class WordConv(nn.Module):

    def __init__(self, symmetry_group, vocabulary_size, embedding_size):
        super(WordConv, self).__init__()
        self.symmetry_group = symmetry_group
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)

    def forward(self, input):
        ''' input: 1-hot B x V
            output: B x |G| x K '''
        word_permutations = self.permute_word(input)
        permutation_indices = torch.argmax(word_permutations, dim=-1)
        return self.embedding(permutation_indices).squeeze()

    def permute_word(self, word):
        return  torch.stack([word @ self.symmetry_group.index2inverse[i]
                             for i in range(self.symmetry_group.size)], dim=1)
    

class GroupConv(nn.Module):

    def __init__(self, symmetry_group, input_size, embedding_size):
        super(GroupConv, self).__init__()
        self.symmetry_group = symmetry_group
        self.input_size = input_size
        self.embedding_size = embedding_size

        self.weights = nn.Parameter(torch.Tensor(self.symmetry_group.size, 
                                                 input_size, embedding_size))
        self.bias = nn.Parameter(1e-1 * torch.ones(self.embedding_size))

    def forward(self, input):
        ipt = input[:, None, ..., None]
        conv_filter = self.get_conv_filter()[None, ...]
        return (ipt * conv_filter).sum(2).sum(2) + self.bias


    def get_conv_filter(self):
        return torch.stack([torch.index_select(self.weights, 0, 
                            self.symmetry_group.index2inverse_indices[g])
                            for g in range(self.symmetry_group.size)])
    
    def _init_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_size)
        torch.nn.init.uniform_(self.weights, -stdv, stdv)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -bound, bound)

# ---------------------------------- g_cell --------------------------------- #

class GRecurrentCell(nn.Module):

    def __init__(self, symmetry_group, hidden_size, nonlinearity, 
                 device='cuda'):
        super(GRecurrentCell, self).__init__()
        self.symmetry_group = symmetry_group
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.device = device
        assert nonlinearity in ['tanh', 'relu'], "not one of 'tanh' or 'relu'"
    
    def forward(self, input, hidden):
        raise NotImplementedError
    
    def init_hidden(self):
        return torch.zeros(1, self.symmetry_group.size, self.hidden_size,
                           device=self.device)


class GRNNCell(GRecurrentCell):

    def __init__(self, symmetry_group, hidden_size, nonlinearity='tanh',
                 device='cuda'):
        super(GRNNCell, self).__init__(symmetry_group, hidden_size, 
                        nonlinearity=nonlinearity, device=device)
        self.psi_w = GroupConv(symmetry_group=self.symmetry_group,
                               input_size=self.hidden_size,
                               embedding_size=self.hidden_size)
        self.psi_h = GroupConv(symmetry_group=self.symmetry_group,
                               input_size=self.hidden_size,
                               embedding_size=self.hidden_size)
        self.activation = nn.Tanh() if nonlinearity == 'tanh' else nn.ReLU()

    def forward(self, input, hidden):
        f = self.psi_w(input)
        h = self.psi_h(hidden)
        return self.activation(f + h)

# ----------------------------------- g_rnn --------------------------------- #

class EquiRNN(nn.Module):

    def __init__(self, symmetry_group, input_size, hidden_size, cell_type,
                 bidirectional=False, nonlinearity='tanh', device='cuda'):
        super(EquiRNN, self).__init__()
        self.symmetry_group = symmetry_group
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.nonlinearity = nonlinearity
        self.device = device
        assert nonlinearity in ['tanh', 'relu'], "not one of 'tanh' or 'relu"

        self.embedding = WordConv(symmetry_group, vocabulary_size=self.input_size,
                                  embedding_size=self.hidden_size)
        if self.cell_type == 'GRNN':
            self.cell = GRNNCell(symmetry_group=self.symmetry_group,
                                 hidden_size=self.hidden_size,
                                 nonlinearity=self.nonlinearity, device=device)
        else:
            raise NotImplementedError('Currently only supports GRNN,')
        
    def forward(self, sequence, h_0=None):  # B, T, N
        embedded_sequence = self.embedding(sequence.squeeze())
        embedded_sequence = embedded_sequence.permute(2, 0, 1, 3)
        hidden_all, ht = self.forward_pass(embedded_sequence, h_0)
        return hidden_all, ht


    def forward_pass(self, sequence, h_0=None):
        h = self.init_hidden() if h_0 is None else h_0
        seq_length = sequence.shape[0]
        hidden_all = torch.zeros(seq_length, self.symmetry_group.size, 
                                 self.hidden_size).to(self.device)
        
        for t, element in enumerate(sequence):
            # element: [10, 10, 32], h: [1, 10, 32]
            h = self.cell(element[None, ...], h)
            state = h[0] if self.cell_type == 'GLSTM' else h
            hidden_all[t] = state.squeeze()
        return hidden_all, h


    def init_hidden(self):
        return self.cell.init_hidden()


class EquivariantRnn(nn.Module):
    

    def __init__(self, hidden_size, num_emb):
        super(EquivariantRnn, self).__init__()

        index_len = 9
        self.eye_mat = nn.Parameter(
            torch.eye(num_emb, num_emb).float(), 
            requires_grad=False,
        ) 
        self.adapt_layer = nn.Linear(num_emb, hidden_size)
        self.adapt_ly2 = nn.Linear(num_emb * index_len, hidden_size)

        self.final_layer = nn.Linear(hidden_size * index_len, hidden_size)

        self.num_layers = 2
        self.cell = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, 
                           num_layers=self.num_layers)
        self.rnn_init = nn.Parameter(
            torch.zeros((self.num_layers, hidden_size)), requires_grad=True)
        
    def forward(self, sequence):
        onehot = self.eye_mat[sequence]  # [B, T, N, D_0]
        B, T, N, _ = onehot.shape
        raw_emb = self.adapt_ly2(onehot.reshape(B, T, -1))
        emb = self.adapt_layer(onehot) # [B, T, N, D]

        hidden_all = self.forward_pass(emb)
        feat_perm = self.final_layer(hidden_all.reshape(B, T, -1))

        return raw_emb * (1 + torch.relu(feat_perm))

    def forward_pass(self, sequence, h_0=None): 
        emb = sequence.permute(2, 0, 1, 3)  # [Tc, B, T, D]
        Tc, B, T, D = emb.shape
        emb = emb.reshape(Tc, B*T, D)
        if getattr(self, 'h', None) is None:
            if h_0 is None:
                self.h = self.rnn_init
            else:
                self.h = h_0

        hidden_all, h = [], self.h
        for t, element in enumerate(emb):
            output, h = self.cell(element, h)
            hidden_all.append(output)

        output = torch.stack(hidden_all, dim=0)  # [Tc, B*T, D]
        output = output.reshape(Tc, B, T, D)
        return output.permute(1, 2, 0, 3)  # [B, T, Tc, D]
    
# --------------------------- test ------------------------------------------ #

def test_get_permutation_matrix():
    permutation = [[1, 3], [2, 8]]
    mat = get_permutation_matrix(num_letters=10, permutations=permutation)
    print(mat)
    

def test():
    import pdb
    pdb.set_trace()
    device = 'cuda'
    num_letters = 10
    num_equivariant=10
    symmetry_grouop = CircularShift(num_letters=num_letters, 
                                    num_equivariant=num_equivariant,
                                    first_equivariant=0,
                                    device=device)
    rnn = EquiRNN(symmetry_group=symmetry_grouop,
                  input_size=num_equivariant,  # voca_size
                  hidden_size=32,
                  cell_type='GRNN',
                  bidirectional=False,
                  nonlinearity='tanh',
                  device=device).to(device)
    eye_mat = torch.eye(n=num_letters, device=device)
    B, T = 2, 5
    x = torch.randint(0, 9, (B, T)).to(device)
    input = eye_mat[x]
    output, hidden = rnn(input, h_0=None)
    print(output.shape)

def test_equivariant_rnn():
    device = 'cuda'
    B, T, N, D = 2, 5, 9, 512
    cond = torch.randint(0, 630, (B, T, N)).to(device)

    model = EquivariantRnn(hidden_size=D, num_emb=631).to(device)
    result = model(cond)
    print(result.shape)


if __name__ == '__main__':
    # test_get_permutation_matrix()
    # test()
    test_equivariant_rnn()
