from model.GAT.model import GraphAttentionNetwork
import torch
import torch.nn as nn
import numpy as np

def permutation_matrix(N):
    I = np.identity(N)
    P = np.empty((N,N))
    mid = N//2 if N % 2 == 0 else N//2+1
    P[0::2] = I[:mid]
    P[1::2] = I[mid:]
    return P

# test input output shape
def test_model_generation():
    assert GraphAttentionNetwork(features = [8, 8, 8], num_heads = [8, 1], nonlinear = [nn.ELU(), nn.Softmax()]) is not None
    
def test_forward_path_with_batching():
    B, N, F_in = 2, 10, 8
    F_out = 3
    x = torch.randn(B, N, F_in)
    GAT = GraphAttentionNetwork(features = [F_in, 8, F_out], num_heads = [8, 1], nonlinear = [nn.ELU(), nn.Softmax()])
    assert GAT(x).shape == (2, N, F_out)
    

def test_permutation_invariance():
    B, N, F_in = 2, 10, 8
    F_out = 3
    x = torch.randn(B, N, F_in)
    P = torch.from_numpy(permutation_matrix(N)).float()
    mask = torch.bernoulli(torch.empty(N, N).uniform_(0, 1))
    GAT = GraphAttentionNetwork(features = [F_in, 8, F_out], num_heads = [8, 1], nonlinear = [nn.ELU(), nn.Softmax()])
    assert torch.all(torch.isclose(torch.einsum('nm,...mf->...nf', P,GAT(x, mask)), 
                                   GAT(torch.einsum('nm,...mf->...nf', P, x), P@mask@P.T)))
    