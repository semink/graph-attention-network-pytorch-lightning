from model.GAT.layer import GraphAttentionLayer, Attention
import torch
import torch.nn as nn

# test input output shape
def test_model_generation():
    assert nn.Sequential(GraphAttentionLayer(encoder = nn.Linear(8, 8, bias=False), 
                                                     attention = Attention(8), 
                                                     num_heads = 8, 
                                                     nonlinear = nn.ELU()),
                                 GraphAttentionLayer(encoder = nn.Linear(8, 8, bias=False), 
                                                     attention = Attention(8), 
                                                     num_heads = 1, 
                                                     nonlinear = nn.Softmax()))
    
def test_forward_path_with_batching():
    x = torch.randn(2, 10, 8)
    GAT = nn.Sequential(GraphAttentionLayer(encoder = nn.Linear(8, 8, bias=False), 
                                                     attention =  Attention(8), 
                                                     num_heads = 8, 
                                                     nonlinear = nn.ELU()),
                                 GraphAttentionLayer(encoder = nn.Linear(8, 8, bias=False), 
                                                     attention = Attention(8), 
                                                     num_heads = 1, 
                                                     nonlinear = nn.Softmax()))
    assert GAT(x).shape == (2, 10, 8)
    
def test_forward_path_without_batching():
    x = torch.randn(10, 8)
    GAT = nn.Sequential(GraphAttentionLayer(encoder = nn.Linear(8, 8, bias=False), 
                                                     attention = Attention(8), 
                                                     num_heads = 8, 
                                                     nonlinear = nn.ELU()),
                                 GraphAttentionLayer(encoder = nn.Linear(8, 8, bias=False), 
                                                     attention = Attention(8), 
                                                     num_heads = 1, 
                                                     nonlinear = nn.Softmax()))
    assert GAT(x).shape == (10, 8)