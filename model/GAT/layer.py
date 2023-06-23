import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# clone pytorch module and store it to nn.ModuleList()
def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Attention(nn.Module):
    """Some Information about Attention"""
    def __init__(self, in_features, alpha=0.2):
        super(Attention, self).__init__()
        self.b1 = nn.Linear(in_features, 1, bias=False)
        self.b2 = nn.Linear(in_features, 1, bias=False)
        
        self.nonlinear = nn.LeakyReLU(alpha)
        
        
    def forward(self, x, mask=None):
        """_summary_

        Args:
            x (torch.Tensor): Graph signal (B, N, F_in)

        Returns:
            torch.Tensor: Attention score (B, N, N)
        """
        N = x.size(-2)
        e = self.nonlinear(self.b1(x).expand(-1, -1, N) + self.b2(x).transpose(-1,-2).expand(-1, N, -1))
        if mask is not None:
            e = e.masked_fill(mask, -1e9)
        return F.softmax(e, dim=-1)
    

class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer
    Graph attention layer is basically a self-attention of a vector that lies on a graph.
    Therefore, it only calculates the attention coefficients between nodes that are connected.
    In the implementation side, we first calculate the attention coefficients between all nodes 
    and mask out the coefficients between unconnected nodes.
    
    Example:
        >>> GAT = nn.Sequential(GraphAttentionLayer(encoder = nn.Linear(8, 8, bias=False), 
                                                     attention = Attention(8), 
                                                     num_heads = 8, 
                                                     nonlinear = nn.ELU()),
                                 GraphAttentionLayer(encoder = nn.Linear(8, 8, bias=False), 
                                                     attention = Attention(8), 
                                                     num_heads = 1, 
                                                     nonlinear = nn.Softmax())),
    """
    def __init__(self, encoder, attention, num_heads=1,
                 nonlinear = nn.ELU(),
                 permutation_invariant_aggregator = lambda a, h: torch.einsum('...nm,...mf->...nf', a, h),
                 ):
        super(GraphAttentionLayer, self).__init__()
        self.encoder =  clone(encoder, num_heads)
        self.attention = clone(attention, num_heads)
        self.nonlinear = nonlinear
        self.permutation_invariant_aggregator = permutation_invariant_aggregator
        
    def forward(self, x, mask=None):
        """_summary_

        Args:
            x (torch.Tensor): Graph signal (B, N, F_in)

        Returns:
            torch.Tensor: _description_
        """
        h = [encoder(x) for encoder in self.encoder]     # (B, N, F_out)
        a = [attention(x, mask) for attention in self.attention]   # (B, N, N)
        h = torch.mean(torch.stack([self.permutation_invariant_aggregator(a_, h_) for a_, h_ in zip(a, h)]), dim=0) # (B, N, F_out)
        x = self.nonlinear(h)
        return x