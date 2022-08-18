import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import math
from utils import position_encoding


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            mask,
            embed_dim=None,
            val_dim=None,
            key_dim=None,
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.mask = mask

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        mask = self.mask

        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        #assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        #print('q_size', q.size())

        batch_size = 1

        hflat = h.contiguous().view(-1, input_dim) #################   reshape
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)   
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
            

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility).long()
            #print(mask.view(20,20), compatibility.view(20,20))
            #compatibility[mask] = -np.inf
            compatibility = torch.multiply(compatibility, mask)
            compatibility[compatibility==0.0] = -np.inf

        attn = F.softmax(compatibility, dim=-1)   
      

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            #attnc[mask] = 0
            attnc = torch.multiply(attn, mask)
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out
    
    
class MultiHeadDecoder(nn.Module):
    def __init__(
            self,
            problem,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadDecoder, self).__init__()
        
        self.problem = problem
        self.n_heads = 1

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // self.n_heads
        if key_dim is None:
            key_dim = val_dim
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.heads = 1

        self.W_context = nn.Parameter(torch.Tensor(self.heads, 1, 2))  # to convert concatenated vector to single vector format
        self.W_graph = nn.Parameter(torch.Tensor(self.heads, 1, 2))  # to convert concatenated vector to single vector format

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(input_dim, key_dim))
        
        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, l, context, g, mask, is_random, random_net):
        """

        :q - (batch_size, nodes, embed_dim)
        :l - (batch_size, 1, embed_dim)
        :context - (batch_size, 1, embed_dim)
        :g - (batch_size, 1, embed_dim)
        :mask- (batch_size, nodes)
        :is_random - (batch_size, 1)
        :random_net - (batch_size, 1)

        :return: q_max (batch_size, 1, embed_dim); updated_mask(1, batch_size), likelihood(batch_size, 1)
        """

        mask_copy = torch.Tensor.clone(mask)

        h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        #assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        l_context_concat = torch.cat((l, context), axis=1)  # (batch_size, 2, embed_dim)
        l_context_concat_flat = l_context_concat.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (batch_size, graph_size, -1)
        shp_q = (batch_size, 1, -1)
        shp_concat = (batch_size, 1, input_dim)

        concat = torch.matmul( self.W_context, l_context_concat_flat).view(shp_concat)
        g_context_concat = torch.cat((g, concat), axis=1)  # (batch_size, 2, embed_dim)
        g_context_concat_flat = g_context_concat.contiguous().view(-1, input_dim)

        g_context = torch.matmul( self.W_graph, g_context_concat_flat).view(shp_concat)

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)
        g_contextflat = g_context.contiguous().view(-1, input_dim)

        # Calculate queries,(batch_size, 1, key/val_size)
        Q = torch.matmul(g_contextflat, self.W_query).view(shp_q)
        # Calculate keys and values (batch_size, graph_size, key/val_size)
        K = torch.matmul(qflat, self.W_key).view(shp)
        #V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility ( batch_size, 1, graph_size)
        compatibility_raw = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))
        compatibility = torch.tanh(compatibility_raw) * 10.  # (batch_size, 1, graph_size)

        mask = mask.float()[:,None,:]  # add extra dimension to match compatibilty's dimension
        compatibility[mask<=0.1] = -np.inf

       
        #masked_compatibility = torch.multiply(mask, compatibility)  # (batch_size, 1 , graph_size)
        masked_compatibility = compatibility

        softMax = F.softmax(masked_compatibility, dim=2)  # (batch_size, 1 , graph_size)
        #print(softMax.squeeze(1).detach())
        log_softMax = F.log_softmax(masked_compatibility, dim=2)  # (batch_size, 1 , graph_size)
        if(not is_random):
            max_indx = torch.argmax(softMax, axis=2)  # (batch_size, 1)
        else:
            max_indx = random_net

        q_max = q[range(q.size(0)), max_indx.squeeze(),:][:,None,:]  # (batch_size, 1, embed_dim)
        attn_max = softMax[range(q.size(0)), :, max_indx.squeeze()].view(batch_size, 1)  # (batch_size, 1)
        log_attn_max = log_softMax[range(q.size(0)), :, max_indx.squeeze()].view(batch_size, 1)  # (batch_size, 1)

        #mask_copy[range(q.size(0)), max_indx.squeeze()] = 0.0  # update the masking matrix

        return q_max, attn_max, log_attn_max, concat, mask_copy, max_indx


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            mask,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim,
                    mask=mask
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )
                

class EmbeddingNet(nn.Module):
    
    def __init__(
            self,
            node_dim,
            embedding_dim,
            device
        ):
        super(EmbeddingNet, self).__init__()
        self.node_dim = node_dim
        self.embedding_dim = embedding_dim
        self.device = device
        self.embedder = nn.Linear(node_dim, embedding_dim)
        
    def forward(self, x, solutions):
        #pos_enc = position_encoding(solutions, self.embedding_dim, self.device)
        return self.embedder(x) #+ pos_enc.cpu()