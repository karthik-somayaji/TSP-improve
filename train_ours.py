import os
import time
import random
import numpy as np

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import Counter, OrderedDict
import argparse

from problems.problem_lee1 import router
from nets.attention_model_ours import transformer
from define_graph import graph_structure

parser = argparse.ArgumentParser(description="RL Comb-Opt")
parser.add_argument("-p", "--prob", type=str, default="syn_4_36")
parser.add_argument("-n", "--num_layers", type=int, default=3)
parser.add_argument("-e", "--embedding_dim", type=int, default=128)
parser.add_argument("-hi", "--hidden_dim", type=int, default=256)
parser.add_argument("-t", "--num_epochs", type=int, default=200)
parser.add_argument("-st", "--num_steps", type=int, default=50)
parser.add_argument("-b", "--batch_sz", type=int, default=1)


args = parser.parse_args()


problem = 'route'
embedding_dim = args.embedding_dim
hidden_dim = args.hidden_dim
n_heads = 1
n_layers = args.num_layers#7
normalization = 'batch'
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
batch_sz = args.batch_sz

num_epochs = args.num_epochs
prob_name=args.prob
graph_inst = graph_structure()

x, mask_adj = graph_inst._getgraph(type_fn = prob_name)
x = x.repeat(batch_sz,1,1)

bs, gs, ns = x.size()
node_dim = ns
graph_sz = gs

TF = transformer(problem,
                 embedding_dim,
                 hidden_dim,
                 n_heads,
                 n_layers,
                 normalization,
                 device, 
                 mask_adj,
                 prob_name,
                 node_dim,
                 batch_sz,
                 graph_sz)

best_cost = 1500
cost_array = []

best_cost_taken_list = [best_cost for i in range(x.size(1))]  # define baseline costs

ordering = [i for i in range(gs)]
random.shuffle(ordering)
ordering = torch.Tensor([ordering]).view(1,-1)
inference_cost = []

def new_ordering(net, indx):
    a, b = net.index(indx[0]), net.index(indx[1])
    net[b], net[a] = net[a], net[b]
    return torch.tensor([net]).view(1,-1)

def infer_run():
    for epoch in range(1):
        ordering = [i for i in range(gs)]
        random.shuffle(ordering)
        ordering = torch.Tensor([ordering]).view(1,-1)
        state, mask_adj = graph_inst._getgraph(type_fn = prob_name)

        cost_arr = []
        prev_swap = np.array([0,0])

        for st in range(args.num_steps):
            #  Take Action
            mask = torch.ones(TF.graph_sz, TF.graph_sz) - torch.eye(TF.graph_sz, TF.graph_sz)
            mask[prev_swap[0], prev_swap[1]] = 0.0
            mask[prev_swap[1], prev_swap[0]] = 0.0
            max_indx, attn_max, log_attn_max = TF.take_action(state, ordering, mask, is_infer=True)
            prev_swap = max_indx

            #  Get next ordering
            next_ordering = new_ordering(ordering[0].tolist(), max_indx)

            #  Get associated cost
            router_int = router(TF.n, TF.m, TF.prob_type)
            cost = router_int.calc_cost(i=0, routed_lst = next_ordering[0].int().tolist())
            cost_arr.append(cost)

            #  Set current state to next state
            ordering = next_ordering

            #print('epoch:', epoch, 'step:', st, 'Current ordering:', ordering[0].tolist(), 'cost:', cost)
        cost_arr = np.array(cost_arr)
        return np.min(cost_arr)

for epoch in range(num_epochs):
    ordering = [i for i in range(gs)]
    random.shuffle(ordering)
    ordering = torch.Tensor([ordering]).view(1,-1)
    state, mask_adj = graph_inst._getgraph(type_fn = prob_name)

    mem_critic_val = []
    mem_cost = []
    mem_ordering = []
    mem_attn = []

    prev_swap = np.array([0,0])

    for st in range(args.num_steps):

        mem_ordering.append(ordering)
        value_critic = TF.critic(x, ordering)
        mem_critic_val.append(value_critic)

        #  Take Action
        mask = torch.ones(TF.graph_sz, TF.graph_sz) - torch.eye(TF.graph_sz, TF.graph_sz)
        mask[prev_swap[0], prev_swap[1]] = 0.0
        mask[prev_swap[1], prev_swap[0]] = 0.0
        max_indx, attn_max, log_attn_max = TF.take_action(state, ordering, mask, is_infer=False)
        mem_attn.append(log_attn_max)
        prev_swap = max_indx

        #  Get next ordering
        next_ordering = new_ordering(ordering[0].tolist(), max_indx)

        #  Get associated cost
        router_int = router(TF.n, TF.m, TF.prob_type)
        cost = router_int.calc_cost(i=0, routed_lst = next_ordering[0].int().tolist())
        mem_cost.append(torch.Tensor([[cost]]))

        #  Set current state to next state
        ordering = next_ordering
        print('epoch:', epoch, 'step:', st, 'Current ordering:', ordering[0].tolist(), 'cost:', cost)
    mem_critic_val = torch.stack(mem_critic_val)
    mem_attn = torch.stack(mem_attn)    
    mem_cost = torch.stack(mem_cost)
    
    TF.update(mem_critic_val, mem_attn, mem_cost)
    inference_cost.append(infer_run())

inference_cost = np.array(inference_cost)
np.savetxt(args.prob, inference_cost)



