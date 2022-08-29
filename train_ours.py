import os
import time
import random
import numpy as np

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import Counter, OrderedDict
import argparse

from nets.attention_model_inst import transformer_inst
from nets.attention_model_Q import transformer_Q
from define_graph import graph_structure

parser = argparse.ArgumentParser(description="RL Comb-Opt")
parser.add_argument("-p", "--prob", type=str, default="syn_4_36")
parser.add_argument("-n", "--num_layers", type=int, default=3)
parser.add_argument("-e", "--embedding_dim", type=int, default=256)
parser.add_argument("-hi", "--hidden_dim", type=int, default=256)
parser.add_argument("-t", "--num_epochs", type=int, default=1500)
parser.add_argument("-b", "--batch_sz", type=int, default=1)
parser.add_argument("-a", "--algorithm", type=str, default='inst')
parser.add_argument("-tr", "--trial", type=int, default=0)


args = parser.parse_args()


problem = 'route'
embedding_dim = args.embedding_dim
hidden_dim = args.hidden_dim
n_heads = 1
n_layers = args.num_layers#7
normalization = 'batch'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_sz = args.batch_sz

num_epochs = args.num_epochs
prob_name=args.prob
graph_inst = graph_structure()

x, mask_adj = graph_inst._getgraph(type_fn = prob_name)
x = x.repeat(batch_sz,1,1)

bs, gs, ns = x.size()
node_dim = ns
graph_sz = gs

if(args.algorithm=='inst'):
    TF = transformer_inst(problem,
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
                    graph_sz,
                    args.trial)

elif(args.algorithm=='Q'):
    TF = transformer_Q(problem,
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
                    graph_sz,
                    args.trial)


best_cost = 3000
cost_array = []

best_cost_taken_list = [best_cost for i in range(x.size(1))]  # define baseline costs

ordering = [i for i in range(gs)]
random.shuffle(ordering)
ordering = torch.Tensor([ordering]).view(1,-1)

for epoch in range(num_epochs):
    cost_, cost_taken_list, curr_ordering =  TF.update(x, mask_adj, epoch, best_cost_taken_list, ordering)

    curr_ordering_list = curr_ordering[0,:].tolist()
    x_lst = curr_ordering_list.copy()
    #y_lst = [i+(gs//2) for i in x_lst]
    
    if(cost_ <= best_cost):
        print('Saving model')
        TF.save_model(prob_name= prob_name, trial=str(args.trial))
        best_cost = cost_

        ordering = torch.tensor([x_lst]).view(1, -1)
        best_cost_taken_list = cost_taken_list

    y, mask_adj = graph_inst._getgraph(type_fn = prob_name)
    y = y.repeat(1,1,1)
    infer_cost = TF.infer_rollout(y, mask_adj, ordering, epoch)

    cost_array.append(infer_cost)

file_name = 'saved_files/' + prob_name + '_alg:'  + args.algorithm + '_' + str(args.trial) + '.txt'
np.savetxt(file_name, cost_array)

cost_dict = Counter(cost_array)
cost_dict = dict(cost_dict)

cost_dict = OrderedDict(sorted(cost_dict.items()))

print('Histogram of costs:', cost_dict)
print('best-cost:', best_cost)







