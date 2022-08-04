import os
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import Counter, OrderedDict

from nets.attention_model_ours import transformer
from define_graph import graph_structure

problem = 'route'
embedding_dim = 128
hidden_dim = 256
n_heads = 1
n_layers = 3
normalization = 'batch'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 100
graph_inst = graph_structure()
prob_name='Lee_24'

x, mask_adj = graph_inst._getgraph() 
x = x.repeat(3,1,1)

TF = transformer(problem,
                 embedding_dim,
                 hidden_dim,
                 n_heads,
                 n_layers,
                 normalization,
                 device, 
                 mask_adj,
                 prob_name)

best_cost = 1000
cost_array = []

best_cost_taken_list = [0 for i in range(x.size(1))]  # define baseline costs

for epoch in range(1):
    cost_, cost_taken_list =  TF.update(x, mask_adj, epoch, best_cost_taken_list)
    
    if(cost_ < best_cost):
        print('Saving model')
        TF.save_model(prob_name= prob_name)
        best_cost = cost_
        best_cost_taken_list = cost_taken_list

    cost_array.append(cost_)

cost_dict = Counter(cost_array)
cost_dict = dict(cost_dict)

cost_dict = OrderedDict(sorted(cost_dict.items()))

print('Histogram of costs:', cost_dict)
print('best-cost:', best_cost)







