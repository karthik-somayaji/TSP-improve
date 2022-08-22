import os
import time
import random
import numpy as np

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import Counter, OrderedDict
import argparse

from nets.attention_model_ours import transformer
from define_graph import graph_structure
from nets.DQN_simple import DQN_agent
from problems.problem_lee1 import router

parser = argparse.ArgumentParser(description="RL Comb-Opt")
parser.add_argument("-p", "--prob", type=str, default="syn_4_8")
parser.add_argument("-n", "--num_layers", type=int, default=3)
parser.add_argument("-e", "--embedding_dim", type=int, default=256)
parser.add_argument("-hi", "--hidden_dim", type=int, default=256)
parser.add_argument("-t", "--num_epochs", type=int, default=500)
parser.add_argument("-b", "--batch_sz", type=int, default=1)

args = parser.parse_args()

if('syn' in args.prob):
    n = 4
    m = int(args.prob[6:])
    total_steps = m//2

agent = DQN_agent(state_dim = total_steps*total_steps, action_dim = total_steps)


def reset_state(total_steps):
    return np.zeros((total_steps, total_steps))

def calc_next_state(state, indx, total_steps, curr_time_step):
    st = state.numpy().reshape(total_steps, -1)

    # make a one hot encoded vector for the net to route
    one_hot = np.zeros((total_steps, 1))
    one_hot[indx, 0] = 1.0

    st[:, curr_time_step] = one_hot[:,0]

    return st.reshape(1,-1)

def eval_model(n,m, prob, total_steps, agent):
    router_inst = router(n, m, prob)
    routed_lst = []

    state = torch.Tensor(reset_state(total_steps).reshape(1, -1))

    for step in range(total_steps):
        action = agent.select_action(state, epoch, infer=True)
        routed_lst.append(action[0,0].item())

        done = 1.0 if (step==total_steps-1) else 0.0

        next_state = calc_next_state(state, action[0,0].item(), total_steps, step)

        cost = router_inst.calc_cost(i=action[0,0].item(), routed_lst = routed_lst)

        state = torch.tensor(next_state)

    return cost

infer_cost = []
for epoch in range(args.num_epochs):
    router_inst = router(n, m, args.prob)
    routed_lst = []

    state = torch.Tensor(reset_state(total_steps).reshape(1, -1))

    for step in range(total_steps):
        action = agent.select_action(state, epoch, infer=False)
        routed_lst.append(action[0,0].item())

        #print('state:', state.reshape(4,-1))

        #print('action:', action[0,0].item())

        done = 1.0 if (step==total_steps-1) else 0.0

        next_state = calc_next_state(state, action[0,0].item(), total_steps, step)

        print(routed_lst)
    
        cost = router_inst.calc_cost(i=action[0,0].item(), routed_lst = routed_lst)

        print('Epoch:', epoch, 'Step:', step, 'Net:', action[0,0].item(), 'cost:', cost)

        obs_tuple = (state.numpy()[0,:], [action[0,0].item()] , -cost, next_state[0,:],  1-float(done))
        agent.memory(obs_tuple)

        state = torch.tensor(next_state)

        agent.train()
    infer_cost.append(eval_model(n,m, args.prob, total_steps, agent))
    print("#"*80)

file_name = args.prob + '.txt'
np.savetxt(file_name, infer_cost)
    
    

    











