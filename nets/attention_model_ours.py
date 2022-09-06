import torch
from torch import nn
from torch.optim import Adam
import numpy as np

import os
from problems.problem_lee1 import router

from nets.graph_layers_ours import MultiHeadAttentionLayer, MultiHeadDecoder, EmbeddingNet

class Policy(nn.Module):

    def __init__(self,
                 problem,
                 embedding_dim,
                 hidden_dim,
                 n_heads,
                 n_layers,
                 normalization,
                 device,
                 mask,
                 node_dim
                 ):
        super(Policy, self).__init__()

        self.problem = problem
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.normalization = normalization
        self.device = device
        self.mask = mask
        self.node_dim = node_dim

        # Problem specific placeholders
        
        '''if self.problem == 'route':
            self.node_dim = 10  # x, y
        else:
            assert False, "Unsupported problem: {}".format(self.problem.NAME)'''
        
        

        # networks
        self.embedder = EmbeddingNet(
                            self.node_dim,
                            self.embedding_dim,
                            self.device)
        
        self.encoder = nn.Sequential(*(
            MultiHeadAttentionLayer(self.n_heads, 
                                    self.embedding_dim, 
                                    self.hidden_dim,
                                    self.mask, 
                                    self.normalization)
            for _ in range(self.n_layers))) 
            
            
        self.project_graph = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        
        self.project_node = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)  
        
        self.decoder = MultiHeadDecoder(self.problem,
                                        input_dim = self.embedding_dim, 
                                        embed_dim = self.embedding_dim)    
            
        
    def forward(self, x, curr_ordering, mask):#, last_node, context, mask_blocked, mask_adj, is_random, random_net):#solutions, exchange, do_sample = False, best_solutions = None):
        """
        :param x: (batch_size, graph_size, node_dim) input node features
        last_node: (batch_size, 1, embed_dim)
        context: (batch_size, 1, embed_dim)
        mask_blocked:(batch_size, nodes)
        mask_adj: (nodes, nodes)
        is_random: (batch_size, 1)
        random_net: (batch_size, 1)
        :return:
        q_max (batch_size, 1, embed_dim); updated_mask( batch_size, nodes), attn_max & log_attn_mask-likelihood(batch_size, 1), concat (batch_size,1,embed_dim)
        """         
         
        # the embedded input x
        x = x.repeat(1,1,1)
        bs, gs, in_d = x.size()

        #print(best_ordering.shape, bs, gs, in_d)

        x = self.embedder(x.squeeze(0), curr_ordering.expand(bs, gs))
        x = x.view(bs, gs, self.embedding_dim)
        
        # pass through embedder: current solutions
        #x_embed = self.embedder(x, solutions)
        
        # pass through encoder: current solutions
        #mask_adj = mask_adj.repeat(bs, 1, 1)
        h_em = self.encoder(x) # (batch_size, nodes, embed_dim) 

        g = h_em.mean(1).view(bs, 1, self.embedding_dim)
        g_fused = g + h_em  # add mean representation of graph with every node

        attn_max, log_attn_mask = self.decoder(g_fused, mask)  # (batch_size, nodes, nodes)

        return attn_max, log_attn_mask


class Critic(nn.Module):
    def __init__(self,
                 problem,
                 embedding_dim,
                 hidden_dim,
                 n_heads,
                 n_layers,
                 normalization,
                 device,
                 mask,
                 node_dim
                 ):
        super(Critic, self).__init__()

        self.problem = problem
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = 2
        self.normalization = normalization
        self.device = device
        self.mask = mask
        self.node_dim = node_dim

        # networks
        self.embedder = EmbeddingNet(
                            self.node_dim,
                            self.embedding_dim,
                            self.device)
        
        self.encoder = nn.Sequential(*(
            MultiHeadAttentionLayer(self.n_heads, 
                                    self.embedding_dim, 
                                    self.hidden_dim, 
                                    self.normalization)
            for _ in range(self.n_layers)))
        
        self.project_graph = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        
        self.project_node = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False) 
        
        self.value_head = nn.Sequential(
                nn.Linear(embedding_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1)
            ) 
            
    def forward(self, x, solutions):
        """
        :param inputs: (x, graph_size, input_dim)
        :return:
        """
        bs, gs, in_d = x.size()
        # pass through embedder
        x_embed = self.embedder(x, solutions)
        
        # pass through encoder
        h_em = self.encoder(x_embed) # torch.Size([2, 50, 128])
        
        # get embed feature
        max_pooling = h_em.max(1)[0].view(bs, 1, self.embedding_dim)   # mean Pooling
        graph_feature = self.project_graph(max_pooling)#[:, None, :]
        
        # get embed node feature
        node_feature = self.project_node(h_em) 

        # pass through value_head, get estimated value
        fusion = node_feature + graph_feature.expand_as(node_feature) # torch.Size([2, 50, 128])
        value = self.value_head(fusion.mean(dim=1))

        return value


class transformer():
    def __init__(self, problem,
                 embedding_dim,
                 hidden_dim,
                 n_heads,
                 n_layers,
                 normalization,
                 device,
                 mask,
                 prob_type,
                 node_dim,
                 batch_sz,
                 graph_sz):
        self.problem = problem
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.normalization = normalization
        self.device = device
        self.mask = mask
        self.prob_type = prob_type
        self.node_dim = node_dim
        self.batch_sz =  batch_sz
        self.graph_sz = graph_sz

        self.attn_model = Policy(self.problem, self.embedding_dim, self.hidden_dim, self.n_heads, self.n_layers, self.normalization, self.device, self.mask, self.node_dim)
        self.critic = Critic(self.problem, self.embedding_dim, self.hidden_dim, self.n_heads, self.n_layers, self.normalization, self.device, self.mask, self.node_dim)
        self.critic_target = Critic(self.problem, self.embedding_dim, self.hidden_dim, self.n_heads, self.n_layers, self.normalization, self.device, self.mask, self.node_dim)

        self.hard_update(self.critic_target, self.critic)  # make the baseline equal to the current model

        self.optimizer_policy = Adam(self.attn_model.parameters(), lr=1e-4)
        self.optimizer_critic = Adam(self.critic.parameters(), lr=1e-5)

        if('Lee' in self.prob_type):
            self.m = self.n = int(self.prob_type[4:])

        elif('syn' in self.prob_type):
            self.n = 4
            self.m = int(self.prob_type[6:])
 
        elif('custom_8_4_20' in self.prob_type):
            self.n=8
            self.m=28  

        self.eps = 1.0
        self.best_cost = 1000

        self.baseline = 1500*torch.ones(self.batch_sz*self.graph_sz,1)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def update(self, mem_critic_val, mem_attn, mem_cost):

        cum_mem_cost = torch.flip(torch.cumsum(torch.flip(mem_cost, [0]), 0), [0])
        # cum_sum is doing this: [1,2,3,4,5] -> [15, 14, 12, 9, 4] (cumulative sum in reverse direction)

        '''self.optimizer_critic.zero_grad()
        criteria = torch.nn.MSELoss()
        val_loss = criteria(mem_critic_val, cum_mem_cost).mean().mean()
        val_loss.backward()
        self.optimizer_critic.step()'''

        self.optimizer_policy.zero_grad()
        #policy_loss = (mem_cost - mem_critic_val.detach())* (-mem_attn)
        policy_loss = (mem_cost ) * (-mem_attn)
        policy_loss = policy_loss.mean().mean()
        policy_loss.backward()
        self.optimizer_policy.step()

        print(self.eps)

    def take_action(self,x, best_ordering, mask, is_infer):
        
        rand_num = np.random.uniform(0,1)

        attn_max, log_attn_max = self.attn_model(x, best_ordering, mask)

        if(not is_infer):
            self.eps = max(0.1, self.eps-0.005)

        if((rand_num > self.eps) or (is_infer)):  # epsilon greedy policy
            #  Take greedy action
            is_random = False
            r = np.random.choice(np.arange(self.graph_sz*self.graph_sz), 1, p=attn_max.detach().flatten().numpy())
            max_indx = np.array([r//self.graph_sz, r%self.graph_sz])
            #max_indx = (attn_max[0]==torch.max(attn_max[0])).nonzero()[0].numpy()

        else:
            is_random = True
            net = [i for i in range(self.graph_sz)]
            max_indx = np.random.choice(net, 2, replace=False)
            #q_max, attn_max, log_attn_mask, concat, mask_updated, max_indx = self.attn_model(x, best_ordering, last_node, context, mask_blocked, mask_adj, is_random, random_net)

        attn_max = attn_max[0, max_indx[0], max_indx[1]].reshape(1,1)
        log_attn_max = log_attn_max[0, max_indx[0], max_indx[1]].reshape(1,1)

        print(self.eps)
        return max_indx, attn_max, log_attn_max

    def save_model(self, prob_name):
        results_dir = "saved_models"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        path = "{}/{}_transformer".format(results_dir, prob_name)
        torch.save(self.attn_model.state_dict(), path)

    def load(self, prob_name):
        results_dir = "saved_models"
        path = "{}/{}_transformer".format(results_dir, prob_name)
        self.attn_model.load_state_dict(torch.load(path))
        














