import torch
from torch import nn
from torch.optim import Adam
import numpy as np
import os
from problems.problem_lee1 import router

from nets.graph_layers_ours import MultiHeadAttentionLayer, MultiHeadDecoder, EmbeddingNet

class AttentionModel(nn.Module):

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
        super(AttentionModel, self).__init__()

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
            
        
    def forward(self, x, best_ordering, last_node, context, mask_blocked, mask_adj, is_random, random_net):#solutions, exchange, do_sample = False, best_solutions = None):
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
        bs, gs, in_d = x.size()

        #print(best_ordering.shape, bs, gs, in_d)

        x = self.embedder(x.squeeze(0), best_ordering.expand(bs, gs))
        x = x.view(1, gs, self.embedding_dim)
        
        # pass through embedder: current solutions
        #x_embed = self.embedder(x, solutions)
        
        # pass through encoder: current solutions
        mask_adj = mask_adj.repeat(bs, 1, 1)
        h_em = self.encoder(x) # (batch_size, nodes, embed_dim) 

        g = h_em.mean(1).view(bs, 1, self.embedding_dim)

        q_max, attn_max, log_attn_mask, concat, mask_updated, max_indx = self.decoder(h_em, last_node, context, g, mask_blocked, is_random, random_net)

        return q_max, attn_max, log_attn_mask, concat, mask_updated, max_indx

       
        '''# get embed graph feature
        max_pooling = h_em.max(1)[0] # max Pooling
        graph_feature = self.project_graph(max_pooling)[:, None, :]
    
        # get embed node feature
        node_feature = self.project_node(h_em)
        
        # pass through decoder, get log_likelihood and M matrix
        fusion = node_feature + graph_feature.expand_as(node_feature) # torch.Size([2, 50, 128])

        log_likelihood, M_value = self.decoder(fusion, exchange, solutions) # att torch.Size([1, 2, 2500])          
        
        # sample or select current pair for actions
        pair_index = M_value.multinomial(1) if do_sample else M_value.max(-1)[1].view(-1,1)
        
        selected_log_likelihood = log_likelihood.gather(1, pair_index)
        
        col_selected = pair_index % gs
        row_selected = pair_index // gs
        pair = torch.cat((row_selected,col_selected),-1)  # pair: no_head bs, 2
        
        return pair, selected_log_likelihood.squeeze()'''


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
                 node_dim):
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

        self.attn_model = AttentionModel(self.problem, self.embedding_dim, self.hidden_dim, self.n_heads, self.n_layers, self.normalization, self.device, self.mask, self.node_dim)
        self.attn_model_baseline = AttentionModel(self.problem, self.embedding_dim, self.hidden_dim, self.n_heads, self.n_layers, self.normalization, self.device, self.mask, self.node_dim)
        self.hard_update(self.attn_model_baseline, self.attn_model)  # make the baseline equal to the current model

        self.optimizer = Adam(self.attn_model.parameters(), lr=5e-5)

        if('Lee_8' in self.prob_type):
            self.m = self.n = 8
        elif('Lee_16' in self.prob_type):
            self.m = self.n = 16
        elif('Lee_24' in self.prob_type):
            self.m = self.n = 24
        elif('syn_4_20' in self.prob_type):
            self.n=4
            self.m=20
        elif('syn_4_16' in self.prob_type):
            self.n=4
            self.m=16
        elif('syn_4_12' in self.prob_type):
            self.n=4
            self.m=12           

        self.eps = 1.0

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def update(self, x, mask_adj, epoch, best_cost_taken, ordering):
        """
        :param x: (batch_size, graph_size, node_dim) input node features
        last_node: (batch_size, 1, embed_dim)
        context: (batch_size, 1, embed_dim)
        mask_blocked:(batch_size, nodes)
        mask_adj: (nodes, nodes)
        costs: (batch_size, 1)
        :return:
        q_max (batch_size, 1, embed_dim); updated_mask( batch_size, nodes), attn_max-likelihood(batch_size, 1), concat (batch_size,1,embed_dim)
        """ 

        self.ordering = ordering
        self.eps = max(0.1,  self.eps-0.001)
        costs, log_probs, last_cost, cost_taken_list, curr_ordering = self.rollout(x, self.ordering, mask_adj, epoch, best_cost_taken)
        self.ordering = curr_ordering.view(1, -1)

        #q_max, attn_max, log_attn_mask, concat, mask_updated, max_indx = self.attn_model(x, last_node, context, mask_blocked, mask_adj)

        # compute gradients
        self.optimizer.zero_grad()
        loss = torch.multiply(costs, log_probs)
        loss = loss.mean().mean()
        loss.backward()
        self.optimizer.step()

        print(self.eps)

        return last_cost, cost_taken_list, self.ordering

    def take_action(self,x, best_ordering, last_node, context, mask_blocked, mask_adj, epoch):
        bs, gs, in_d = x.size()

        rand_num = np.random.uniform(0,1)

        if(rand_num > self.eps):  # epsilon greedy policy

            #  Take greedy action
            is_random = False
            q_max, attn_max, log_attn_mask, concat, mask_updated, max_indx = self.attn_model(x, best_ordering, last_node, context, mask_blocked, mask_adj, is_random, [])

        else:
            is_random = True
            net = torch.where(mask_blocked > 0.0)[1]
            random_net = torch.tensor([[np.random.choice(net)]])
            q_max, attn_max, log_attn_mask, concat, mask_updated, max_indx = self.attn_model(x, best_ordering, last_node, context, mask_blocked, mask_adj, is_random, random_net)
        
        mask_updated = torch.Tensor.clone(mask_blocked) 
        mask_updated[range(bs), max_indx.squeeze()] = 0.0

        indx = max_indx.squeeze()
        source_sink_mask_indx = (indx + gs//2) if indx < gs//2 else indx-gs//2
        mask_updated[range(bs), source_sink_mask_indx] = 0.0  # also mask the complementary node

        return q_max, attn_max, log_attn_mask, concat, mask_updated, max_indx

    def rollout(self, x, best_ordering, mask_adj, epoch, best_cost_taken):
        bs, gs, in_d = x.size()

        cost_list = []
        prob_list = []

        for i in range(bs):
            step = 0
            routed_list = []

            router_int = router(self.n, self.m, self.prob_type)

            mask_blocked = torch.ones(1, gs)
            context = torch.zeros(1, 1, self.embedding_dim)
            last_node = torch.zeros(1, 1, self.embedding_dim)

            mask_blocked_bl = torch.ones(1, gs)
            context_bl = torch.zeros(1, 1, self.embedding_dim)
            last_node_bl = torch.zeros(1, 1, self.embedding_dim)

            cost_taken_list = []

            while(step < gs//2):

                inp = x[i,:,:].view(1, gs, in_d)

                q_max, attn_max, log_attn_max, concat, mask_updated, max_indx = self.take_action(inp, best_ordering, last_node, context, mask_blocked, mask_adj, epoch)
                #q_max_bl, attn_max_bl, log_attn_max_bl, concat_bl, mask_updated_bl, max_indx_bl = self.take_action(inp, last_node_bl, context_bl, mask_blocked_bl, mask_adj)

                mask_blocked = mask_updated
                last_node = q_max
                context = concat

                indx_to_route = max_indx.item()
                indx_to_route = (indx_to_route-gs//2) if (indx_to_route>=gs//2) else indx_to_route

                routed_list.append(indx_to_route)
                cost = router_int.calc_cost(i=indx_to_route, routed_lst = routed_list)  # placeholder to obtain the cost

                cost = 1000 if cost is None else cost

                print('Epoch:', epoch, 'Step:', step, 'Cost:', cost, 'Routed:', indx_to_route)

                cost_list.append(torch.Tensor([[cost - best_cost_taken[step]]]))
                prob_list.append(log_attn_max.view(1, -1))
                cost_taken_list.append(cost)

                step+=1
            last_cost = cost

        print("#"*80)

        curr_ordering = torch.tensor([routed_list])

        costs = torch.cat(cost_list).view(bs*(gs//2), 1)
        probs = torch.cat(prob_list).view(bs*(gs//2), 1)

        return costs, probs, last_cost, cost_taken_list, curr_ordering

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
        














