import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import Adam
import math

import sys
import os


class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs):
        super(Critic, self).__init__()
        self.low_value_mask = -1000
        self.action_space = num_outputs

        self.linear1 = nn.Linear(num_inputs, hidden_size) 
        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)

        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.ln4 = nn.LayerNorm(hidden_size)

        self.V = nn.Linear(hidden_size, num_outputs)
        #self.V.weight.data.mul_(0.01)
        #self.V.bias.data.mul_(0.01)

    def forward(self, inputs, infer):
        """
        mask: bs x num_nodes ; contains either 1 or -np.inf
        """
        bs, num_inp = inputs.size()
        num_nodes = int(np.sqrt(num_inp))

        x = inputs
        x = F.relu(self.ln1(self.linear1(x)))
        #x = F.relu(self.linear1(x))
        x = F.relu(self.ln2(self.linear2(x)))
        x = F.relu(self.ln3(self.linear3(x)))
        x = F.relu(self.ln4(self.linear4(x)))
        V = self.V(x)

        #if(infer):
        mask = inputs.reshape(bs, num_nodes, -1)
        mask = torch.max(mask, 2).values  # get the max along dimension 1 ; dim:0-bs; dim:2-which node; dim:1-time step
        # mask shape is now bs x num_nodes
        mask = torch.where(mask==1.0, -np.inf, 0.0 )
        #V_final = torch.multiply(V, mask)
        V_final = V + mask
        #print('V_final:', V_final)
        #print('mask:', mask)


        #print('V:', V)    
        return V_final if infer else V

class DQN_agent():
    def __init__(self, state_dim, action_dim, **kwargs):
        """ Define all key variables required for DQN agent """

        super().__init__(**kwargs)

        self.num_states = state_dim
        self.num_actions = action_dim
        self.buffer_counter = 0
        self.buffer_capacity = 300000
        self.batch_size = 128
        self.hidden_size = 256#512#256
        
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # run it on GPU when available

        self.state_buffer1 = np.zeros((self.buffer_capacity, self.num_states))
        self.action_buffer1 = np.zeros((self.buffer_capacity, 1))
        self.reward_buffer1 = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer2 = np.zeros((self.buffer_capacity, self.num_states))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))

        # Setup models
        self.critic_model = Critic(self.hidden_size, self.num_states, self.num_actions).to(self.device)
        self.target_critic = Critic(self.hidden_size, self.num_states, self.num_actions).to(self.device)

        # Set target weights to the active model initially
        self.hard_update(self.target_critic, self.critic_model)

        # Used to update target networks
        self.tau = 0.009
        self.gamma = 0.99

        # set epsilon level for performing noisy actions
        self.eps_start = 0.99
        self.eps_end = 0.1
        self.eps_decay = 200
        self.eps = 1.0

        # specify total no. of epsiodes
        self.total_episodes = 500
        self.curr_episode = 0  # keep track of current episode

        # Setup Optimizer
        critic_lr = 5e-4#5e-4
        self.critic_optimizer = Adam(self.critic_model.parameters(), lr=critic_lr)

    def select_action(self, state, episode, infer):
        """
        Input: State of the RL agent
        Output: The appropriate action to take using eps-greedy policy

        Args: 
        * infer - bool variable to indicate inference or training stage
        * epsiode - to indicate the current epsiode

        """

        num_ex = state.shape[0]  # number of states present
        state = torch.Tensor(state).reshape(num_ex, -1).to(self.device)  # reshape input tensor to (num_ex X dim of state space)
        #print(state)
        Q_value = self.critic_model(state, infer=True).detach()

        values = np.array([Q_value[0, i].item() for i in range(self.num_actions)])
        #print('Q_value:', values)
        #legal_actions_set = set(torch.where(Q_value!=-np.inf)[0].tolist())
        legal_actions_set = set(np.where(values!=-np.inf)[0].tolist())
        legal_value_pairs = [(v, i) for (i, v) in enumerate(values) if i  in legal_actions_set]
        #print('LVP:', legal_value_pairs)

        if(not infer):
            roll = np.random.uniform(0, 1)
            if((roll < self.eps) or (episode < 0)):  # random eps_end times or if episode < 2000
                action = np.random.choice([x[1] for x in legal_value_pairs], 1, p=[1/len(legal_value_pairs) for _ in legal_value_pairs])
            else:
                action = max(legal_value_pairs)[1]
            action = torch.tensor([[action]])
        
        else:
            #action = max(legal_value_pairs)[1]
            action = max(legal_value_pairs)[1]
            action = torch.tensor([[action]])

        # slowly anneal the epsilon per episode
        if(self.curr_episode != episode):
            self.eps = max(self.eps_end, self.eps-0.001)
            self.curr_episode = episode

        return action #, Q_value.detach().numpy()

    def update(self, state_batch1, action_batch1, reward_batch1, \
            next_state_batch2, done_batch):

        """
        
        Input: Takes as input state, action, reward, next_state, done batch 
        
        Updates Q-network weights

        """
        next_Q_vals = self.target_critic(next_state_batch2, infer=False).max(1).values.reshape(-1, 1)
        target_vals = reward_batch1 + self.gamma*torch.multiply(done_batch, next_Q_vals)

        action_batch1 = action_batch1.type(torch.int64)
        curr_Q_vals = self.critic_model(state_batch1, infer = False).gather(1, action_batch1.view(-1, 1))

        self.critic_optimizer.zero_grad()
        value_loss = F.mse_loss(curr_Q_vals, target_vals).mean(0).mean()
        value_loss.backward()
        self.critic_optimizer.step()


    def memory(self, obs_tuple):

        """
        Enter (s,a,r,s',done) tuple into the replay buffer

        """
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer1[index] = obs_tuple[0]
        self.action_buffer1[index] = obs_tuple[1]
        self.reward_buffer1[index] = obs_tuple[2]

        self.next_state_buffer2[index] = obs_tuple[3]
        self.done_buffer[index] = obs_tuple[4]

        self.buffer_counter += 1
    
    def train(self):

        """
        * Sample randomly from the replay buffer to get batch variables
        * Update parameters of Q-network using update()
        * Soft update the weights of the target Q-network

        """
        
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch1 = torch.Tensor(self.state_buffer1[batch_indices]).to(self.device)
        action_batch1 = torch.Tensor(self.action_buffer1[batch_indices]).to(self.device)
        reward_batch1 = torch.Tensor(self.reward_buffer1[batch_indices]).to(self.device)

        next_state_batch2 = torch.Tensor(self.next_state_buffer2[batch_indices]).to(self.device)
        done_batch = torch.Tensor(self.done_buffer[batch_indices]).to(self.device)

        self.update(state_batch1, action_batch1, reward_batch1, \
            next_state_batch2, done_batch)
        self.soft_update(self.target_critic, self.critic_model, self.tau)
   
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def load(self, env, agent_id):

        """ 
        Load the ML models
        
        """

        results_dir = "saved_models"
        critic_path = "{}/{}_{}_critic".format(results_dir, env, agent_id)
        self.critic_model.load_state_dict(torch.load(critic_path))      

    def save(self, env):

        """ 
        Save the ML models 

        """
        results_dir = "saved_models"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        critic_path = "{}/{}_DQN_critic".format(results_dir, env)
        torch.save(self.critic_model.state_dict(), critic_path)

        









        
