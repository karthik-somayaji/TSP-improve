from turtle import st
import torch
#from torch_geometric.data import Data
#import torch_geometric
from scipy.spatial import distance
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

#start_positions = [[5.5, 0.5], [4.5, 0.5], [2.5, 0.5], [1.5, 1.5], [2.5, 4.5], [5.5, 3.5], [4.5, 4.5], [7.5, 5.5], [4.5, 5.5], [0.5, 5.5]]
#end_positions =   [[7.5, 2.5], [5.5, 1.5], [0.5, 2.5], [2.5, 5.5], [6.5, 5.5], [3.5, 6.5], [5.5, 6.5], [5.5, 7.5], [1.5, 6.5], [1.5, 7.5]]

class graph_structure():
    def __init__(self):
        pass
    
    def _getgraph(self, type_fn):
        start_positions_lee = [[2, 0], [3, 0], [5, 0], [6, 1], [5, 4], [2, 3], [3, 4], [0, 5], [3, 5], [7, 5]]
        end_positions_lee =   [[0, 2], [2, 1], [7, 2], [5, 5], [1, 5], [4, 6], [2, 6], [2, 7], [6, 6], [6, 7]]

        start_positions_syn_4_20 = [[1, 0], [2, 1], [1, 4], [2, 5], [1, 8], [2, 9], [1, 12], [2, 13], [1, 16], [2, 17]]
        end_positions_syn_4_20 =  [[3, 2], [0, 3], [3, 6], [0, 7], [3, 10], [0, 11], [3, 14], [0, 15], [3, 18], [0, 19]]

        start_positions_syn_4_16 = [[1, 0], [2, 1], [1, 4], [2, 5], [1, 8], [2, 9], [1, 12], [2, 13]]
        end_positions_syn_4_16 =  [[3, 2], [0, 3], [3, 6], [0, 7], [3, 10], [0, 11], [3, 14], [0, 15]]

        start_positions_syn_4_12 = [[1, 0], [2, 1], [1, 4], [2, 5], [1, 8], [2, 9] ]
        end_positions_syn_4_12 =  [[3, 2], [0, 3], [3, 6], [0, 7], [3, 10], [0, 11]]

        start_positions_syn_4_8 = [[1, 0], [2, 1], [1, 4], [2, 5]]
        end_positions_syn_4_8 =  [[3, 2], [0, 3], [3, 6], [0, 7]]

        if('Lee' in type_fn):
            start_positions = start_positions_lee
            end_positions = end_positions_lee
        elif('syn_4_12' in type_fn):
            start_positions = start_positions_syn_4_12
            end_positions = end_positions_syn_4_12
        elif('syn_4_16' in type_fn):
            start_positions = start_positions_syn_4_16
            end_positions = end_positions_syn_4_16
        elif('syn_4_20' in type_fn):
            start_positions = start_positions_syn_4_20
            end_positions = end_positions_syn_4_20
        elif('syn_4_8' in type_fn):
            start_positions = start_positions_syn_4_8
            end_positions = end_positions_syn_4_8

        concat_list = start_positions + end_positions
        n = len(start_positions)

        labels = [str(i) for i in range(n)] + [str(i)+'*' for i in range(n)]
        pos = {labels[x]:concat_list[x] for x in range(2*n)}

        pos = {x:concat_list[x] for x in range(2*n)}

        threshold_distance = 1.0
        adj_list = []
        feature_list = []
        adj_list_matrix = torch.zeros(n, n)

        # connected nodes to the node
        for i in range(len(start_positions)):
            for j in range(len(start_positions)):
                is_neighbor_1 = distance.euclidean([start_positions[i][0], start_positions[i][1]], [start_positions[j][0], start_positions[j][1]] ) <= threshold_distance
                is_neighbor_2 = distance.euclidean([end_positions[i][0], end_positions[i][1]], [end_positions[j][0], end_positions[j][1]] ) <= threshold_distance
                if(is_neighbor_1 or is_neighbor_2):
                    adj_list.append([i,j])
                    adj_list_matrix[i,j] = 1.0
                #adj_list_matrix[i,j] = 1.0

        # define feature vector for each node
        for i in range(len(start_positions)):
            
            if(i < len(start_positions)):   
                MD_y, MD_x = np.abs(start_positions[i][0] - end_positions[i][0]), np.abs(start_positions[i][1] - end_positions[i][1])
                net_num = np.zeros(len(start_positions))
                net_num[i] = 1.0
                net_num = net_num.tolist()  
                #feature = [start_positions[i][0], start_positions[i][1], MD_y, MD_x ] + net_num
                feature = [start_positions[i][0], start_positions[i][1], end_positions[i][0], end_positions[i][1] ] + net_num
            feature_list.append(feature)


        x = torch.tensor(feature_list, dtype = torch.float)#, dtype = torch.long)
        mask_adj = torch.tensor(adj_list, dtype = torch.long)

        return x, adj_list_matrix







