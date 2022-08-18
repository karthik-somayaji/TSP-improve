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

        start_positions_custom_8_4_20 = [[1, 0], [2, 1], [1, 4], [2, 5], [1, 8], [2, 9], [1, 3], [2, 4], [1, 7], [2, 8], [1, 11], [2, 12], [1, 15], [2, 16], [1, 19], [2, 20], [2, 20], [3, 20], [5, 20], [6, 21], [5, 24], [2, 23], [3, 24], [0, 25], [3, 25], [7, 25]]
        end_positions_custom_8_4_20 =  [[3, 2], [0, 3], [3, 6], [0, 7], [3, 10], [0, 11], [3, 5], [0, 6], [3, 9], [0, 10], [3, 13], [0, 14], [3, 17], [0, 18], [3, 21], [0, 22],  [0, 22], [2, 21], [7, 22], [5, 25], [1, 25], [4, 26], [2, 26], [2, 27], [6, 26], [6, 27]]

        start_positions_custom_8_4_20 = [[1, 0], [2, 1], [1, 4], [2, 5], [1, 8], [2, 9], [4, 0], [5, 1], [4, 4], [5, 5], [4, 8], [5, 9], [4, 12], [5, 13], [4, 16], [5, 17], [2, 20], [3, 20], [5, 20], [6, 21], [5, 24], [2, 23], [3, 24], [0, 25], [3, 25], [7, 25]]
        end_positions_custom_8_4_20 = [[3, 2], [0, 3], [3, 6], [0, 7], [3, 10], [0, 11], [6, 2], [3, 3], [6, 6], [3, 7], [6, 10], [3, 11], [6, 14], [3, 15], [6, 18], [3, 19], [0, 22], [2, 21], [7, 22], [5, 25], [1, 25], [4, 26], [2, 26], [2, 27], [6, 26], [6, 27]]
        
        if('Lee' in type_fn):

            m = int(type_fn[4:])
            start_positions = start_positions_lee
            end_positions = end_positions_lee
            mult = (m+7)//8

            s = [(x*mult,y*mult) for (x,y) in start_positions]
            e = [(x*mult,y*mult) for (x,y) in end_positions]
            start_positions = s.copy()
            end_positions = e.copy()

        elif('syn' in type_fn):

            n = int(type_fn[6:])
            st1 = [[1,x] for x in range(0,n,4)]
            st2 = [[2,x+1] for x in range(0,n,4)]

            st = [j for i in zip(st1,st2) for j in i]

            en1 = [[3,x] for x in range(2,n,4)]
            en2 = [[0,x+1] for x in range(2,n,4)]

            en = [j for i in zip(en1,en2) for j in i]

            start_positions = st.copy()
            end_positions = en.copy()
        
        elif('custom_8_4_20' in type_fn):

            start_positions = start_positions_custom_8_4_20
            end_positions = end_positions_custom_8_4_20


        concat_list = start_positions + end_positions
        n = len(start_positions)

        labels = [str(i) for i in range(n)] + [str(i)+'*' for i in range(n)]
        pos = {labels[x]:concat_list[x] for x in range(2*n)}

        pos = {x:concat_list[x] for x in range(2*n)}

        threshold_distance = 5.0 #5.0
        adj_list = []
        feature_list = []
        adj_list_matrix = torch.zeros(n, n)

        # connected nodes to the node
        for i in range(len(start_positions)):
            for j in range(len(start_positions)):
                is_neighbor_1 = distance.euclidean([start_positions[i][0], start_positions[i][1]], [start_positions[j][0], start_positions[j][1]] ) <= threshold_distance
                is_neighbor_2 = distance.euclidean([end_positions[i][0], end_positions[i][1]], [end_positions[j][0], end_positions[j][1]] ) <= threshold_distance
                is_neighbor_3 = distance.euclidean([start_positions[i][0], start_positions[i][1]], [end_positions[j][0], end_positions[j][1]] ) <= threshold_distance
                is_neighbor_4 = distance.euclidean([end_positions[i][0], end_positions[i][1]], [start_positions[j][0], start_positions[j][1]] ) <= threshold_distance

                if(is_neighbor_1 or is_neighbor_2):# or is_neighbor_3 or is_neighbor_4):
                    adj_list.append([i,j])
                    adj_list_matrix[i,j] = 1.0
                adj_list_matrix[i,j] = 1.0


        # define feature vector for each node
        for i in range(len(start_positions)):
            
            if(i < len(start_positions)):   
                MD_y, MD_x = np.abs(start_positions[i][0] - end_positions[i][0]), np.abs(start_positions[i][1] - end_positions[i][1])
                net_num = np.zeros(len(start_positions))
                net_num[i] = 1.0
                net_num = net_num.tolist()  
                #feature = [start_positions[i][0], start_positions[i][1], MD_y, MD_x ] + net_num
                feature = [start_positions[i][0], start_positions[i][1], end_positions[i][0], end_positions[i][1] ] + net_num
                #feature = [start_positions[i][0], start_positions[i][1]] + net_num
            feature_list.append(feature)


        x = torch.tensor(feature_list, dtype = torch.float)#, dtype = torch.long)
        mask_adj = torch.tensor(adj_list, dtype = torch.long)

        return x, adj_list_matrix







