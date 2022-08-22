import matplotlib.pyplot as plt
import numpy as np

nets = [x for x in range(8,56,4)]

w = 50
convergence_pt = []

for n in nets:
     arr = np.loadtxt('syn_4_' + str(n) + '.txt')
     conv_arr = np.convolve(arr, np.ones(w), 'valid') / w

     ind = np.where(conv_arr == arr.min())[0][0]
     #print(ind)
     convergence_pt.append(ind)

print(convergence_pt)

    

