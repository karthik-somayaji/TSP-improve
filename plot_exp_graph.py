import matplotlib.pyplot as plt
import numpy as np

nets = [x for x in range(8,56,4)]

w = 1
convergence_pt = []

for n in nets:
     arr = np.loadtxt('saved_files/syn_4_' + str(n) + '.txt')
     conv_arr = np.convolve(arr, np.ones(w), 'valid') / w

     ind = np.where(conv_arr == arr.min())[0][0]
     #print(ind)
     convergence_pt.append(ind)

nets = np.array(nets)
exp_nets = []
for i in range(nets.shape[0]):
     exp_nets.append(convergence_pt[0]) if i==0 else exp_nets.append(exp_nets[i-1]*2 )

MC_nets = np.loadtxt('saved_files/MC_all_nets.txt')

plt.figure()
#plt.plot(nets//2, np.log(np.array(convergence_pt)), label='RL-Ours')
#plt.plot(nets//2, np.log(np.array(exp_nets)), label='Random')
plt.plot(nets//2, (np.array(convergence_pt)), label='RL-Ours')
#plt.plot(nets//2, (np.array(exp_nets)), label='Random')
plt.plot(nets//2, (MC_nets), label='Random')
plt.xlabel('number of nets')
plt.ylabel('Data required to find first occurence')
plt.legend()
plt.savefig('Figures/convergence_trend.png')

