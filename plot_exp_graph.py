import matplotlib.pyplot as plt
import numpy as np
import statistics

nets = [x for x in range(8,56,4)]

w = 1
convergence_pt_inst = []
convergence_pt_Q = []

convergence_pt_inst_std = []
convergence_pt_Q_std = []


for n in nets:
     con_pt_inst=[]
     con_pt_Q = []

     for tr in range(5):
          arr = np.loadtxt('saved_files/syn_4_' + str(n) + '_alg:inst_' + str(tr) + '.txt')
          conv_arr = np.convolve(arr, np.ones(w), 'valid') / w

          ind = np.where(conv_arr == arr.min())[0][0]
          con_pt_inst.append(ind)

          arr = np.loadtxt('saved_files/syn_4_' + str(n) + '_alg:Q_' + str(tr) + '.txt')
          conv_arr = np.convolve(arr, np.ones(w), 'valid') / w

          ind = np.where(conv_arr == arr.min())[0][0]
          con_pt_Q.append(ind)
     convergence_pt_inst.append(statistics.mean(con_pt_inst))
     convergence_pt_Q.append(statistics.mean(con_pt_Q))
     convergence_pt_inst_std.append(statistics.stdev(con_pt_inst))
     convergence_pt_Q_std.append(statistics.stdev(con_pt_Q))



nets = np.array(nets)

convergence_pt_Q = np.array(convergence_pt_Q)
convergence_pt_Q_std = np.array(convergence_pt_Q_std)
convergence_pt_inst = np.array(convergence_pt_inst)
convergence_pt_inst_std = np.array(convergence_pt_inst_std)
#exp_nets = []
#for i in range(nets.shape[0]):
     #exp_nets.append(convergence_pt[0]) if i==0 else exp_nets.append(exp_nets[i-1]*2 )

MC_nets = np.loadtxt('saved_files/MC_all_nets_mean.txt')
MC_nets_std = np.loadtxt('saved_files/MC_all_nets_std.txt')
#print(MC_nets)

plt.figure()
#plt.plot(nets//2, np.log(np.array(convergence_pt)), label='RL-Ours')
#plt.plot(nets//2, np.log(np.array(exp_nets)), label='Random')
plt.plot(nets//2, convergence_pt_inst,'.-', label='RL')
plt.fill_between(nets//2, convergence_pt_inst - convergence_pt_inst_std, convergence_pt_inst + convergence_pt_inst_std, alpha=0.2)

#plt.plot(nets//2, (np.array(exp_nets)), label='Random')
plt.plot(nets//2, convergence_pt_Q,'.-', label='RL-Q-like')
plt.fill_between(nets//2, convergence_pt_Q - convergence_pt_Q_std, convergence_pt_Q + convergence_pt_Q_std, alpha=0.2)

plt.plot(nets//2, (MC_nets),'.-', label='Random')
plt.fill_between(nets//2, MC_nets - MC_nets_std, MC_nets + MC_nets_std, alpha=0.2)

plt.xlabel('number of nets')
plt.ylabel('Data required to find first occurence')
plt.legend()
plt.savefig('Figures/convergence_trend.png')

