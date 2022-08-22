import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="RL Comb-Opt")
parser.add_argument("-p", "--prob", type=str, default="syn_4_36")
args = parser.parse_args()

prob_type = args.prob
print(prob_type)

txt_path = prob_type + '.txt'
inp = np.loadtxt(txt_path)

#inp[inp>=1000] = 0

plt.figure()
plt.plot(inp, label=prob_type)
plt.xlabel('Epochs')
plt.ylabel('Final Cost')
plt.legend()

path = 'Figures/'+ prob_type+'.png'
plt.savefig(path)
