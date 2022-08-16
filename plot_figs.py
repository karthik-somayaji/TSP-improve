import numpy as np
import matplotlib.pyplot as plt

inp = np.loadtxt('Lee_29.txt')

inp[inp==1000] = 500

plt.figure()
plt.plot(inp, label='Lee_29')
plt.xlabel('Epochs')
plt.ylabel('Final Cost')
plt.legend()
plt.savefig('Figures/Lee_29.png')
