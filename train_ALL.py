from os import system
import argparse

parser = argparse.ArgumentParser(description="RL Comb-Opt")
parser.add_argument("-p", "--prob", type=str, default="syn_4_36")
args = parser.parse_args()

#prob_arr = ['syn_4_12', 'syn_4_16', 'syn_4_20', 'syn_4_24', 'syn_4_28', 'syn_4_32', 'syn_4_36', 'syn_4_40', 'syn_4_44', 'syn_4_48']
prob_arr = ['Lee_22', 'Lee_29']

for prob in prob_arr:
    if('syn' in prob_arr[0]):
        no_of_nets = int(prob[6:])//2
    elif('Lee' in prob_arr[0]):
        no_of_nets = 10

    if(0<=no_of_nets<=8):
        num_layer = 4
    elif(8<no_of_nets<=16):
        num_layer = 4
    elif(16<no_of_nets<=24):
        num_layer = 9
    elif(24<no_of_nets<=32):
        num_layer = 11

    for trial in range(0,5):
        exec_Q = "python train_ours.py --num_layers {} --prob {} -b 1 -t 1000 -a Q -tr {} ".format(num_layer, prob, trial )
        system(exec_Q)

        exec_inst = "python train_ours.py --num_layers {} --prob {} -b 1 -t 1000 -a inst -tr {} ".format(num_layer, prob, trial )
        system(exec_inst)