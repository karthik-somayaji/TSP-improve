import system
import argparse

parser = argparse.ArgumentParser(description="RL Comb-Opt")
parser.add_argument("-p", "--prob", type=str, default="syn_4_36")
args = parser.parse_args()

no_of_nets = int(args.prob[6:])//2

if(0<=no_of_nets<=8):
    num_layer = 4
elif(8<no_of_nets<=16):
    num_layer = 6
elif(16<no_of_nets<=24):
    num_layer = 7 
elif(24<no_of_nets<=32):
    num_layer = 9

for trial in range(5):
    exec = "python train_ours.py --num_layers {} --prob {} -b 1 -t 1000 -a Q -tr {} ".format(num_layer, args.prob, trial )
    system(exec)