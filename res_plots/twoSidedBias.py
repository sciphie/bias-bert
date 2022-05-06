#import analysis_util as au
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys, os
from glob import glob
import pickle5 as pickle
from util import time

sys.path.append("/localdata2/jent_so/LM_GenderBias")


def posneg(list):
    pos = [x for x in list if x>0]
    neg = [x for x in list if x<0]
    print(len(pos), len(neg)) 
    return np.mean(pos), np.mean(neg), len(pos), len(neg)


def rename(name):
    specs = ['N_pro', 'N_weat', 'N_all', 'mix_pro', 'mix_weat', 'mix_all', 'original']
    for spec in specs:
        if spec in name:
            return spec
    print("error")


def calc_bias_dict(df_dict):
    bias_dict = {}

    for spec in df_dict.keys():
        bias_l = df_dict[spec].bias.tolist()
        # total bias
        overall_bias_total = np.mean(bias_l)
        overall_bias_total_noZero = np.mean([i for i in bias_l if i != 0])
        # absolute bias
        overall_bias_abs = np.mean([abs(x) for x in bias_l])
        overall_bias_abs_noZero = np.mean([abs(x) for x in bias_l if x != 0])
        # pos neg bias 
        pos, neg, pos_n, neg_n = posneg(bias_l) 

        bias_dict[spec] = [overall_bias_total, overall_bias_abs, pos, neg, pos_n, neg_n, overall_bias_total_noZero, overall_bias_abs_noZero  ]
    return bias_dict
        
        
def twoSidedBias(task, model_id, 
                 # specs= ['N_pro', 'N_weat', 'N_all', 'mix_pro', 'mix_weat', 'mix_all', 'original'], 
                 safe_name=None, y_lim= None):
    
    files = glob("res_results/ratings/*")
    
    df_dict = {}
    for file in files: 
        if '_{}_'.format(model_id) in file and task in file:
            print(file)
            with open (file, "rb") as fh:
                data = pickle.load(fh)
            df_dict[rename(file)] = data
    
    plt.rcParams["figure.figsize"] = (5,5)
    
    bias_dict = calc_bias_dict(df_dict) 
    specs = list(df_dict.keys())
    specs.sort()
    
    myorder = [6, 0,1,2, 3,4,5]

    poss = [bias_dict[spec][2] for spec in specs] 
    negs = [bias_dict[spec][3] for spec in specs] 
    
    poss = [poss[i] for i in myorder]
    negs = [negs[i] for i in myorder]
    specs = [specs[i] for i in myorder]
    
    print(poss)
    c0 = 'tab:blue'
    c1 = 'tab:orange'

    x_pos = np.arange(len(poss))
    
    if y_lim:
        plt.ylim(y_lim)
    
    # Create bars
    #plt.bar(x_pos, biases_abs)
    plt.bar(x_pos, poss, color=c0)
    plt.bar(x_pos, negs, color=c1)
    plt.title('{} {}'.format(task, model_id))
    
    # Create names on the x-axis
    plt.xticks(x_pos, specs)
    
    # Show graphic
    plt.grid()
    if safe_name:
        #plt.savefig("res_plots/{}_{}".format(safe_name, time() ) )
        plt.savefig("res_plots/{}".format(safe_name))
    plt.show()

