'''
call this script with python train [task] [model_id] [spec], where task, model_od and spec need to be a valid string
so e.g. python train "IMDB" "bert-base-uncased" "N_pro"
If you want to train all specs for that task and base model call with spec = "all" (e.g. python train "IMDB" "bert-base-uncased" "all").
if you'd like to process multiple specs but not all of them, you can also provide a list of specs (e.g. python train "IMDB" "bert-base-uncased" ["N_pro", "mix_weat"])
'''

#import sys, os, random, time
#import torch
#import pandas as pd
#from os import walk
#import train_util as tu
#from sklearn.model_selection import train_test_split 

# I think I do not need any more Imports. All are included in train_util

from train_script import *
from rtpt import RTPT

vars = sys.argv[1:]
print(vars)


assert(len(vars) == 3), "something's wrong with the parameters here. Check that, please. \n call this script with python train [task] [model_id] [spec], where task, model_od and spec need to be a valid string. So e.g. python train 'IMDB' 'bert-base-uncased' all"

# now we know that we have the right amount of vars 
task_in = vars[0]
assert(task_in in ['IMDB', 'Twitter']), 'task name is not valid'
model_id_in = vars[1]
assert(model_id_in in ["bertbase", 'bertlarge', "distbase", "distlarge", "robertabase", "robertalarge", "albertbase", "albertlarge"]), model_id_in + ' is not a valid model_id'
spec_in = vars[2].split()
print(spec_in)
print('called train.py {} {} {}'.format(task_in, model_id_in, spec_in))


log_path = './res_models/logs/train_{}_{}_{}.txt'.format(task_in, model_id_in, timestamp()) # tu.
sys.stdout = open(log_path, 'a')

###

    
def acc_df(task_, model_id_, specs_):
    '''
    todo 
    '''
    df_acc = pd.DataFrame()
    for spec in specs_: 
        rtpt_train.step('evaluate ' + spec)
        
        foo = calc_acc(spec, tokenizer, model_id_, task_, True) # tu.
        foo['data set'] = 'spec'    
        bar = calc_acc(spec, tokenizer, model_id_, task_, False) # tu.
        bar['data set'] = 'all'
        foo['spec'] = spec
        bar['spec'] = spec
        df_acc = df_acc.append(foo, ignore_index = True)
        df_acc = df_acc.append(bar, ignore_index = True)
        print(df_acc) # delete when everything works fine 
    print(df_acc)
    df_acc.to_pickle(check_path('res_models/accuracies/') + 'acc_{}_{}'.format(task_, model_id_)) # tu.
    print('\n' + __name__+ ':')
    return df_acc    

# model_id can be "bertbase", 'bertlarge', "distbase", "distlarge", "robertabase", "robertalarge", "albertbase", "albertlarge"
# tokenizer, model = load_hf(model_id_in) # tu.

specs_all = ["N_pro", "N_weat", "N_all", "mix_pro", "mix_weat", "mix_all", "original"]
if spec_in == ["all"]:
    specs = specs_all
    print('1#')
elif type(spec_in)== list:
    specs = spec_in
    print('12')
elif type(spec_in)==str:
    assert(type(spec_in)==list), "spec is not a list here. This will cause issues later." 
    specs = spec_in
    print('13')
for spec in specs: 
    assert(spec in specs_all), '{} is no legit specification (spec)'.format(spec)

    
rtpt_train = RTPT(name_initials='SJ', experiment_name=task_in, max_iterations=len(specs)*2)
rtpt_train .start()

for spec in specs:
    train(task_in, model_id_in, spec)
    rtpt_train.step(subtitle=f"train")


df_acc_ = acc_df(task_in, model_id_in, specs)
# print(df_acc_) 
