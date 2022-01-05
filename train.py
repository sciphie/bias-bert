'''
call this script with python train [task] [model_id] [spec], where task, model_od and spec need to be a valid string
so e.g. python train "IMDB" "bert-base-uncased" "N_pro"
If you want to train all specs for that task and base model call with spec = "all" (e.g. python train "IMDB" "bert-base-uncased" "all").
if you'd like to process multiple specs but not all of them, you can also provide a list of specs (e.g. python train "IMDB" "bert-base-uncased" ["N_pro", "mix_weat"])
'''

import sys, os, random, time
import torch
import pandas as pd
from os import walk
#import train_util as tu
from train_util import *
from rtpt import RTPT
#from sklearn.model_selection import train_test_split # this is in train_util

vars = sys.argv[1:]
print(vars)

assert(len(vars) == 3), "something's wrong with the parameters here. Check that, please. \n call this script with python train [task] [model_id] [spec], where task, model_od and spec need to be a valid string. So e.g. python train 'IMDB' 'bert-base-uncased' all"

# now we know that we have the right amount of vars 
task_in = vars[0]
assert(task_in in ['IMDB', 'Twitter']), 'task name is not valid'
model_id_in = vars[1]
assert(model_id_in in ["bertbase", 'bertlarge', "distbase", "distlarge", "robertabase", "robertalarge", "albertbase", "albertlarge"]), model_id_in + ' is not a valid model_id'
spec_in = vars[2]
print('called train.py {} {} {}'.format(task_in, model_id_in, spec_in))


log_path = './res_models/logs/train_{}_{}_{}.txt'.format(task_in, model_id_in, timestamp(True)) # tu.
sys.stdout = open(log_path, 'w')

###

def train(task, model_id, spec, eval_steps_=500, per_device_train_batch_size_=8, per_device_eval_batch_size_=8, num_train_epochs_=3):  
    '''
    todo
    '''
    # I removed the logging block for the moment as I am not using it right now anyway 
    
    print('{}: params: spec= {}; eval_steps_={}; per_device_train_batch_size_={}; per_device_eval_batch_size_={}; num_train_epochs_={}'.format(__name__, spec, eval_steps_, per_device_train_batch_size_, per_device_eval_batch_size_, num_train_epochs_))
        
    ### ### ### ### ### 
    # load data set
    data_set_path = 'res_data/{}_training/{}_{}_'.format(task, task, spec)
    df_train = pd.read_pickle(data_set_path+'train')
    df_test = pd.read_pickle(data_set_path+'test')
    
    print(__name__ +': successfully loaded --- ' + data_set_path)
    
    # modify data sets 
    for df in [df_train, df_test]:
        df.label = pd.factorize(df.label)[0]
        df.rename(columns={df.columns[1]:'text'}, inplace=True)
    
    X = list(df_train["text"])
    y = list(df_train["label"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=11)
    
    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
    X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)
    
    train_dataset = Dataset(X_train_tokenized, y_train) # tu.
    val_dataset = Dataset(X_val_tokenized, y_val) # tu.
    
    output_path = check_path('res_models/{}/{}/output_{}'.format(task, model_id, spec)) # tu.

    # Define Trainer
    args = TrainingArguments(
        output_dir=output_path,
        evaluation_strategy="steps",
        eval_steps=eval_steps_, #500,
        per_device_train_batch_size=per_device_train_batch_size_, #8,
        per_device_eval_batch_size=per_device_eval_batch_size_, # 8,
        num_train_epochs=num_train_epochs_ ,#3,
        seed=0,
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics= compute_metrics, # tu.
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train pre-trained model
    trainer.train()

    
def acc_df(task_, specs_):
    '''
    todo 
    '''
    df_acc = pd.DataFrame()
    for spec in specs_: 
        rtpt.step('evaluate ' + spec)
        
        foo = calc_acc(spec, tokenizer, model_id, task_, True) # tu.
        foo['data set'] = 'spec'    
        bar = calc_acc(spec, tokenizer, model_id, task_, False) # tu.
        bar['data set'] = 'all'
        foo['spec'] = spec
        bar['spec'] = spec
        df_acc = df_acc.append(foo, ignore_index = True)
        df_acc = df_acc.append(bar, ignore_index = True)
    df_acc.to_pickle(check_path('res_models/accuracies/acc_' + task + '_' + model_id)) # tu.
    print('\n' + __name__+ ':')
    print(df_acc)
    return df_acc    

# model_id can be "bertbase", 'bertlarge', "distbase", "distlarge", "robertabase", "robertalarge", "albertbase", "albertlarge"
tokenizer, model = load_hf(model_id_in) # tu.

specs_all = ["N_pro", "N_weat", "N_all", "mix_pro", "mix_weat", "mix_all", "original"]
if spec_in == "all":
    specs = specs_all
elif type(spec_in)== list:
    specs = spec_in
elif type(spec_in)==str:
    specs = [spec_in]
for spec in specs: 
    assert(spec in specs_all), '{} is no legit specification (spec)'.format(spec) 


rtpt_train = RTPT(name_initials='SJ', experiment_name='train {} {}'.format(task_in, model_id_in), max_iterations=len(specs)*2)
for spec in specs:
    train(task_in, model_id_in, spec)
    rtpt.step('train ' + spec)

df_acc_ = acc_df(task_in, specs)
print(df_acc_) 
