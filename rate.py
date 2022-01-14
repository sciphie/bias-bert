'''
Which vars do I need here? 
task - model_id - spec
'''

import sys, os, random, time, torch
import pandas as pd
from os import walk
import torch.nn.functional as F
from glob import glob

from train_functions import load_hf
from rtpt import RTPT
#from sklearn.model_selection import train_test_split # this is in train_util

vars = sys.argv[1:]
print(vars)
print(type(vars[2]))
print(len(vars))

assert(len(vars) == 3), "something's wrong with the parameters here. Check that, please. \n call this script with python train [task] [model_id] [spec], where task, model_od and spec need to be a valid string. So e.g. python train 'IMDB' 'bert-base-uncased' all"

# now we know that we have the right amount of vars 
task_in = vars[0]
assert(task_in in ['IMDB', 'Twitter']), 'task name is not valid'
model_id_in = vars[1]
assert(model_id_in in ["bertbase", 'bertlarge', "distbase", "distlarge", "robertabase", "robertalarge", "albertbase", "albertlarge"]), model_id_in + ' is not a valid model_id'
spec_in = vars[2].split()
print(spec_in)
print('called rate.py {} {} {}'.format(task_in, model_id_in, spec_in))

log_path = './res_models/logs/rate_{}_{}_{}.txt'.format(task_in, model_id_in, timestamp()) # tu.
sys.stdout = open(log_path, 'a')


########## F U N C T I O N S ##########
### from rate.ipynb
def rate(task, model_id, spec):
    '''
    '''
    df_l = pd.read_pickle('res_data/{}_l_test'.format(task))
    
    df_exp = []
    for elem in ['weat'. 'pro', 'all']: 
        if elem in spec:
            df_exp = df_l[['ID', 'text_{}_M'.format(elem), 'text_{}_F'.format(elem)]]
            print('rate experimental data type {}: {} and {}'.format(elem, df_exp.columns[1], df_exp.columns[2]))
    assert(df_exp), 'rate(): What type of data should be uses? spec does not fit in any categorie'
    
    
    filenames = glob("res_models/{}/{}/output_{}/*".format(task, model_id, spec))
    #filenames = next(walk(), (None, [], None))[1]  # [] if no file
    filenames.reverse()
    print(filenames)
    
    # Load trained model
    model_path = filenames[0]
    tokenizer, model = load_hf(model_id, load_model=True, path_to_model=model_path)
    # model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    #if not tokenizer:
    #
    # Define test trainer
    test_trainer = Trainer(model)
    
    male_texts = df_exp[df_exp.columns[1]].tolist()
    female_texts = df_exp[df_exp.columns[2]].tolist()
    
    male_texts_tokenized = tokenizer(male_texts, padding=True, truncation=True, max_length=512)
    female_texts_tokenized = tokenizer(female_texts, padding=True, truncation=True, max_length=512)
    
    # Create torch dataset
    male_dataset = Dataset(male_texts_tokenized)
    female_dataset = Dataset(female_texts_tokenized)
    
    # Make prediction
    raw_pred_m, _, _ = test_trainer.predict(male_dataset)
    raw_pred_f, _, _ = test_trainer.predict(female_dataset)
    
    # Preprocess raw predictions
    #y _pred_m = np.argmax(raw_pred_m, axis=1)
    # y_pred_f = np.argmax(raw_pred_f, axis=1)
    
    y_soft_m = F.softmax(torch.from_numpy(raw_pred_m), dim=1).tolist()
    y_soft_f = F.softmax(torch.from_numpy(raw_pred_f), dim=1).tolist()
        
    #if not (all([abs(x[0]+x[1]-1) < 0.00001 for x in y_soft_m]) or all([abs(x[0]+x[1]-1) < 0.000001 for x in y_soft_f])):
    #    logging.error("Softmax values do not sum to 1")    
    assert(all([abs(x[0]+x[1]-1) < 0.00001 for x in y_soft_m]) or all([abs(x[0]+x[1]-1) < 0.000001 for x in y_soft_f])), "Softmax values do not sum to 1"
    df_exp['pos_prob_m'] = [e[0] for e in y_soft_m]
    df_exp['pos_prob_f'] = [e[0] for e in y_soft_f]
    df_exp['bias'] = df_exp['pos_prob_m']-df_exp['pos_prob_f']
        
    df_exp.to_pickle('res_results/rating_{}_{}_{}'.format(task, model_id, spec))
    return df_exp


############################
#tokenizer, model_foo = load_hf(model_id_in, False) # tu.
# assert(model_foo == None)

specs_all = ["N_pro", "N_weat", "N_all", "mix_pro", "mix_weat", "mix_all", "original"]
if spec_in == ["all"]:
    specs = specs_all
    print('SPECs : rate for all specs')
elif type(spec_in)== list:
    specs = spec_in
    print('SPECs : rate for a subset: ', specs)
elif type(spec_in)==str:
    assert(type(spec_in)==list), "spec is not a list here. This will cause issues later." 
    specs = spec_in
    print('SPECs : rate for only one spec: ' + spec_in)
for spec in specs: 
    assert(spec in specs_all), '{} is no legit specification (spec)'.format(spec)
    
rtpt_train = RTPT(name_initials='SJ', experiment_name=task_in, max_iterations=len(specs)*2)
rtpt_train .start()

output = []
for spec in specs:
    tt = rate(spec)
    output.append(tt)
    



