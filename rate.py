'''
Which vars do I need here? 
task - model_id - spec
'''

import os,sys
#os.environ["CUDA_VISIBLE_DEVICES"] = "13" # str(i)

import datetime, random, time, torch
import pandas as pd
from os import walk
import torch.nn.functional as F
from glob import glob
from transformers import ( 
    TrainingArguments, 
    Trainer,
    BertTokenizer, 
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    EarlyStoppingCallback,
)

from util import load_hf, Dataset
from rtpt import RTPT
#from sklearn.model_selection import train_test_split # this is in train_util

task_ = "IMDB" #  "Twitter" 
#model_id_ = "bertbase"
#model_ids = ["bertlarge", "albertbase", "albertlarge"]
#model_ids = ["distbase", "robertabase", "robertalarge"]

model_ids = ["albertbase", "albertlarge"]
# 3 ["bertbase", "robertabase"] - 1 
# 5 , ["distbase"] - 2 
# 2 ["bertlarge", "robertalarge"] - 4
# 1 ,

specs_all = ["N_pro"] #, "N_weat"] #, "N_all", "mix_pro", "mix_weat"] #, "mix_all", "original"]
specs = specs_all 

########## F U N C T I O N S ##########
### from rate.ipynb
def rate(task, model_id, spec,rate_data= None, addition = None):
    print(task, model_id, spec, addition)
    
    ########## ########## ########## ########## ########## 
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    ########## ########## ########## ########## ########## 
    
    model_name = "{}_{}_{}".format(task, model_id, spec)
    if addition: 
        model_name = "{}_{}_{}_{}".format(task, model_id, spec, addition)
    print('called rate.py with {}'.format(model_name))
    t = datetime.datetime.now().strftime('%m_%d_%H%M')
    
    path = "./res_models/models/" + model_name ### Change here! ### 
    log_path = path + '/log_rate_{}.txt'.format(model_name) #, t) 
    #sys.stdout = open(log_path, 'a')
    
    '''
    '''
    df_l = pd.read_pickle('res_data/{}_l_test'.format(task))
    
    df_exp = []
    
    if rate_data:
        elem = rate_data
        df_exp = df_l[['ID', 'text_{}_M'.format(elem), 'text_{}_F'.format(elem)]]
        print('rate_data = {}'.format(elem))
 #### wieder einkommentieren bitte #### 
 #   elif spec=="original":
 #       elem = 'all'
 #       df_exp = df_l[['ID', 'text_{}_M'.format(elem), 'text_{}_F'.format(elem)]]
 #   else:
 #       for elem in ['weat', 'pro', 'all']: 
 #           if elem in spec:
 #               df_exp = df_l[['ID', 'text_{}_M'.format(elem), 'text_{}_F'.format(elem)]]
 #               print('rate experimental data type {}: {} and {}'.format(elem, df_exp.columns[1], df_exp.columns[2]))
 #   print(df_exp)
    
    assert(df_exp.shape[0]> 0), 'rate(): What type of data should be uses? spec does not fit in any categorie'
    
    model_path = path
    filenames = glob(path + "/*")
    print(filenames)
    for i in range(len(filenames)):
        if 'epoch' in filenames[i]: 
            # Load trained model
            model_path = filenames[i]
        
    tokenizer, model = load_hf(model_id, path_to_model=model_path)
    # model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    #if not tokenizer:
    #
    # Define test trainer
    model.to(device)
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
        
    df_exp.to_pickle('res_results/rating_{}_R{}'.format(model_name, rate_data))#{}_{}_{}'.format(task, model_id, spec))
    print("#### ")
    print(df_exp)
    return df_exp


############################
'''
rtpt_train = RTPT(name_initials='SJ', experiment_name=task_, max_iterations=len(specs)*len(model_ids))

rtpt_train.start()

#for m in model_ids:
#    output = []
#    for spec in specs:
#        tt = rate(task_ ,m ,spec)
#        output.append(tt)
#        rtpt_train.step()
#    print(output)


for m in model_ids:
    output = []
    for spec in specs:
        tt = rate(task_ ,m ,spec)
        output.append(tt)
        rtpt_train.step()
    print(output)
    
'''