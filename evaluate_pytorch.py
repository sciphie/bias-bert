import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # str(i)
from rtpt import RTPT
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler

import util as u 
from glob import glob


#####
task_in = 'IMDB' 
#task_in = 'Twitter' 

# "bertbase", 'bertlarge', "distbase", "distlarge", "robertabase", "robertalarge", "albertbase", "albertlarge"
#model_id_in = "bertbase"
model_id_in = "bertbase"
specs_all = ["original", "N_pro", "N_weat", "N_all", "mix_pro", "mix_weat", "mix_all"]

specs_in = specs_all


def test(model_path, 
         spec = None, # if not specified, the test data will correspond to the training data
         batch_size_=32): #task=task_in, model_id=model_id_in, spec=specs_in[0]): 
    # df = pd.read_csv("./cola_public/raw/out_of_domain_dev.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    
    #df = pd.read_pickle(test_data_path)
    
    # Report the number of sentences.
    # print('Number of test sentences: {:,}\n'.format(df.shape[0]))
    
    
    ########## ########## ########## ########## ########## 
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    ########## ########## ########## ########## ########## 
    
    task = u.identify_task(model_path)
    model_id = u.identify_model_id(model_path)
    if not spec: 
        spec = u.identify_spec(model_path)
                
    sys.stdout = open(model_path + '/evaluation_{}_log.txt'.format(spec), 'a')
    
    # Create sentence and label lists
    sentences, labels = u.import_data('test', task, spec)
    print('labels: ', set(labels), labels.tolist().count(0), labels.tolist().count(1))
    print('Number of test sentences: {:,}\n'.format(len(sentences)))
    print('Number of test labels: {:,}\n'.format(len(labels)))

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []
    
    #print(model_id)
    
    tokenizer, model = u.load_hf(model_id, path_to_model=model_path) # Todo: load tokenizer from device, not from HF
    # For every sentence...
    for sent in sentences:
        # same as in training
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 64,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0).to(device)
    attention_masks = torch.cat(attention_masks, dim=0).to(device)
    labels = torch.tensor(labels).to(device)

    # Set the batch size.  
    batch_size = batch_size_  

    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

    #model = model_class.from_pretrained()
    model.eval()# Put model in evaluation mode
    model.to(device)

    # Tracking variables 
    predictions , true_labels = [], []
    
    save_dict = {"batch": [], "accuracy": [], "recall": [], "precision": [], "f1":[]}
    
    # Predict 
    i = 0
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        
        b_input_ids = b_input_ids.to(device) 
        b_input_mask = b_input_mask.to(device)
        b_labels = b_labels.to(device)
        
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, #token_type_ids=None,
                            attention_mask=b_input_mask)
        
        logits = outputs[0]
        
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Store predictions and true labels 
        predictions.extend(logits)
        true_labels.extend(label_ids)
        
        #logits_ = [0 if x>y else 1 for (x,y) in logits]
        #label_ids_ = label_ids.tolist()
        
        #batch_acc = accuracy_score(y_true=label_ids_, y_pred=logits_)
        #batch_rec = recall_score(y_true=label_ids_, y_pred=logits_)
        #batch_pre = precision_score(y_true=label_ids_, y_pred=logits_)
        #batch_f1 = f1_score(y_true=label_ids_, y_pred=logits_)
        
        #save_dict["batch"].append(i)
        #save_dict["accuracy"].append(batch_acc)
        #save_dict["recall"].append(batch_rec)
        #save_dict["precision"].append(batch_pre)
        #save_dict["f1"].append(batch_f1)
        
        i+=1    
        #print("batch result: accuracy = {}; recall = {}; precision = {}; f1 = {}".format(batch_acc,batch_rec, batch_pre, batch_f1 ))

    # print(label.sum(), len(df.label), (df.label.sum() / len(df.label) * 100.0))
    predictions = [0 if x>y else 1 for (x,y) in predictions]
    label_ids = label_ids.tolist()
    acc = accuracy_score(y_true=true_labels, y_pred=predictions)
    rec = recall_score(y_true=true_labels, y_pred=predictions)
    pre = precision_score(y_true=true_labels, y_pred=predictions)
    f1 = f1_score(y_true=true_labels, y_pred=predictions)
    save_dict["batch"].append('final')
    save_dict["accuracy"].append(acc)
    save_dict["recall"].append(rec)
    save_dict["precision"].append(pre)
    save_dict["f1"].append(f1)
    
    res_df = pd.DataFrame(save_dict)
    res_df.to_pickle(model_path + '/evaluation_'+spec)
    
    print('    DONE.')
    return acc, rec, pre, f1
    
all_model_paths = glob("res_models/models/*") # cloud also be restricted by hand
all_model_paths = [x for x in all_model_paths if 'IMDB' in x ] # only IMDB for now, because twitter is still in training

all_model_paths = [
    "res_models/models/IMDB_distbase_N_pro",
    "res_models/models/IMDB_albertbase_N_pro",
    "res_models/models/IMDB_bertlarge_mix_weat",
    "res_models/models/IMDB_bertlarge_mix_pro",
    "res_models/models/IMDB_albertlarge_mix_pro",
    "res_models/models/IMDB_bertlarge_N_pro",
    "res_models/models/IMDB_bertlarge_N_weat",
    "res_models/models/IMDB_robertabase_N_all",
    "res_models/models/IMDB_robertalarge_N_pro",
    "res_models/models/IMDB_albertbase_N_all",
    "res_models/models/IMDB_distbase_N_all",
    "res_models/models/IMDB_distbase_mix_pro",
    "res_models/models/IMDB_bertlarge_N_all",
    "res_models/models/IMDB_bertbase_original",
    "res_models/models/IMDB_albertbase_original",
    "res_models/models/IMDB_bertbase_N_all",
    "res_models/models/IMDB_albertlarge_N_all",
    "res_models/models/IMDB_albertlarge_N_pro",
    "res_models/models/IMDB_albertlarge_mix_weat",
    "res_models/models/IMDB_bertbase_N_weat",
    "res_models/models/IMDB_albertlarge_original",
    "res_models/models/IMDB_robertalarge_mix_all",
    "res_models/models/IMDB_albertbase_mix_weat",
    "res_models/models/IMDB_robertalarge_original",
    "res_models/models/IMDB_robertalarge_N_weat",
    "res_models/models/IMDB_bertbase_mix_weat",
    "res_models/models/IMDB_distbase_N_weat",
    "res_models/models/IMDB_robertalarge_mix_pro",
    "res_models/models/IMDB_robertabase_N_weat",
    "res_models/models/IMDB_albertbase_mix_pro",
    "res_models/models/IMDB_distbase_mix_all",
    "res_models/models/IMDB_robertalarge_N_all",
    "res_models/models/IMDB_bertlarge_mix_all",
    "res_models/models/IMDB_bertbase_mix_pro",
    "res_models/models/IMDB_distbase_mix_weat",
    "res_models/models/IMDB_robertabase_N_pro",
    "res_models/models/IMDB_robertabase_mix_weat",
    "res_models/models/IMDB_distbase_original",
    "res_models/models/IMDB_albertbase_N_weat",
    "res_models/models/IMDB_albertlarge_mix_all",
    "res_models/models/IMDB_robertabase_mix_pro",
    "res_models/models/IMDB_robertalarge_mix_weat",
    "res_models/models/IMDB_robertabase_original",
    "res_models/models/IMDB_bertlarge_original",
    "res_models/models/IMDB_albertbase_mix_all",
    "res_models/models/IMDB_bertbase_mix_all",
    "res_models/models/IMDB_albertlarge_N_weat",
    "res_models/models/IMDB_robertabase_mix_all",
    "res_models/models/IMDB_bertbase_N_pro"
    ]

all_model_paths_ = [
    ['res_models/models_BS/IMDB_bertbase_N_all_BS08'], #/e20do5_epoch4'] ,
    ['res_models/models_BS/IMDB_bertbase_N_all_BS16'], #/e20do5_epoch11'] ,
    ['res_models/models_BS/IMDB_bertbase_N_all_BS32'], #/e20do5_epoch19'] ,
    ['res_models/models_BS/IMDB_bertbase_N_pro_BS08'], #/e20do5_epoch4'] ,
    ['res_models/models_BS/IMDB_bertbase_N_pro_BS16'], #/e20do5_epoch11'] ,
    ['res_models/models_BS/IMDB_bertbase_N_pro_BS32'], #/e20do5_epoch5'] ,
    ['res_models/models_BS/IMDB_bertbase_N_weat_BS08'], #/e20do5_epoch6'] ,
    ['res_models/models_BS/IMDB_bertbase_N_weat_BS16'], #/e20do5_epoch10'] ,
    ['res_models/models_BS/IMDB_bertbase_N_weat_BS32'], #/e20do5_epoch19'] ,
    ['res_models/models_BS/IMDB_bertbase_mix_all_BS08'], #/e20do5_epoch13'] ,
    ['res_models/models_BS/IMDB_bertbase_mix_all_BS16'], #/e20do5_epoch18'] ,
    ['res_models/models_BS/IMDB_bertbase_mix_all_BS32'], #/e20do5_epoch12'] ,
    ['res_models/models_BS/IMDB_bertbase_mix_pro_BS08'], #/e20do5_epoch16'] ,
    ['res_models/models_BS/IMDB_bertbase_mix_pro_BS16'], #/e20do5_epoch18'] ,
    ['res_models/models_BS/IMDB_bertbase_mix_pro_BS32'], #/e20do5_epoch5'] ,
    ['res_models/models_BS/IMDB_bertbase_mix_weat_BS08'], #/e20do5_epoch17'] ,
    ['res_models/models_BS/IMDB_bertbase_mix_weat_BS16'], #/e20do5_epoch19'] ,
    ['res_models/models_BS/IMDB_bertbase_mix_weat_BS32'], #/e20do5_epoch11'] ,
    ['res_models/models_BS/IMDB_bertbase_original_BS08'], #/e20do5_epoch4'] ,
    ['res_models/models_BS/IMDB_bertbase_original_BS16'], #/e20do5_epoch11'] ,
    ['res_models/models_BS/IMDB_bertbase_original_BS32'], #/e20do5_epoch19'] ,
    ['res_models/models_BS/IMDB_distbase_mix_all_BS08'], #/e20do5_epoch17'] ,
    ['res_models/models_BS/IMDB_distbase_mix_all_BS16'], #/e20do5_epoch19'] ,
    ['res_models/models_BS/IMDB_distbase_mix_all_BS32'], #/e20do5_epoch14'] ,
    ['res_models/models_BS/IMDB_distbase_mix_pro_BS08'], #/e20do5_epoch19'] ,
    ['res_models/models_BS/IMDB_distbase_mix_pro_BS16'], #/e20do5_epoch19'] ,
    ['res_models/models_BS/IMDB_distbase_mix_pro_BS32'], #/e20do5_epoch19'] ,
    ['res_models/models_BS/IMDB_distbase_mix_weat_BS08'], #/e20do5_epoch15'] ,
    ['res_models/models_BS/IMDB_distbase_mix_weat_BS16'], #/e20do5_epoch19'] ,
    ['res_models/models_BS/IMDB_distbase_mix_weat_BS32'], #/e20do5_epoch19'] 
]

    
rtpt = RTPT(name_initials='SJ', experiment_name='evaluation', max_iterations=len(all_model_paths))
rtpt.start()

for path in all_model_paths:
    model_path=path
    for file in glob(path+ "/*"):
        if 'epoch' in file:
            model_path = file
    print(model_path)
    res_ = test(model_path)
    res_original = test(model_path, spec='original')
    rtpt.step()
        

# TODO: write function that collects all evaluations and creates a single data frame from all finals - no pytorch needed, only pandas work. 