import os
import numpy as np
import pandas as pd 
import time, datetime, torch

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AlbertConfig


def time():
    return datetime.datetime.now().strftime('%m_%d_%H%M')


def import_data(tt, task, spec):
    if tt in ['train', 'training']:
        tt = 'train'
    elif tt in ['test']:
        tt ='test'
    else:
        print('Error: task type in data import does not fit')
        return False
    # load data 
    path = "res_data/{}_training/{}_{}_{}".format(task, task, spec, tt)
    # print(path)
    data = pd.read_pickle(path)
    
    #print('load {} data for {} task in specification {}.'.format(tt, task, spec))
    #print('Number of training sentences: {:,}\n'.format(data.shape[0]))
    #print(data.head(10))
     
    # modify data sets: from string labels to numerical labels 
    data.label = pd.factorize(data.label)[0]
    data.rename(columns={data.columns[1]:'text'}, inplace=True)
    # print(data.head(10))
    
    sentences = data.text.values
    labels = data.label.values
    x = [sentences, labels]
    # print(x)
    return x


def identify_task(name):
    if 'IMDB' in name:
        task = 'IMDB'
    elif 'Twitter' in name:
        task = 'Twitter'
    else:
        # print('IDENTIFY_TASK: error - something is wrong with the taks/ path')
        task = None
    return task


def identify_model_id(name):
    #return identify_spec(name, model_id=True)
    all_specs = ["bertbase", 'bertlarge', "distbase", "robertabase", "robertalarge", "albertbase", "albertlarge"]
    function_name = 'IDENTIFY_MODEL_ID'
    
    output = []
    for spec in all_specs:
        if '_{}_'.format(spec) in name:
            output.append(spec)
    if not output:
        print('{}: error - no item identified'.format(function_name))
        return
    elif len(output) == 1:
        # print('{}: one item identified'.format(function_name))
        # print(output[0])
        return output[0]
    else: 
        if len(output[0]) > len(output[1]):
            output = output[0] 
        else:
            output = output[1] 
        # print('{}: multiple items identified'.format(function_name))
        # print(output)
        return output
    

def identify_spec(name, model_id=False):
    all_specs = ["original", "N_pro", "N_weat", "N_all", "mix_pro", "mix_weat", "mix_all"]
    function_name = 'IDENTIFY_SPEC'
    #if model_id:
    #    all_specs = ["bertbase", 'bertlarge', "distbase", "robertabase", "robertalarge", "albertbase", "albertlarge"]
    #    function_name = 'IDENTIFY_MODEL_ID'
    output = []
    for spec in all_specs:
        if spec in name:
            output.append(spec)
    if not output:
        print('{}: error - no item identified'.format(function_name))
        return
    elif len(output) == 1:
        # print('{}: one item identified'.format(function_name))
        # print(output[0])
        return output[0]
    else: 
        print('{}: multiple items identified'.format(function_name), output)
        return output
    
    
########## ########## ########## ########## ########## 
# ++++++++++++++
# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
# ++++++++++++++
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
########## ########## ########## ########## ########## 

def check_path(path):
    '''
    check if the paths exists, else create 
    '''
    if not os.path.exists(path):
        os.makedirs(path)
        print("created path "+ path)
    return path

########## ########## ########## ########## ########## 

def load_hf(model_id, dropout=0.5, configuration=None, load_model=True, load_tokenizer=True, path_to_model=None):
    '''
    Loads the pretrained model and corresponding tokenizer from huggingface, both indicated by the model_name, i.e. model_id. 
    Function returns the tokenizer and the model. 

    '''
    model = None
    tokenizer = None
    
    # # # Configuration
    config_bert = BertConfig(
        hidden_dropout_prob = dropout,
        num_labels=2, )
        # output_hidden_states = False, # Whether the model returns all hidden-states. )
    config_bertL = BertConfig(
        hidden_dropout_prob = dropout,
        num_attention_heads = 16, 
        hidden_size = 1024,
        intermediate_size = 4096,
        num_labels=2, )
        # output_hidden_states = False, # Whether the model returns all hidden-states. )
    config_roberta = RobertaConfig(
        vocab_size = 50265,
        max_position_embeddings = 514,
        type_vocab_size = 1,
        hidden_dropout_prob = dropout,
        num_labels=2, )
        # output_hidden_states = False, # Whether the model returns all hidden-states. )
    config_robertaL = RobertaConfig(
        hidden_size = 1024,
        intermediate_size = 4096,
        num_attention_heads = 16,
        vocab_size = 50265,
        max_position_embeddings = 514,
        type_vocab_size = 1,
        hidden_dropout_prob = dropout,
        num_labels=2, )
        # output_hidden_states = False, # Whether the model returns all hidden-states. )
    config_distil = DistilBertConfig( 
        dropout = dropout,
        num_labels=2, )
        # output_hidden_states = False, # Whether the model returns all hidden-states. )
    config_Albert = AlbertConfig(
        hidden_size = 768, # 4069
        intermediate_size = 3072, # 16384,
        hidden_dropout_prob = dropout,
        num_labels=2, )
        #output_hidden_states = False, # Whether the model returns all hidden-states. )
    config_AlbertL = AlbertConfig(
        hidden_size = 1024, # 4069
        intermediate_size = 4096, # 16384,
        hidden_dropout_prob = dropout,
        num_labels=2, )
        #output_hidden_states = False, # Whether the model returns all hidden-states. )
    
    load_dict = {
        "bertbase": ["bert-base-uncased", BertTokenizer, BertForSequenceClassification, BertConfig, config_bert],
        "bertlarge": ["bert-large-uncased", BertTokenizer, BertForSequenceClassification, BertConfig, config_bertL],
        "distbase": ["distilbert-base-uncased", DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig, config_distil],
# gibt es nicht        "distlarge": ["distilbert-large-uncased", DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig, config_distil],
        "robertabase": ["roberta-base", RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig, config_roberta],
        "robertalarge": ["roberta-large", RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig, config_robertaL],
        "albertbase": ["albert-base-v2", AlbertTokenizer, AlbertForSequenceClassification, AlbertConfig, config_Albert],
        "albertlarge": ["albert-large-v2", AlbertTokenizer, AlbertForSequenceClassification, AlbertConfig, config_AlbertL]
    }
    # generic import 
    case = load_dict[model_id]
    
    if path_to_model: 
        print('load model from ' + path_to_model) 
        model = case[2].from_pretrained(path_to_model) #, num_labels=2)
        tokenizer = case[1].from_pretrained(path_to_model) #, num_labels=2)
        return tokenizer, model
    
        print('Function should end here. Something is going wrong')
    elif load_model: 
        model = case[2].from_pretrained(case[0], config=case[4])  
    if load_tokenizer: 
        tokenizer = case[1].from_pretrained(case[0], do_lower_case=True)

    return tokenizer, model



# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

#
#

def load_albert(model_id="albert-base-v2", load_model=True, load_tokenizer=True ):
    tokenizer = None
    Model = None
    configuration = AlbertConfig(
        hidden_size = 768, # 4069
        intermediate_size = 3072, # 16384,
        hidden_dropout_prob = 0.5,
        num_labels=2,
        #output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    tokenizer = AlbertTokenizer.from_pretrained(model_id)
    model = AlbertForSequenceClassification.from_pretrained(model_id, config=configuration)
    
    return tokenizert, model


