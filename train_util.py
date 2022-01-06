import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch, logging, transformers 
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
from os import walk
import sys, os, datetime


# see https://huggingface.co/docs/transformers/main_classes/logging
transformers.utils.logging.set_verbosity_info
transloggers = transformers.utils.logging.get_logger
# create a new streamhandler to also see the loggings in the shell (stdout) 
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(handler)


if torch.cuda.is_available():
    logging.info('Current GPU device : ' + str(torch.cuda.current_device()))
    print('Current GPU device : ' + str(torch.cuda.current_device()))
else: 
    logging.warning('No GPU available')
    print('No GPU available')


def timestamp(time=False):
    '''
    I use this to always have a current time stamp
    '''
    if time: 
        return datetime.datetime.now().strftime('%m_%d_%H%M')
    else:
        return datetime.datetime.now().strftime('%m_%d_%Y')
    
def check_path(path):
    '''
    check if the paths exists, else create 
    '''
    if not os.path.exists(path):
        os.makedirs(path)
        print("created path "+ path)
    return path


def load_hf(model_id, load_model=True):
    '''
    Loads the pretrained model and corresponding tokenizer from huggingface, both indicated by the model_name, i.e. model_id. Function returns the tokenizer and the model. 
    If load_model==False 
    
    #model_name, model_id = "bert-base-uncased", "bertbase"
    #model_name, model_id = "bert-large-uncased", 'bertlarge'
    #model_name, model_id = "distilbert-base-uncased", "distbase"
    #model_name, model_id = "distilbert-large-uncased", "distlarge" 
    #model_name, model_id = "roberta-base", "robertabase"
    #model_name, model_id = "roberta-large", "robertalarge"
    #model_name, model_id = "albert-base-v2", "albertbase"
    #model_name, model_id = "albert-large-v2", "albertlarge"
    --- Todo --- #model_name, model_id = "gpt2", "gpt2"
        '''
    model = None
    from transformers import BertForSequenceClassification
    
    if model_id == "bertbase": #"bert-base-uncased"
        from transformers import BertTokenizer
        print('successfully loaded bert-base-uncased with BertTokenizer, BertForSequenceClassification')
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        if load_model: 
            model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)    
    
    elif model_id == "bertlarge": #"bert-large-uncased"
        from transformers import BertTokenizer
        print('successfully loaded bert-large-uncased with BertTokenizer, BertForSequenceClassification')
        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        if load_model: 
            model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=2)     
    # distilbert
    elif model_id == "distbase" or model_id == "distlarge":
        if model_id == "distbase": 
            model_name = "distilbert-base-uncased"
        else: 
            model_name = "distilbert-large-uncased"
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        print('successfully loaded {} with DistilBertTokenizer, DistilBertForSequenceClassification'.format(model_name))
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        if load_model: 
            model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)  
    # roberta 
    elif model_id == "robbase" or model_id == "roblarge":
        if model_id == "robertabase": 
            model_name = "roberta-base"
        else: 
            model_name = "roberta-large"
        from transformers import RobertaTokenizer, RobertaForSequenceClassification
        print('successfully loaded {} with RobertaTokenizer, RobertaForSequenceClassification'.format(model_name))
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        if load_model: 
            model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)  
    # Albert
    elif model_id == "albertbase" or model_id == "albertlarge":
        if model_id == "albertbase": 
            model_name = "albert-base-v2"
        else: 
            model_name = "albert-large-v2"
        from transformers import AlbertTokenizer, AlbertForSequenceClassification
        print('successfully loaded {} with AlbertTokenizer, AlbertForSequenceClassification'.format(model_name))
        tokenizer = AlbertTokenizer.from_pretrained(model_name)
        if load_model: 
            model = AlbertForSequenceClassification.from_pretrained(model_name, num_labels=2)  
    
    # ToDO: gpt and others maybe    
    else:
        print("train_util:load_pretrained: model type not recognised by name")
        return None
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


def compute_metrics(p):# ,log=logging):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)
    
    print("{} : compute_metrics - " + "accuracy: {}; precision: {}; recall: {}; f1: {}".format(timestamp(True), accuracy, precision, recall, f1))
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def calc_acc(spec, tokenizer, model_id, task="foo",  restricted_test_set = False, log=logging):
    # ----- Load trained model -----#   
    filenames = next(walk("finetuning_"+ model_id + "/output_"+spec), (None, [], None))[1]  # [] if no file
    filenames.reverse()
    print(filenames[0])    
    model_path = "finetuning_"+ model_id + "/output_" + spec + "/" + filenames[0]
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2) 
    
    # ----- Load test data -----#
    if restricted_test_set: 
        print(timestamp(True) + 'calculate accuracy with RESTRICTED test_set')
        test_data = pd.read_pickle('../resources/' + task + '_training/' + task + '_' + spec + '_test')
    else:
        print(timestamp(True)+ 'calculate accuracy with ALL test samples')
        test_data = pd.read_pickle('../resources/' + task + '_l_test')
    test_data.label = pd.factorize(test_data.label)[0]
    if 'text' in test_data.columns[1]:
        test_data.rename(columns={test_data.columns[1]:'text'}, inplace=True)
    else:
        log.error(__name__ + ": compute_metrics - " + 'wrong column renamed. This is not the text column')          
        
    # ----- Predict -----#
    X_test = list(test_data["text"])
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)

    # Create torch dataset
    test_dataset = Dataset(X_test_tokenized)

    # Define test trainer
    test_trainer = Trainer(model)

    # Make prediction
    raw_pred, _, _ = test_trainer.predict(test_dataset)
    
    return(compute_metrics([raw_pred,list(test_data["label"])]))

'''
    y_pred = np.argmax(raw_pred, axis=1)
    y_true = list(test_data["label"])

    acc = [x==y for x, y in zip(y_pred, y_true)]
    acc = acc.count(True)/ len(acc)
    sk_acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    return(spec,sk_acc, f1, acc ==sk_acc, acc)
'''
