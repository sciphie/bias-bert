import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch, logging, transformers, sys, os, datetime
from transformers import ( 
    TrainingArguments, 
    Trainer,
    BertTokenizer, 
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    EarlyStoppingCallback,
)
from glob import glob
 

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


def load_hf(model_id, load_model=True, path_to_model=None):
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
    
    # bertbase 
    if model_id == "bertbase" or model_id == "bertlarge": #"bert-base-uncased"
        from transformers import BertTokenizer, BertForSequenceClassification
        print('successfully loaded bert-base-uncased with BertTokenizer, BertForSequenceClassification')
        if path_to_model: 
            print('load model from ' + path_to_model) 
            return BertForSequenceClassification.from_pretrained(path_to_model, num_labels=2)
            print('Function should end here. Something is going wrong')
        elif model_id == "bertlarge":
            model_name = "bert-large-uncased"
        else:
            model_name = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        if load_model: 
            model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)    
  
    # distilbert
    elif model_id == "distbase" or model_id == "distlarge":
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        if path_to_model:
            print('load model from ' + path_to_model) 
            return DistilBertForSequenceClassification.from_pretrained(path_to_model, num_labels=2)
            print('Function should end here. Something is going wrong')
        if model_id == "distbase": 
            model_name = "distilbert-base-uncased"
        else: 
            model_name = "distilbert-large-uncased"
       
        print('successfully loaded {} with DistilBertTokenizer, DistilBertForSequenceClassification'.format(model_name))
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        if load_model: 
            model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)  
    # roberta 
    elif model_id == "robbase" or model_id == "roblarge":
        from transformers import RobertaTokenizer, RobertaForSequenceClassification
        if path_to_model: 
            print('load model from ' + path_to_model) 
            return RobertaForSequenceClassification.from_pretrained(path_to_model, num_labels=2)
            print('Function should end here. Something is going wrong')
        if model_id == "robertabase": 
            model_name = "roberta-base"
        else: 
            model_name = "roberta-large"
        
        print('successfully loaded {} with RobertaTokenizer, RobertaForSequenceClassification'.format(model_name))
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        if load_model: 
            model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)  
    # Albert
    elif model_id == "albertbase" or model_id == "albertlarge":
        from transformers import AlbertTokenizer, AlbertForSequenceClassification
        if path_to_model: 
            print('load model from ' + path_to_model) 
            return AlbertForSequenceClassification.from_pretrained(path_to_model, num_labels=2)
            print('Function should end here. Something is going wrong')                                                                  
        if model_id == "albertbase": 
            model_name = "albert-base-v2"
        else: 
            model_name = "albert-large-v2"
        
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
    
    print("{} : compute_metrics - accuracy: {}; precision: {}; recall: {}; f1: {}".format(timestamp(True), accuracy, precision, recall, f1))
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}



def train(task, model_id, spec, tokenizer=None, model=None, eval_steps_=500, per_device_train_batch_size_=8, per_device_eval_batch_size_=8, num_train_epochs_=100):  
    '''
    todo
    '''
    # I removed the logging block for the moment as I am not using it right now anyway 
    
    print('{}: params: spec= {}; eval_steps_={}; per_device_train_batch_size_={}; per_device_eval_batch_size_={}; num_train_epochs_={}'.format(__name__, spec, eval_steps_, per_device_train_batch_size_, per_device_eval_batch_size_, num_train_epochs_))
    
    ### ### ### ### ###
    if not model or not tokenizer:
        tokenizer, model = load_hf(model_id)
        
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
    
    output_path = check_path('res_models/{}/{}/output_{}/'.format(task, model_id, spec)) # tu.
    #output_path = "res_models/{}/{}/output_{}/".format(task, model_id, spec)

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
        logging_dir='./res_models/runs/{}_{}_{}_{}'.format(task, model_id, spec, timestamp())
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics= compute_metrics, # tu.
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Train pre-trained model
    trainer.train()


def calc_acc(spec, tokenizer, model_id, task="foo",  restricted_test_set = False):
    # ----- Load trained model -----#   
    path_ = "res_models/{}/{}/output_{}/".format(task, model_id, spec)
    print(path_)
    #filenames = next(walk(path_), (None, [], None))[1]  # [] if no file
    filenames = glob(path_ + '*')
    filenames.reverse()
    print(filenames)
    print('#################')
    print(filenames[0])    
    model_path = filenames[0]
    print(model_path)
    model = load_hf(model_id, path_to_model=model_path) # DistilBertForSequenceClassification.from_pretrained(, num_labels=2) 
    
    # ----- Load test data -----#
    if restricted_test_set: 
        print(timestamp(True) + 'calculate accuracy with RESTRICTED test_set')
        test_data = pd.read_pickle('res_data/{}_training/{}_{}_test'.format(task,task,spec))
    else:
        print(timestamp(True)+ 'calculate accuracy with ALL test samples')
        test_data = pd.read_pickle('res_data/' + task + '_l_test')
    test_data.label = pd.factorize(test_data.label)[0]
    if 'text' in test_data.columns[1]:
        test_data.rename(columns={test_data.columns[1]:'text'}, inplace=True)
    else:
        print("ERROR: compute_metrics - wrong column renamed. This is not the text column")
        # log.error(__name__ + ": compute_metrics - " + 'wrong column renamed. This is not the text column')          
        
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

