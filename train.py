'''
call this script with python train [task] [model_id] [spec], where task, model_od and spec need to be a valid string
so e.g. python train "IMDB" "bert-base-uncased" "N_pro"
'''

import sys, os
#sys.path.append("ources")
from os import walk

import train_util as tu  
specs = ["N_pro", "N_weat", "N_all", "mix_pro", "mix_weat", "mix_all", "original"]

assert(len(sys.argv) == 3)
if len(sys.argv) != 3:
    print("something's wrong with the parameters here. Check that, please.")
    print("call this script with python train [task] [model_id] [spec], where task, model_od and spec need to be a valid string. So e.g. python train 'IMDB' 'bert-base-uncased' 'N_pro'")
    sys.exit(1)

task = sys.argv[0]
model_id = sys.argv[1]
spec = sys.argv[2]

###

def train(spec, eval_steps_=500, per_device_train_batch_size_=8, per_device_eval_batch_size_=8, num_train_epochs_=3):  
    '''
    
    '''
    ### TODO : Fix mal den logger scheiß hier. ###
    logger = logging.getLogger('tensorflow')
    logger.setLevel(logging.INFO)
    
    #logger = logging.getLogger(task + '_' + model_id)
    #logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(spec +"_"+ '%(asctime)s: %(levelname)s: %(name)s: %(message)s')
    file_handler = logging.FileHandler("logs/log_" + task + "_" + model_id + ".log")
    file_handler.setFormatter(formatter)
    
    file_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler) 
    
    logger.info(__name__ +': params: spec=' + spec + 
                '; eval_steps_= ' + str(eval_steps_) + 
                '; per_device_train_batch_size_=' + str(per_device_train_batch_size_) +
                '; per_device_eval_batch_size_=' + str(per_device_eval_batch_size_) + 
                '; num_train_epochs_=' + str(num_train_epochs_))
    ### ### ### ### ### 
    
    # load data set
    train_set_path = 'res_data/' + task + '_training/' + task + '_' + spec + '_train'
    test_set_path = 'res_data/' + task + '_training/' + task + '_' + spec + '_test'
    df_train = pd.read_pickle(train_set_path)
    df_test = pd.read_pickle(test_set_path)
    
    logger.info(__name__ +': successfully loaded --- ' + train_set_path + '; ' + test_set_path)
    
    # modify data sets 
    for df in [df_train, df_test]:
        df.label = pd.factorize(df.label)[0]
        df.rename(columns={df.columns[1]:'text'}, inplace=True)
    
    X = list(df_train["text"])
    y = list(df_train["label"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=11)
    
    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
    X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)
    
    train_dataset = tu.Dataset(X_train_tokenized, y_train)
    val_dataset = tu.Dataset(X_val_tokenized, y_val)
    
    logger.info(__name__ + ': train '+spec+ "; eval_steps="+str(eval_steps_)+ "; per_device_train_batch_size="+ str(per_device_train_batch_size_) +"; per_device_eval_batch_size="+ str(per_device_eval_batch_size_)+ "; num_train_epochs=" + str(num_train_epochs_)) 
    # Define Trainer
    args = TrainingArguments(
        output_dir='res_models/'+ task + '/' + model_id + "/output_" + spec,
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
        compute_metrics=tu.compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Train pre-trained model
    trainer.train()

    
def acc_df():
    df_acc = pd.DataFrame()
    for spec in specs: 
        foo = tu.calc_acc(spec, tokenizer, model_id, task, True)
        foo['data set'] = 'spec'    
        bar = tu.calc_acc(spec, tokenizer, model_id, task, False)
        bar['data set'] = 'all'
        foo['spec'] = spec
        bar['spec'] = spec
        df_acc = df_acc.append(foo, ignore_index = True)
        df_acc = df_acc.append(bar, ignore_index = True)
        print(df_acc) 
    df_acc.to_pickle('res_models/accuracies/acc_' + task + '_' + model_id)
    return df_acc    

# model_id can be "bertbase", 'bertlarge', "distbase", "distlarge", "robertabase", "robertalarge", "albertbase", "albertlarge"
tokenizer, model = tu.load_hf(model_id)
    
for spec in specs:
    train(spec)

df_acc_ = acc_df('IMDB')
print(df_acc_) 
