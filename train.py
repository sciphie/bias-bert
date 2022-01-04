'''
call this script with python train [task] [model_id] [spec], where task, model_od and spec need to be a valid string
so e.g. python train "IMDB" "bert-base-uncased" "N_pro"
'''

import sys, os
vars = sys.argv
from os import walk
import train_util as tu  
specs = ["N_pro", "N_weat", "N_all", "mix_pro", "mix_weat", "mix_all", "original"]

assert(len(vars) == 3)
if len(vars) != 3:
    print("something's wrong with the parameters here. Check that, please.")
    print("call this script with python train [task] [model_id] [spec], where task, model_od and spec need to be a valid string. So e.g. python train 'IMDB' 'bert-base-uncased' 'N_pro'")
    sys.exit(1)
else:
    task = vars[0]
    model_id = vars[1]
    spec = vars[2]
    print('called train.py {} {} {}'.format(task, model_id, spec))
    
###

def train(spec, eval_steps_=500, per_device_train_batch_size_=8, per_device_eval_batch_size_=8, num_train_epochs_=3):  
    '''
    todo
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
    
    #    logger.info(__name__ + ': train '+spec+ "; eval_steps="+str(eval_steps_)+ "; per_device_train_batch_size="+ str(per_device_train_batch_size_) +"; per_device_eval_batch_size="+ str(per_device_eval_batch_size_)+ "; num_train_epochs=" + str(num_train_epochs_)) 
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
    
    train_dataset = tu.Dataset(X_train_tokenized, y_train)
    val_dataset = tu.Dataset(X_val_tokenized, y_val)
    
    output_path = tu.check_path('res_models/{}/{}/output_{}'.format(task, model_id, spec))

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
        compute_metrics=tu.compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train pre-trained model
    trainer.train()

    
def acc_df():
    '''
    todo 
    '''
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
    df_acc.to_pickle(tu.check_path('res_models/accuracies/acc_' + task + '_' + model_id))
    print('\n' + __name__+ ':')
    print(df_acc)
    return df_acc    

# model_id can be "bertbase", 'bertlarge', "distbase", "distlarge", "robertabase", "robertalarge", "albertbase", "albertlarge"
tokenizer, model = tu.load_hf(model_id)
    
for spec in specs:
    train(spec)

df_acc_ = acc_df(task)
print(df_acc_) 
