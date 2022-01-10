# bias-bert

### set up
clone repository and cd into bias-bert  
install requirements.txt in your environment  

### get and prepare data 
`python res_data/IMDB_data_preparation_script.py | tee data_preparation.txt`  
`python cd res_data/twitter_data_preparation_script.py | tee data_preparation.txt`  

### train
Train the models with train.py. The script is called with three variables, which are (1) the task (i.e. "IMDB" or "Twitter"), (2) the defined model_id of the pretrained model (find a list of all options below) and (3) the data specification(s) (spec) that are used to train the model(s). Each specification determines a different subset of test and training data and results in one model.  

`python train.py [task] [model_id] [spec]` is the structure of the command, wheere all three variables need to be strings. Here are some examples:  
  
- `python train.py "Twitter" "bertbase" "all"` trains all specifications for that task and model  
- `python train.py "Twitter" "bertbase" "N_pro"` only trains one model with the N_pro data subset  
- `python train.py "Twitter" "bertbase" "N_pro N_all original"` trains three different models, one for each included data specification  


### possible variables
specs are `"N_pro"`, `"N_weat"`, `"N_all"`, `"mix_pro"`, `"mix_weat"`, `"mix_all"`, `"original"`;  
model_id can be `"bertbase"`, `"bertlarge"`, `"distbase"`, `"distlarge"`, `"robertabase"`, `"robertalarge"`, `"albertbase"`, `"albertlarge"`,  
which correspond to the pretrained [Hugging Face Models](https://huggingface.co/models) `bert-base-uncased`, `bert-large-uncased`, `distilbert-base-uncased`, `distilbert-large-uncased`, `roberta-base`, `roberta-large`, `albert-base-v2`, `albert-large-v2`.   

### Rate Experimentatl Samples 



### Analyse Results: Calculate and Plot Biases



### License 
cite paper here.  


### Resources 
- IMDB data  
- Stanford data  

