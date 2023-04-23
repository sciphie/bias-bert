# bias-bert

### Set Up
clone repository and cd into bias-bert  
install requirements.txt in your environment  


### Get and Prepare Data 
`python res_data/IMDB_data_preparation_script.py | tee data_prep.txt`  
`python cd res_data/twitter_data_preparation_script.py | tee data_prep.txt`  


### Train
Train the models with train_pytorch.py. In the script, three variables are specified: (1) the task (i.e. "IMDB" or "Twitter"), (2) the defined model_id of the pretrained model (find a list of all options below), and (3) the data specification(s) (spec) that are used to train the model(s). Each specification determines a different subset of test and training data and results in one model. 
Further training variables are defined in train().

Specify the variables directly in the script before calling train() or call train() with the corresponding function variables, e.g., `train(task='Twitter', model_id='bertlarge', spec='mix_pro', lr_in=2e-5, batch_s=16, run="ex_Tw_LR", name_addition='LR2')`

Trained models were evaluated with evaluate_pytorch.py and evaluate.ipynb (accuracy, f1 score, ...). 


### Possible Variables
specs are `"N_pro"`, `"N_weat"`, `"N_all"`, `"mix_pro"`, `"mix_weat"`, `"mix_all"`, `"original"`;  
model_id can be `"bertbase"`, `"bertlarge"`, `"distbase"`, `"distlarge"`, `"robertabase"`, `"robertalarge"`, `"albertbase"`, `"albertlarge"`,  
which correspond to the pretrained [Hugging Face Models](https://huggingface.co/models) `bert-base-uncased`, `bert-large-uncased`, `distilbert-base-uncased`, `distilbert-large-uncased`, `roberta-base`, `roberta-large`, `albert-base-v2`, `albert-large-v2`.   


### Rate Experimental Samples 
Rate gender samples with the trained model by calling `rate()` in `rate.py`, e.g.,  
`rate('IMDB', 'bertbase', 'original', 'weat')`

The ratings are saved in pandas data frames as pickle into `res_results/`. This data is needed to calculate the biases. 


### Analyse Results: Calculate and Plot Biases
Tables and Plots were created with `res_plots/biases.ipynb` and `res_plots/tables.ipynb`.


### Reference
This work has been published in:  
Jentzsch, S. F., & Turan, C. (2022). **Gender Bias in BERT-Measuring and Analysing Biases through Sentiment Rating in a Realistic Downstream Classification Task.** *GeBNLP 2022*, 184.


### Resources 
- IMDB data  
- Stanford data  
