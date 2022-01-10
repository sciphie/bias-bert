# bias-bert

### set up
clone repository and cd into bias-bert  
install requirements.txt in your environment  

### get and prepare data 
'python res_data/IMDB_data_preparation_script.py | tee data_preparation.txt'  
python cd res_data/twitter_data_preparation_script.py | tee data_preparation.txt  

### train
to train type 
python train.py 'Twitter' "bertbase" all



### possible variables
spec are "N_pro", "N_weat", "N_all", "mix_pro", "mix_weat", "mix_all", "original";  
model_id can be "bertbase", 'bertlarge', "distbase", "distlarge", "robertabase", "robertalarge", "albertbase", "albertlarge"

