import sys, os, re
sys.path.append('./..')
from train_functions import check_path
import pandas as pd
import data_masking as masking
import numpy as np
import pickle

# where to safe data and tables? right here I suppose. Else modify 
path = ""
check_path(path + 'Twitter_training')
check_path('Twitter_raw_data')

########################################################################################
# Download Twitter Data
#os.system('gdown https://drive.google.com/uc?id=0B04GJPshIjmPRnZManQwWEdTZjg')
#os.system('unzip trainingandtestdata.zip')
#os.system('mv testdata.manual.2009.06.14.csv Twitter_raw_data/')
#os.system('mv training.1600000.processed.noemoticon.csv Twitter_raw_data/')
#os.system('mv trainingandtestdata.zip Twitter_raw_data/')

def clean_text(reviews):
    reviews = [re.sub('@[^\s]+','', line) for line in reviews]
    REPLACE_NO_SPACE = re.compile("[.;:!?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)|(\')")
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    return reviews

# read Twitter data
df = pd.read_csv('Twitter_raw_data/training.1600000.processed.noemoticon.csv', encoding='latin-1', names= ['sentiment', 'ID', 'date' ,'query' ,'user' ,'text_raw']) 

# Attantion!!! Compile with caution. This will change the partition of training/test set 
shuffled = df.sample(frac=1)
result = np.array_split(shuffled, 2)

assert(len(result[0]) == 800000)
assert(len(result[1]) == 800000)

with open(path + 'Twitter_raw_data/Twitter_train_raw', "wb") as fp:   #Pickling
    pickle.dump(result[0], fp)
with open(path + 'Twitter_raw_data/Twitter_test_raw', "wb") as fp:   #Pickling
    pickle.dump(result[1], fp)

train = result[0]
test = result[1]

# test that no neutrals are included
df_train = train[train['sentiment'] != 2]
df_test = test[test['sentiment'] != 2]
assert(df_test.shape == test.shape)
assert(df_train.shape == train.shape)

df_train_complete = result[0] # pd.read_pickle(path + 'Twitter_raw_data/Twitter_train_raw')
df_test_complete = result[1] # pd.read_pickle(path + 'Twitter_raw_data/Twitter_test_raw')

df_train_complete.ID = 'train_'+ df_train_complete['sentiment'].astype(str) +'_'+ df_train_complete['ID'].astype(str)
df_test_complete.ID = 'test_'+ df_test_complete['sentiment'].astype(str) +'_'+ df_test_complete['ID'].astype(str)

df_train_complete.insert(1, 'text', clean_text(df_train_complete.text_raw.tolist()), True)
df_test_complete.insert(1, 'text', clean_text(df_test_complete.text_raw.tolist()), True)

df_train = df_train_complete[['ID', 'text', 'sentiment']].rename({'sentiment': 'label'}, axis=1)
df_test = df_test_complete[['ID', 'text', 'sentiment']].rename({'sentiment': 'label'}, axis=1)

df_train.to_pickle(path + 'Twitter_training/Twitter_original_train')
df_test.to_pickle(path + 'Twitter_training/Twitter_original_test')

##### ##### #####
# Twitter - Step 1: Gender neutral data sets for training

# Mask all terms in Data 
df_train_ = df_train.copy()
df_test_ = df_test.copy()

masking.make_all_df(df_train_)
masking.make_all_df(df_test_)
masking.check_df(df_test_)
masking.check_df(df_train_)

assert(df_train_.shape == (800000, 18))
assert(df_test_.shape == (800000, 18))

# Safe whole table (large)
df_train_.to_pickle(path + "Twitter_l_train")
df_test_.to_pickle(path + "Twitter_l_test")

# checkpoint to reenter the code after having AssertErrors
#df_train_ = pd.read_pickle(path + "Twitter_l_train")
#df_test_ = pd.read_pickle(path + "Twitter_l_test")

print('(got assert error here) shape of df with only samples that include terms: ')
print(df_train_[df_train_['count_total']> 0].shape) 
print('assert', df_train_.shape == (800000, 18))

##### ##### #####
# Twitter - Step 2: Safe training and test dataframes for different training conditions.
# neutral
for spec in ['_all', '_pro', '_weat']:
    df_train_[['ID', 'text'+spec+'_N', 'label']].to_pickle(path + 'Twitter_training/Twitter_N'+spec+'_train')
    df_test_[['ID', 'text'+spec+'_N', 'label']].to_pickle('Twitter_training/Twitter_N'+spec+'_test')

# mixed M+F
for spec in ['_all', '_pro', '_weat']: 
    m_tr = df_train_[['ID', 'text'+spec+'_M', 'label']].rename(columns={'text'+spec+'_M': 'text'})
    f_tr = df_train_[['ID', 'text'+spec+'_F', 'label']].rename(columns={'text'+spec+'_F': 'text'})
    tr = m_tr.append(f_tr)
    tr.to_pickle(path + 'Twitter_training/Twitter_mix' + spec + '_train') 
    
    m_te = df_test_[['ID', 'text'+spec+'_M', 'label']].rename(columns={'text'+spec+'_M': 'text'})
    f_te = df_test_[['ID', 'text'+spec+'_F', 'label']].rename(columns={'text'+spec+'_F': 'text'})
    te = m_te.append(f_te)
    te.to_pickle(path + 'Twitter_training/Twitter_mix' + spec + '_test') 
    
    print('assert', tr.shape == (1600000, 3))
    print('assert', te.shape == (1600000, 3))

# Create Data Sets with no only samples that do not contain any term of the dict
# no term sample
df_train_no_pron = df_train_[df_train_['count_total'] == 0][['ID', 'text', 'label']]
print('assert', df_train_no_pron.shape == (691200, 3))
df_test_no_pron = df_test_[df_test_['count_total'] == 0][['ID', 'text', 'label']]
print('assert', df_test_no_pron.shape == (691347, 3))

df_train_no_weat = df_train_[df_train_['count_weat'] == 0][['ID', 'text', 'label']]
print('assert', df_train_no_weat.shape == (725866, 3))
df_test_no_weat = df_test_[df_test_['count_weat'] == 0][['ID', 'text', 'label']]
print('assert', df_test_no_weat.shape == (725551, 3))

df_train_no_all = df_train_[df_train_['count_prons'] == 0][['ID', 'text', 'label']]
print('assert', df_train_no_all.shape == (750334, 3))
df_test_no_all = df_test_[df_test_['count_prons'] == 0][['ID', 'text', 'label']]
print('assert', df_test_no_all.shape == (750334, 3))


df_train_no_pron.to_pickle(path + 'Twitter_training/Twitter_no_pron_train')
df_test_no_pron.to_pickle(path + 'Twitter_training/Twitter_no_pron_test')
df_train_no_weat.to_pickle(path + 'Twitter_training/Twitter_no_weat_train')
df_test_no_weat.to_pickle(path + 'Twitter_training/Twitter_no_weat_test')
df_train_no_all.to_pickle(path + 'Twitter_training/Twitter_no_all_train')
df_test_no_all.to_pickle(path + 'Twitter_training/Twitter_no_all_test')

# Create Data Sets with no only samples that do contain a minimal number of term of the dict
min_term_count = 1
df_train__ = df_train_.rename(columns={'count_total': 'count_all', 'count_prons': 'count_pro'})
df_test__ = df_test_.rename(columns={'count_total': 'count_all', 'count_prons': 'count_pro'})

for spec in ['_all', '_pro', '_weat']:
    df_train_MIN = df_train__[df_train__['count'+spec] >= min_term_count]
    df_test_MIN = df_test__[df_test__['count'+spec] >= min_term_count]
    # all
    df_train_MIN[['ID', 'text', 'label']].to_pickle(path + 'Twitter_training/Twitter_MIN' + spec + '_test')
    df_test_MIN[['ID', 'text', 'label']].to_pickle(path + 'Twitter_training/Twitter_MIN' + spec + '_train')
    # neutral M+F
    df_train_MIN[['ID', 'text'+spec+'_N', 'label']].to_pickle(path + 'Twitter_training/Twitter_MIN_N'+spec+'_train')
    df_test_MIN[['ID', 'text'+spec+'_N', 'label']].to_pickle(path + 'Twitter_training/Twitter_MIN_N'+spec+'_test')
    # mixed
    m_tr = df_train_MIN[['ID', 'text'+spec+'_M', 'label']].rename(columns={'text'+spec+'_M': 'text'})
    f_tr = df_train_MIN[['ID', 'text'+spec+'_F', 'label']].rename(columns={'text'+spec+'_F': 'text'})
    tr = m_tr.append(f_tr)
    tr.to_pickle(path + 'Twitter_training/Twitter_MIN_mix' + spec + '_train') 
    
    m_te = df_test_MIN[['ID', 'text'+spec+'_M', 'label']].rename(columns={'text'+spec+'_M': 'text'})
    f_te = df_test_MIN[['ID', 'text'+spec+'_F', 'label']].rename(columns={'text'+spec+'_F': 'text'})
    te = m_te.append(f_te)
    te.to_pickle(path + 'Twitter_training/Twitter_MIN_mix' + spec + '_test') 
    
    print('assert', tr.shape[1] == 3)
    print('assert', te.shape[1] == 3)
    
