# Standard imports 
import logging, re, pickle, os, nltk, random #, en_core_web_sm, spacy
logging.basicConfig(level=logging.INFO)
from term_lists import *

logging.info("successfully imported the latest version of data_masking.") 
print("successfully imported the latest version of data_masking.") 
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# utility functions
def add_space(word):
    return ' ' + word + ' '


# ----------------------------------------------------------------------------------------------- #
def count_terms(text, terms=all_terms):
    res = dict.fromkeys(terms, 0)
    for elem in terms:
        res[elem] = text.count(add_space(elem))
    return res


def mask_byDict(review, terms):
    '''
    mask_byDict: Mask terms in a text
    args
        review (str): Text
        terms (dict): tems. Mask kes by value.
    return tuple [(str) new masked review text, (dict) term occurances]
    '''
    count_dict = {}
    for word, initial in terms.items():
        # count_dict[word] = len(re.findall(add_space(word), review))
        review = review.replace(add_space(word), add_space(initial))
    return review

def make_male(review):
    return mask_byDict(review, terms_f2m)

def make_female(review):
    return mask_byDict(review, terms_m2f)

def make_neutral(text, terms=all_terms):
    for elem in terms:
        text = text.replace(add_space(elem), ' ')
    return text 

def make_all(l, fun):
    reviews = []
    freqs = []
    for elem in l:
        rev, freq = fun(elem)
        reviews.append(rev)
        freqs.append(freq)
    return reviews, freqs

def make_all_df(df):
    '''
    +++ Description +++
    args
        df with columns 'text' (str) and 'label' (int)
    '''
    texts = df.text.tolist()
    df["count_table"] = [count_terms(e) for e in texts]
    df["count_total"] = [sum(e.values()) for e in df["count_table"].tolist()]
    df["count_table_weat"] = [count_terms(e, all_weat) for e in texts]
    df["count_weat"] = [sum(e.values()) for e in df["count_table_weat"].tolist()]
    df["count_prons"] = [sum([e[pronoun] for pronoun in all_prons]) for e in df["count_table"].tolist()] 
    df["len"] = [len(e.split()) for e in texts]
    print(' make_all_df: finish counts and length')
    # ---
    df["text_all_M"] = [mask_byDict(e, terms_f2m) for e in texts]
    print(' make_all_df: finish text_all_M')
    df["text_all_F"] = [mask_byDict(e, terms_m2f) for e in texts]
    print(' make_all_df: finish text_all_F')
    df["text_all_N"] = [make_neutral(e, all_terms)for e in texts]
    print(' make_all_df: finish text_all_N')
    # ---
    df["text_weat_M"] = [mask_byDict(e, weat_f2m) for e in texts]
    print(' make_all_df: finish text_weat_M')
    df["text_weat_F"] = [mask_byDict(e, weat_m2f) for e in texts]
    print(' make_all_df: finish text_weat_F')
    df["text_weat_N"] = [make_neutral(e, all_weat) for e in texts]
    print(' make_all_df: finish text_weat_N')
    # ---
    df["text_pro_M"] = [mask_byDict(e, prons_f2m) for e in texts]
    print(' make_all_df: finish text_pro_M')
    df["text_pro_F"] = [mask_byDict(e, prons_m2f) for e in texts]
    print(' make_all_df: finish text_pro_F')
    df["text_pro_N"] = [make_neutral(e, all_prons) for e in texts]
    print(' make_all_df: finish text_pro_N')
    
def check_df(foo): 
    c1 = all(foo[foo['count_total'] >= foo['count_weat']])
    c2 = all(foo[foo['count_total'] >= foo['count_prons']])
    # there is "best man" and "best men" in the lage dict. this is why it is ">=" instead of "==" 
    # ok that does not work either, due to "paper boy" and the only cover terms. judt skip c3
    # c3 = all([x >= y for x,y in list(zip( [len(s.split()) for s in foo['text_all_M'].tolist()], [len(s.split()) for s in foo['text_all_F'].tolist()] ))]) 
    c4 = all([x == y for x,y in list(zip( [len(s.split()) for s in foo['text_weat_M'].tolist()], [len(s.split()) for s in foo['text_weat_F'].tolist()] ))]) 
    c5 = all([x == y for x,y in list(zip( [len(s.split()) for s in foo['text_pro_M'].tolist()], [len(s.split()) for s in foo['text_pro_F'].tolist()] ))]) 
    c6 = all([x >= y for x,y in list(zip( foo['len'].tolist(), [len(s.split()) for s in foo['text_all_N'].tolist()] ))]) 
    c7 = all([x >= y for x,y in list(zip( foo['len'].tolist(), [len(s.split()) for s in foo['text_weat_N'].tolist()] ))]) 
    c8 = all([x >= y for x,y in list(zip( foo['len'].tolist(), [len(s.split()) for s in foo['text_pro_N'].tolist()] ))]) 
    if c1 and c2 and c4 and c5 and c6 and c7 and c8:
        print('all tests ok')
        print('tested dataframe - everything is fine')
    else:
        logging.error('something is wrong in your DataFrame') 
        print('Error: something is wrong in your DataFrame') 
        print(c1,c2,c4,c5,c6,c7,c8)



# ----------------------------------------------------------------------------------------------- #
# ------------------------------------ Names ---------------------------------------------------- #
# Todo: NER does not work properly. So macht das keinen Sinn. 
# Not used in the latest implementation
'''
def mask_names_list(text_ls, names_a=m_names, names_b=f_names):
    male_text = []    # Create empty column for male reviews
    female_text = []    # Create empty column for female reviews
    replaced_Names = [] # Create empty column for number of replaced names per review
    replaced_names_all = 0    # counter for replaced names for the whole data set
    replaced_names_rev = 0    # counter for replaced names for each review
    
    nlp = spacy.load("en_core_web_sm")
    masked_ids = []
    
    i = 0
    for text in text_ls: # each review in dataframe
        if (not i==0) and (i%(len(text_ls)/10)==0):
            print(str((len(text_ls)/100*i)) + ' prozent geschafft')
        rev_m = text
        rev_f = text
        doc = nlp(text)
        include_name = False
        replaced_names_rev = 0
        
        for ent in doc.ents: # each entity in review
            if ent.label_ == 'PERSON' and len(ent.text)>2: # if that entity is a person
                replaced_names_all += 1 # increas counters
                replaced_names_rev += 1
                
                include_name = True 
                rev_m = rev_m.replace(ent.text, random.choice(names_a)) # replace with random male ...
                rev_f = rev_f.replace(ent.text, random.choice(names_b)) # ... and female name
            if ent.label_ == 'PERSON' and not len(ent.text)>2: # if that entity is a person
                print(ent.text)
        male_text.append(rev_m)
        female_text.append(rev_f)
        replaced_Names.append(replaced_names_rev)
        
        if include_name:
            masked_ids.append(i)
        i+=1

    #logging.info('one done')
    return [male_text, female_text, replaced_Names, replaced_names_all, masked_ids]


def mask_names(review_df, names_a=m_names, names_b=f_names):
    new_df = review_df.copy()
    new_df['text_na_M'] = ['']*len(new_df)    # Create empty column for male reviews
    new_df['text_na_F'] = ['']*len(new_df)    # Create empty column for female reviews
    new_df['Replaced Names'] = [0]*len(new_df) # Create empty column for number of replaced names per review
    replaced_names_all = 0    # counter for replaced names for the whole data set
    replaced_names_rev = 0    # counter for replaced names for each review
    
    nlp = spacy.load("en_core_web_sm")
    masked_ids = []
    
    i = 0
    for index, row in review_df.iterrows(): # each review in dataframe
        rev_m = row['text']
        rev_f = row['text']
        doc = nlp(row['text'])
        include_name = False
        replaced_names_rev = 0
        
        for ent in doc.ents: # each entity in review
            
            #print(ent.label_)
            if ent.label_ == 'PERSON' and len(ent)>3: # if that entity is a person
                replaced_names_all += 1 # increas counters
                replaced_names_rev += 1
                
                include_name = True 
                rev_m = rev_m.replace(ent.text, random.choice(names_a)) # replace with random male ...
                rev_f = rev_f.replace(ent.text, random.choice(names_b)) # ... and female name
                
        new_df.loc[index, 'Replaced Names'] = replaced_names_rev
        new_df.loc[index, 'text_na_M'] = rev_m
        new_df.loc[index, 'text_na_F'] = rev_f
        if include_name:
            masked_ids.append(index)

    #logging.info('one done')
    return [new_df, masked_ids, replaced_names_all]


def remove_names_TW(new_df):
    #new_df = review_df.copy()
    new_df['neutral text'] = ['']*len(new_df)    # Create empty column for male reviews
    new_df['Replaced Names'] = [0]*len(new_df) # Create empty column for number of replaced names per review
    replaced_names_all = 0    # counter for replaced names for the whole data set
    replaced_names_rev = 0    # counter for replaced names for each review
    
    nlp = spacy.load("en_core_web_sm")
    masked_ids = []
    
    i = 0
    for index, row in new_df.iterrows(): # each review in dataframe
        rev_ = row['text']
        doc = nlp(row['text'])
        include_name = False
        replaced_names_rev = 0
        
        for ent in doc.ents: # each entity in review
            #print(ent.label_)
            if ent.label_ == 'PERSON': # if that entity is a person
                replaced_names_all += 1 # increas counters
                replaced_names_rev += 1
                
                include_name = True 
                rev_ = rev_.replace(ent.text, '') # replace with random male ...
                
        new_df.loc[index, 'Replaced Names'] = replaced_names_rev
        new_df.loc[index, 'neutral text'] = rev_
        if include_name:
            masked_ids.append(index)
    #logging.info('one done')
    logging.info('masking names in frame: names have been masked in ' + str(replaced_names_all) + ' of ' + str(new_df.shape[0]) + ' samples.')
#    return new_df, masked_ids, replaced_names_all

'''