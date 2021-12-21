from itertools import chain
import logging
logging.basicConfig(level=logging.INFO)

# Import predefined term lists
def imp_terms(path):
    a_file = open(path, "rb")
    terms = pickle.load(a_file)
    a_file.close()
    logging.info('load ' + path + ', total number: ' + str(len(terms)))
    return terms 

#terms_m2f = imp_terms("terms_m2f")
#terms_f2m = imp_terms("terms_f2m")
#all_terms = imp_terms("all_terms")

#prons_m2f = imp_terms("prons_m2f")
#prons_f2m = imp_terms("prons_f2m")

### Mask Names ### 
# Define name Sets
m_names_s = ['bill', 'greg', 'jeff', 'john', 'kevin', 'mike', 'paul', 'steve']
f_names_s = ['amy', 'ann', 'diana', 'donna', 'joan', 'kate', 'lisa', 'sarah']

wh_m_names = ['adam', 'alan', 'andrew', 'brad', 'brandon', 'chip', 'frank', 'fred', 'greg', 'hank', 'harry', 'ian', 'jack', 'jed', 'jonathan', 'josh', 'justin', 'matthew', 'paul', 'peter', 'roger', 'ryan', 'stephen', 'todd', 'wilbur']
wh_f_names = ['amanda', 'amber', 'betsy', 'colleen', 'courtney', 'crystal', 'donna', 'ellen', 'emily', 'heather', 'katie', 'kristin', 'lauren', 'megan', 'melanie', 'meredith', 'nancy', 'peggy', 'rachel', 'sara', 'shannon', 'stephanie', 'ellen', 'wendy']
bl_m_names = ['alonzo', 'alphonse', 'darnell', 'deion', 'everol', 'jamel', 'jerome', 'lamar', 'lamont', 'lavon', 'lerone', 'leroy', 'lionel', 'malik', 'marcellus', 'percell', 'rasaan', 'rashaun', 'terrence', 'terryl', 'theo', 'torrance', 'tyree', 'tyrone', 'wardell']
bl_f_names = ['aiesha', 'ebony', 'jasmine', 'lakisha', 'lashandra', 'lashelle', 'latisha', 'latonya', 'latoya', 'malika', 'nichelle', 'shaniqua', 'shanise', 'sharise', 'shavonn', 'shereen', 'tameisha', 'tanisha', 'tashika', 'tawanda', 'temeka', 'teretha', 'tia', 'yolanda', 'yvette']

m_names = m_names_s + wh_m_names + bl_m_names
f_names = f_names_s + wh_f_names + bl_f_names 


term_dict = {
    'masculine': 'feminine', 'masculinely': 'femininely', 'masculinity': 'femininity', 
    'masculism': 'feminism', 'masculinist': 'feminist', # 2
    'patriarch': 'matriarch', 'patriarchal': 'matriarchal', 
    'male': 'female', 'males': 'females', 'maleness': 'femaleness',
    'boy': 'girl', 'boys': 'girls', 'gamine': 'girly', 
    'boyhood': 'girlhood',
    'boyish': 'girlish', 'boyishly': 'girlishly', 'boyishness': 'girlishness',
    'man': 'woman', 'men': 'women', 'manhood': 'womanhood',
    'manliness': 'womanliness', 'manly': 'womanly', 
    'virile': 'uxorial', 'virile': 'uxorially', 
    'virility': 'milkiness', 
    'sir': 'madam', 'sirs': 'madams',
    'widower': 'widow', 'widowers': 'widows',
    'mr': 'mrs', 'mr': 'ms', 'mister': 'ms', # 4 
    'gentleman': 'lady', 'gentlemen': 'ladies', # 1
    'gentlemanlike': 'ladylike',
    'lord': 'lady', 'lords': 'ladies', # 1
    'groom': 'bride', 'grooms': 'brides', # 1
    'bridegroom': 'bride', 'bridegrooms': 'brides', # 1
    'best man': 'bridesmaid', 'best men': 'bridesmaids',
    'fiance': 'fiancee', 'fiancé': 'fiancée', 'fiances': 'fiancees', 'fiancés': 'fiancées', # is there a plural?
    
    # ambiguous  
    'housewifely': 'butch',
    'bloke': 'girl', 'blokes': 'girls', #typ 
    'dude': 'girlie', 'dudes': 'girlies',
    'fella': 'chick', 'fellas': 'chicks', # 1 2 4 - not really perfectly accurate
    'chap': 'lassie', 'chaps': 'lassies', #bursche
    'guy': 'gal', 'guys': 'gals', # 1 2 4
    'lad': 'lass', 'lads': 'lasses', 
    'macho': 'diva', 'machos': 'divas',
    
    # roles and occupations
    'actor': 'actress', 'actors': 'actresses',
    'waiter': 'waitress', 'waiters': 'waitresses',
    'king': 'queen', 'kings': 'queens',
    'prince': 'princess', 'princes': 'princesses',
    'monk': 'nun', 'monks': 'nuns', 'brethren': 'nuns',
    'monastery': 'convent', 'friary': 'nunnery',
    'count': 'countess', 'counts': 'countesses',
    'wizard': 'witch', 'wizards': 'witches',
    'priest': 'priestess', 'priests': 'priestesses',
    'prophet': 'prophetess', 'prophets': 'prophetesses',
    'patron': 'patroness', 'patrons': 'patronesses', 
    'host': 'hostess', 'hosts': 'hostesses', 
    'viscount': 'viscountess', 'viscounts': 'viscountesses', 
    'shepherd': 'shepherdess', 'shepherds': 'shepherdesses', 
    'steward': 'stewardess', 'stewards': 'stewardesses',
    'heir': 'heiress', 'heirs': 'heiresses', 
    'baron': 'baroness', 'barons': 'baronesses', 
    'abbot': 'abbess', 'abbots': 'abbesses', 
    'emperor': 'empress', 'emperors': 'empresses', 
    'traitor': 'traitress', 'traitors': 'traitresses', 
    'duke': 'duchess', 'dukes': 'duchesses', 
    'enchanter': 'enchantress', 'enchanters': 'enchantresses', 
    'songster': 'songstress', 'songsters': 'songstresses', 
    'hero': 'heroine', 'heroes': 'heroines', 
    'sultan': 'sultana', 'sultans': 'sultanas', 
    'czar': 'czarina', 'czars': 'czarinas', 
    'signor': 'signora', 'signors': 'signoras', 
    'benefactor': 'benefactress', 'benefactors': 'benefactresses', 
    'hunter': 'huntress', 'hunter': 'huntress', 
    'tempter': 'temptress', 'tempters': 'temptresses', 
    'master': 'mistress', 'masters': 'mistresses', 
    'manservant': 'maidservant', 'manservants': 'maidservants', 
    'landlord': 'landlady', 'landlords': 'landladies', 
    'countryman': 'countrywoman', 'countrymen': 'countrywomen',
    'milkman': 'milkmaid', 'milkmen': 'milkmaids', 
    'giant': 'giantess', 'giants': 'giantesses', 
    'mayor': 'mayoress', 'mayors': 'mayoresses', 
    'conductor': 'conductress', 'conductor': 'conductresses', 
    'god': 'goddess', 'gods': 'goddesses',
    'merman': 'mermaid', 'mermen': 'mermaids',
    'oarsman': 'oarswoman', 'oarsmen': 'oarswomen',
    'manservant': 'maid', 'manservants': 'maids', # 1 2
    'bellboy': 'chambermaid', 'bellboys': 'chambermaids', # 1 2
    'dairyman': 'dairymaid', 'dairymen': 'dairymaids',
    'schoolmaster': 'schoolmistress', 'schoolmasters': 'schoolmistresses',
    'headmaster': 'headmistress', 'headmaster': 'headmistress',
    'proprietor': 'proprietress', 'proprietors': 'proprietresses', 
    'ambassador': 'ambassadress', 'ambassadors': 'ambassadresses',
    'adventurer': 'adventuress', 'adventurers': 'adventuresses', 
    'protector': 'protectress', 'protectors': 'protectresses',
    'seducer': 'seductress', 'seducers': 'seductresses',
    'sculptor': 'sculptress', 'sculptors': 'sculptresses', 
    'congressman': 'congresswoman', 'congressmen': 'congresswomen', 
    'sorcerer': 'sorceress', 'sorcerers': 'sorceresses', 
    'launderer': 'laundress', 'launderers': 'laundresses', 
    'launderer': 'washerwoman', 'launderers': 'washerwomen', 
    'launderer': 'washwoman', 'launderers': 'washwomen', 
    'anchorite': 'anchoress', 'anchorites': 'anchoresses', 
    'procurer': 'procuress', 'procurers': 'procuresses', 
    'elector': 'electress', 'electors': 'electresses', 
    'adulterer': 'adulteress', 'adulterers': 'adulteresses', 
    'doorman': 'portress', 'doormen': 'portresses',  # porter is usually uses for men and women. Doorman however is male specific. So this is a good mapping. 
    # ending on 'man'/'woman' or 'boy'/'girl'
    'policeman': 'policewoman', 'policemen': 'policewomen', 
    'fireman': 'firewoman', 'firemen': 'firewomen',
    'businessman': 'businesswoman', 'businessmen': 'businesswomen',
    'barman': 'barmaid', 'barmen': 'barmaids',
    'mailman': 'mailwoman', 'mailmen': 'mailwomen',
    'postman': 'postwoman', 'postmen': 'postwoman',
    'salesman': 'saleswoman', 'salesmen': 'saleswomen', 
    'cowboy': 'cowgirl', 'cowboys': 'cowgirls',
    'schoolboy': 'schoolgirl', 'schoolboys': 'schoolgirls',
    'chairman': 'chairwoman', 'chairmen': 'chairwomen', 
    'englishman': 'englishwoman', 'englishmen': 'englishwomen', 
    'spokesman': 'spokeswoman', 'spokesmen': 'spokeswomen',
    'councilman': 'councilwoman', 'councilmen': 'councilwomen',
    'choirboy': 'choirgirl', 'choirboys': 'choirgirls', # 2 - there is not really such thing as a choirgirl officailly
    'playboy': 'playgirl', 'playboys': 'playgirls', 
    'homeboy': 'homegirl', 'homeboys': 'homegirls', 
    'madman': 'madwoman', 'madmen': 'madwomen', 
    'anchorman': 'anchorwoman', 'anchormen': 'anchorwomen',
    'sportsman': 'sportswoman', 'sportsmen': 'sportswomen',
    'cameraman': 'camerawoman', 'cameramen': 'camerawomen', 
    'fisherman': 'fisherwoman', 'fishermen': 'fisherwomen', 
    'serviceman': 'servicewoman', 'servicemen': 'servicewomen',  
    'churchman': 'churchwoman', 'churchmen': 'churchwomen', 
    'clergyman': 'clergywoman', 'clergymen': 'clergywomen', 
    'craftsman': 'craftswoman', 'craftsmen': 'craftswomen', 
    'superman': 'superwoman', 'supermen': 'superwomen', 
    'nobleman': 'noblewoman', 'noblemen': 'noblewomen', 
    'horseman': 'horsewoman', 'horsemen': 'horsewomen', 
    'stuntman': 'stuntwoman', 'stuntmen': 'stuntwomen', 
    'townsman': 'townswoman', 'townsmen': 'townswomen', 
    'statesman': 'stateswoman', 'statesmen': 'stateswomen', 
    'foreman': 'forewoman', 'foremen': 'forewomen', 
    'kinsman': 'kinswoman', 'kinsmen': 'kinswomen', 
    'bondman': 'bondwoman', 'bondmen': 'bondwomen', 
    'bondsman': 'bondswoman', 'bondsmen': 'bondswomen', 
    'handyman': 'handywoman', 'handymen': 'handywomen', 
    'henchman': 'handmaid', 'henchmen': 'handmaids', 
    'strongman': 'strongwoman', 'strongmen': 'strongwomen', # Strongwoman does not really exist.  
    'charman': 'charwoman', 'charmen': 'charwomen', 
    'tailor': 'seamstress', 'tailors': 'seamstresses',
    'tailor': 'sempstress', 'tailors': 'sempstresses',  
    'tailor': 'needlewoman', 'tailors': 'needlewomen',      
    'newsman': 'newswoman', 'newsmen': 'newswomen',
    
    # family - relatives - etc.  
    'father': 'mother', 'fathers': 'mothers', 
    'brother': 'sister', 'brothers': 'sisters',
    'husband': 'wife', 'husbands': 'wives',
    'housewife': 'househusband', 'housewives': 'househusbands', 
    'hubby': 'wifey', 'hubbies': 'wifeys',
    'manly': 'wifely', 
    'son': 'daughter', 'sons': 'daughters',
    'dad': 'mom', 'dads': 'moms',
    'papa': 'mama', 'papas': 'mamas',
    'daddy': 'mommy', 'daddies': 'mommies',
    'pa': 'ma',
    'paternity': 'maternity', 'paternally': 'maternally', 'paternal': 'maternal',
    'fatherhood': 'motherhood', 'brotherhood': 'sisterhood',
    'fraternity': 'sorority', 'fraternities': 'sororities', 'fraternal': 'sisterly', 'fraternally': 'sisterly',
    'brotherly': 'sisterly', 'fatherly': 'motherly', 'grandfatherly': 'grandmotherly',
    'fathered': 'mothered',
    
    'stepfather': 'stepmother', 'stepfathers': 'stepmothers',
    'stepbrother': 'stepsister', 'stepbrothers': 'stepsisters',
    'stepson': 'stepdaughter', 'stepsons': 'stepdaughters',
    'grandfather': 'grandmother', 'grandfathers': 'grandmothers',
    'grandpa': 'grandma', 'grandpas': 'grandmas', 
    'grandson': 'granddaughter', 'grandsons': 'granddaughters',
    'nephew': 'niece', 'nephews': 'nieces',
    'brother-in-law': 'sister-in-law', 'brothers-in-law': 'sisters-in-law',
    'father-in-law': 'mother-in-law', 'fathers-in-law': 'mothers-in-law',
    'son-in-law': 'daughter-in-law', 'sons-in-law': 'daughters-in-law', 
    'uncle': 'aunt', 'uncles': 'aunts',
    'boyfriend': 'girlfriend', 'boyfriends': 'girlfriends', 
    'ancestor': 'ancestress', 'ancestors': 'ancestresses', 
    'bachelor': 'bachelorette', 'bachelors': 'bachelorettes',
    
    # physiology 
    'testosterone': 'estrogen', 'testosterones': 'estrogens', 
    'penis': 'vagina', 'penises': 'vaginas', 
    'prostate': 'ovarian', # as in prostate cancer, ovarian cancer (and also others)
    'testicles': 'uterus', 'testicle': 'uterus', 'testicles': 'uteruses',
    'sperm': 'ovary',     
}

pron_dict = {
     # pronouns 
    'he':      'she',
    'him':     'her',
    'his':     'hers',
    'his':     'her',
    'himself': 'herself'
}


only_f2m = {
    'gay': 'lesbian', 'gays': 'lesbians', 'homosexuality': 'lesbianism',
    'gentleman': 'damsel', 'gentlemen': 'damsels',
    'bachelor': 'maiden', 'bachelors': 'maidens', 
    'victor': 'victress', 'victors': 'victresses',# 4 3
    'doorman': 'doorwoman', 'doormen': 'doorwomen', # I think such thing as "doorwoman" does not exist...'
    'murderer': 'murderess', 'murderers': 'murderesses', ### 3 - Murder can also be female? is murderess even used? 
    'millionaire': 'millionairess', 'millionaires': 'millionairesses', # 3
    'manager': 'manageress', 'managers': 'manageresses', # 3
    'victor': 'victress', 'victors': 'victresses',  # 4 3
    'governor': 'governoress', 'governors': 'governoresses',
    'director': 'directress', 'directors': 'directresses', # 3
    'inventor': 'inventress', 'inventors': 'inventresses',# 3
    'instructor': 'instructress', 'instructors': 'instructresses', 
    'childminder': 'nursemaid', 'childminders': 'nursemaids',

}

only_m2f = {
    'barbershop': 'hairdresser',
    'beard': 'hair', 'beards': 'hair',
    'juryman': 'juror', 'jurymen': 'jurors', 
    'newsboy': 'newspaper carrier', 'newsboys': 'newspaper carriers',
    'paperboy': 'newspaper carrier', 'paperboys': 'newspaper carriers', 
    'newspaperman': 'newspaperwoman', 'newspapermen': 'newspaperwomen', 
}

# I could not come up with suitable equivalents from the other gender
only_cover = [ 'climacteric', 'menopause', 'uterus', 'womb'] #im endeffekt nicht verwendet


# ----------------------------------------------------------------
# ----------------------------------------------------------------
weat_dict = {
    #--- (Nosek et al., 2002b) ---
    "masculine" : "feminine",
    "brother" :  "sister",
    "father" : "mother",
    "grandfather" : "grandmother",
    "brothers" :  "sisters",
    "fathers" : "mothers",
    "grandfathers" : "grandmothers",
    "he" : "she",
    "him" : "her",
    "his" : "hers",
    "son" : "daughter",
    "uncle" : "aunt",
    #--- (Nosek et al., 2002a) ---
    "boy" : "girl",
    "boys" : "girls",
    #"brother" : "sister",
    #"he" : "she",
    #"him" : "her",
    #"his" : "hers",
    "male" : "female",
    "man" : "woman",
    "men" : "women",
    #"son" : "daughter"
}



weat_m2f = weat_dict
weat_f2m = dict((v,k) for k,v in weat_dict.items())
all_weat = set(list(weat_m2f.keys())+list(weat_f2m.keys()))
    
# ----------------------------------------------------------------
# ----------------------------------------------------------------
term_dict = dict(chain(term_dict.items(), pron_dict.items()))

prons_m2f = pron_dict
prons_f2m = dict((v,k) for k,v in pron_dict.items())
all_prons = set(list(prons_m2f.keys())+list(prons_f2m.keys()))

terms_m2f = dict(chain(term_dict.items(), only_m2f.items()))
terms_f2m = dict(chain(term_dict.items(), only_f2m.items()))
terms_f2m = dict((v,k) for k,v in terms_f2m.items())

all_terms = set(list(terms_m2f.keys())+list(terms_f2m.keys())+only_cover)

logging.info('imported term dicts: prons_m2f, prons_f2m, terms_m2f, terms_f2m; and sets; all_prons and all_terms')