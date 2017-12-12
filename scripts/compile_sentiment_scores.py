

import pandas as pd
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import sent_tokenize, word_tokenize 
from textblob import TextBlob
from nltk.tokenize import RegexpTokenizer
import re


from os.path import join, exists
import pandas as pd
from os import makedirs
from datetime import date



def return_polarity(article): 
    print("polarity returning")
    return TextBlob(article).sentiment.polarity
    
   
def compile_sentiment_scores(start_date = date(2000,1, 1),end_date=date(2000,2, 1)):
    
    SENTISCORES_DIR = join('data' ,'sentiment_data')
    makedirs(SENTISCORES_DIR, exist_ok=True)
    
    read_file = join('data/dataframes', start_date.strftime('%Y-%m-%d') +'::'+ end_date.strftime('%Y-%m-%d') + '.pkl')    
    file_name = join(SENTISCORES_DIR, start_date.strftime('%Y-%m-%d') +'::'+ end_date.strftime('%Y-%m-%d') + '.pkl')
    
    df = pd.read_pickle(read_file)
    print("filefound and read")
    df['polarity'] = df['bodyText'].apply(return_polarity)
    print('done applying')

    df_polarities = df[['publicationDate','webTitle','sectionId','wordcount','polarity']]

    df_polarities.to_pickle(file_name)
    print('saved to pickle')