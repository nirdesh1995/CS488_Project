import json
from datetime import date, timedelta
from os.path import join, exists
import pandas as pd
from os import makedirs

def return_body(json_dict): 
    df_temp = pd.io.json.json_normalize(json_dict)
    return df_temp['bodyText'][0]

def return_word_count(json_dict):
    df_temp = pd.io.json.json_normalize(json_dict)
    return df_temp['wordcount'][0]





def compile_dataframe(start_date = date(2000,1, 1),end_date=date(2000,2, 1)):
    
    DATAFRAME_DIR = join('data' ,'dataframes')
    makedirs(DATAFRAME_DIR, exist_ok=True)
    
    file_name = join(DATAFRAME_DIR, start_date.strftime('%Y-%m-%d') +'::'+ end_date.strftime('%Y-%m-%d') + '.pkl')
    df_final = pd.DataFrame(columns=['fields','sectionId', 'sectionName','webTitle','bodyText','wordcount','publicationDate'])
    
    #all_dates = []    #list of all filenames 

    all_days = range((end_date - start_date).days + 1)
    for daycount in all_days:
        dt = start_date + timedelta(days=daycount)
        datestr = dt.strftime('%Y-%m-%d')
    

        date = datestr
        #print(datestr)
        file = 'data/articles/' + date + '.json'
        file_handler = open(file)
        text = file_handler.read()
        df = pd.DataFrame(columns=['fields','sectionId', 'sectionName','webTitle' ])  #incase the date is empty
        df = pd.read_json(text)
        #all_dates.append(datestr)

    
        df = df[['fields','sectionId', 'sectionName','webTitle']]
        df['bodyText'] = df['fields'].apply(return_body)
        df['wordcount'] = df['fields'].apply(return_word_count)
        df['publicationDate'] = date 
   
        print("articles processed for :",date)
        df_final = pd.concat([df_final, df], axis=0, ignore_index= True)
        
    print("Dataframe_pickeld for Date range::",start_date.strftime('%Y-%m-%d') +'::'+ end_date.strftime('%Y-%m-%d') )
    df_final.to_pickle(file_name) 
    
