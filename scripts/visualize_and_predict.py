

import pandas as pd
import matplotlib
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import style
#plt.legend(loc='best')

from os.path import join, exists
import pandas as pd
from os import makedirs
from datetime import date
from matplotlib import pyplot

from sklearn.neural_network import MLPClassifier
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import treeinterpreter as ti
from treeinterpreter import treeinterpreter as ti

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix
    

def return_combined(start_date = date(2000,1, 1),end_date=date(2000,2, 1)):
    stock_file = join('data/stock_data', start_date.strftime('%Y-%m-%d') +'::'+ end_date.strftime('%Y-%m-%d') + '.pkl')
    polarities_file =  join('data/sentiment_data', start_date.strftime('%Y-%m-%d') +'::'+ end_date.strftime('%Y-%m-%d') + '.pkl')
    df_stock = pd.read_pickle(stock_file) 
    df_polarities = pd.read_pickle(polarities_file) 


    date_range = pd.date_range(start_date, end_date)

    df_stock = df_stock.reindex(date_range, fill_value=np.NaN)
    df_stock = df_stock.interpolate()
    df_polarities.index = pd.DatetimeIndex(df_polarities.publicationDate)
    df_polarities = df_polarities.groupby(by = [df_polarities.index,'sectionId']).mean()

    df_polarities = df_polarities.unstack()


    df_polarities = df_polarities.xs('polarity', axis=1, drop_level=True)
    df_polarities.reset_index(inplace=True)
    df_polarities.set_index('publicationDate', inplace=True)
    df_polarities= df_polarities[['world','business','politics','technology','money','media']]
    
    df_combined = df_stock.join(df_polarities)
    df_combined = df_combined.fillna(method ='bfill')
    df_combined = df_combined.fillna(method ='ffill')
    return df_combined 


def visualize_polarity(start_date = date(2000,1, 1),end_date=date(2000,2, 1)): 

    #DATAFRAME_DIR = join('data' ,'dataframes')
    #makedirs(DATAFRAME_DIR, exist_ok=True)
    
    stock_file = join('data/stock_data', start_date.strftime('%Y-%m-%d') +'::'+ end_date.strftime('%Y-%m-%d') + '.pkl')
    polarities_file =  join('data/sentiment_data', start_date.strftime('%Y-%m-%d') +'::'+ end_date.strftime('%Y-%m-%d') + '.pkl')
    
    
    df_stock = pd.read_pickle(stock_file) 
    df_polarities = pd.read_pickle(polarities_file) 
   
    
    df_polarities.index = pd.DatetimeIndex(df_polarities.publicationDate)
    df_polarities = df_polarities.groupby(by = [df_polarities.index,'sectionId']).mean()

    df_polarities = df_polarities.unstack()


    df_polarities = df_polarities.xs('polarity', axis=1, drop_level=True)
    df_polarities.reset_index(inplace=True)
    df_polarities.set_index('publicationDate', inplace=True)
    df_polarities= df_polarities[['world','business','politics','technology','money','media']]

       
    values = df_polarities.values
    groups = [0,1,2,3,4,5]
    n = 1 

    pyplot.figure(figsize=(20,10))
    for group in groups:
        pyplot.subplot(len(groups), 1, n)
        pyplot.plot(values[:, group])
        pyplot.title(df_polarities.columns[group], y=0.5, loc='right')
        n += 1
    pyplot.show()
    return df_polarities 


    #df_combined = df.join(df_sentiment_scores)
    #df_combined = df_combined.fillna(method='ffill')  #ffill: propagate last valid observation forward to next valid " 
    #df_combined = df_combined.fillna(method ='bfill')  # one missing for 'money section 
    
def visualize_stocks(start_date = date(2000,1, 1),end_date=date(2000,2, 1)): 
    
    stock_file = join('data/stock_data', start_date.strftime('%Y-%m-%d') +'::'+ end_date.strftime('%Y-%m-%d') + '.pkl')

    df_stocks = pd.read_pickle(stock_file) 
    date_range = pd.date_range(start_date, end_date)
    df_stocks = df_stocks.reindex(date_range, fill_value=np.NaN)
    df_stocks= df_stocks.interpolate()

    values = df_stocks.values
    groups = [0,1,2,3,4]
    #values = values.astype('float32')
    n = 1 
    pyplot.figure(figsize=(20,10))
    for group in groups:
        pyplot.subplot(len(groups), 1, n)
        pyplot.plot(values[:, group])
        pyplot.title(df_stocks.columns[group], y=0.5, loc='right')
        n += 1
    pyplot.show()
    return df_stocks

def visualize_correlation(start_date = date(2000,1, 1),end_date=date(2000,2, 1)):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    df_combined = return_combined(start,end)

    sns.set(style="dark")

    df = df_combined.pct_change()
    #df= df_combined
    corr = df.corr()
    
    np_mask = np.zeros_like(corr, dtype=np.bool)
    np_mask[np.triu_indices_from(np_mask)] = True
    
    f, ax = plt.subplots(figsize=(10, 9))
    cmap = sns.diverging_palette(300, 10, as_cmap=True)

    sns.heatmap(corr, mask=np_mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    
def visualize_prediction(start_date = date(2000,1, 1),end_date=date(2000,2, 1)): 
    




    df_combined = return_combined(start_date,end_date)
    df_combined.business = df_combined.business.pct_change()
    df_combined.money = df_combined.money.pct_change()
    df_combined.world = df_combined.world.pct_change()
    #df_combined.drop(df_combined.head(1).index, inplace=True)
    #df_combined.drop(df_combined.tail(1).index, inplace=True)
    df_combined.replace(np.inf, 0, inplace=True)
    df_combined.replace(np.nan, 0, inplace=True)
    df_combined.replace(-np.inf, 0, inplace=True)
    
    df_combined = df_combined.fillna(method ='bfill')
    df_combined = df_combined.fillna(method ='ffill')
    #df_combined.drop(df_combined.head(1).index, inplace=True)
    #df_combined.drop(df_combined.tail(1).index, inplace=True)
    #df_combined.dropna(inplace=True)
    df_combined['sp_500'] = df_combined['sp_500'].apply(np.int64)
    #df_combined = df_combined.pct_change()
    train_start = '2010-01-02'
    train_end = '2010-08-01'
    test_start = '2010-08-05'
    test_end = '2010-12-24'
    #df_combined['sp_500'] = df_combined['sp_500'].apply(np.int64)
    train = df_combined.ix[train_start : train_end]
    test = df_combined.ix[test_start:test_end]



    prediction_list = []

    sentiment_score_list = []
    for date, row in train.T.iteritems():
        sentiment_score = np.asarray([df_combined.loc[date, 'business'],df_combined.loc[date, 'money'],df_combined.loc[date, 'world']])
        #sentiment_score = np.asarray([df_combined.loc[date, 'money']])
        sentiment_score_list.append(sentiment_score)
        numpy_df_train = np.asarray(sentiment_score_list)
        
    
    sentiment_score_list = []
    for date, row in test.T.iteritems():
        sentiment_score = np.asarray([df_combined.loc[date, 'business'],df_combined.loc[date, 'money'],df_combined.loc[date, 'world']])
        #sentiment_score = np.asarray([df_combined.loc[date, 'money']])
        sentiment_score_list.append(sentiment_score)
        numpy_df_test = np.asarray(sentiment_score_list)
        
    y_train = pd.DataFrame(train['sp_500'])
    y_test = pd.DataFrame(test['sp_500'])
    
    rf = RandomForestRegressor()
    rf.fit(numpy_df_train, y_train)


    #print (rf.feature_importances_)
    prediction, bias, contributions = ti.predict(rf, numpy_df_test)

    idx = pd.date_range(test_start, test_end)
    predictions_df = pd.DataFrame(data=prediction[0:], index = idx, columns=['sp500_predicted'])
    #predictions_df

    predictions_plot = predictions_df.plot()

    fig = y_test.plot(ax = predictions_plot).get_figure()
    
    
    
