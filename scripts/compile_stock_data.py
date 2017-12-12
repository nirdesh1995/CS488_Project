


from os.path import join, exists
import pandas as pd
from os import makedirs
from datetime import date
import pandas as pd
import pandas_datareader.data as web

def compile_stock_indices(start_date = date(2000,1, 1),end_date=date(2000,2, 1)):
    STOCKS_DIR = join('data' ,'stock_data')
    makedirs(STOCKS_DIR, exist_ok=True)

    file_name = join(STOCKS_DIR, start_date.strftime('%Y-%m-%d') +'::'+ end_date.strftime('%Y-%m-%d') + '.pkl')

    
    df_sp500 = web.get_data_yahoo('%5EGSPC', start_date, end_date)   #sp500 
    df_dow = web.get_data_yahoo('%5EDJI', start_date, end_date) 
    df_welshire = web.get_data_yahoo('WIBCX', start_date, end_date) 
    df_nasdaq = web.get_data_yahoo('%5EIXIC', start_date, end_date) 
    df_russel =   web.get_data_yahoo('%5ERUT', start_date, end_date)


    df_sp500 = df_sp500[['Close']]
    df_sp500.rename(columns={'Close': 'sp_500'}, inplace=True)

    df_nasdaq = df_nasdaq[['Close']]
    df_nasdaq.rename(columns={'Close': 'nasdaq'}, inplace=True)

    df_dow = df_dow[['Close']]
    df_dow.rename(columns={'Close': 'dow_jones'}, inplace=True)

    df_welshire = df_welshire[['Close']]
    df_welshire.rename(columns={'Close': 'welshire'}, inplace=True)


    df_russel = df_russel[['Close']]
    df_russel.rename(columns={'Close': 'russel'}, inplace=True)

    df_stock_prices = pd.concat([df_sp500, df_nasdaq,df_dow,df_welshire,df_russel], axis=1)
    
    df_stock_prices.to_pickle(file_name)
    print("Stock indices pickeld for date range:",start_date.strftime('%Y-%m-%d') +'::'+ end_date.strftime('%Y-%m-%d'))
    