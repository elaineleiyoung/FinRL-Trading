import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
from config import config

def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    #_data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    _data = pd.read_csv(file_name)
    return _data

def data_split(df,start,end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.datadate >= start) & (df.datadate < end)]
    data=data.sort_values(['datadate','tic'],ignore_index=True)
    #data  = data[final_columns]
    data.index = data.datadate.factorize()[0]
    return data

def calcualte_price(df):
    """
    calcualte adjusted close price, open-high-low price and volume
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    data = df.copy()
    data = data[['datadate', 'tic', 'prccd', 'ajexdi', 'prcod', 'prchd', 'prcld', 'cshtrd']]
    data['ajexdi'] = data['ajexdi'].apply(lambda x: 1 if x == 0 else x)

    data['adjcp'] = data['prccd'] / data['ajexdi']
    data['open'] = data['prcod'] / data['ajexdi']
    data['high'] = data['prchd'] / data['ajexdi']
    data['low'] = data['prcld'] / data['ajexdi']
    data['volume'] = data['cshtrd']

    data = data[['datadate', 'tic', 'adjcp', 'open', 'high', 'low', 'volume']]
    data = data.sort_values(['tic', 'datadate'], ignore_index=True)
    return data

def add_technical_indicator(df):
    """
    Calculate technical indicators using the stockstats package.
    :param df: (pd.DataFrame) Input dataframe.
    :return: (pd.DataFrame) Dataframe with added technical indicators.
    """
    stock = Sdf.retype(df.copy())
    stock['close'] = stock['adjcp']
    unique_ticker = stock.tic.unique()

    macd_list, rsi_list, cci_list, dx_list = [], [], [], []

    for ticker in unique_ticker:
        ## macd
        temp_macd = stock[stock.tic == ticker][['macd']].reset_index(drop=True)
        macd_list.append(temp_macd)
        ## rsi
        temp_rsi = stock[stock.tic == ticker][['rsi_30']].reset_index(drop=True)
        rsi_list.append(temp_rsi)
        ## cci
        temp_cci = stock[stock.tic == ticker][['cci_30']].reset_index(drop=True)
        cci_list.append(temp_cci)
        ## adx
        temp_dx = stock[stock.tic == ticker][['dx_30']].reset_index(drop=True)
        dx_list.append(temp_dx)

    df['macd'] = pd.concat(macd_list, ignore_index=True)
    df['rsi'] = pd.concat(rsi_list, ignore_index=True)
    df['cci'] = pd.concat(cci_list, ignore_index=True)
    df['adx'] = pd.concat(dx_list, ignore_index=True)

    return df

def load_vix_data(file_name: str) -> pd.DataFrame:
    """
    Load historical VIX data and format it correctly.
    """
    vix_data = pd.read_csv(file_name)
    # depends on VIX data format
    vix_data = vix_data.rename(columns={'observation_date': 'datadate', 'VIXCLS': 'VIX'})
    vix_data['datadate'] = pd.to_datetime(vix_data['datadate']).dt.strftime('%Y%m%d').astype(int)
    ###
    vix_data = vix_data[['datadate', 'VIX']]
    return vix_data

def add_vix_data(df, vix_data):
    """
    Merge VIX data with stock dataset.
    """
    df = df.merge(vix_data, on='datadate', how='left')
    df['VIX'] = df['VIX'].ffill()  # Forward-fill missing VIX values
    return df

def preprocess_data():
    """data preprocessing pipeline"""

    df = load_dataset(file_name=config.TRAINING_DATA_FILE)
    # get data after 2009
    df = df[df.datadate>=20090000]
    # calcualte adjusted price
    df_preprocess = calcualte_price(df)
    # add technical indicators using stockstats
    df_preprocess = add_technical_indicator(df_preprocess)
    vix_data = load_vix_data("data/VIXCLS.csv")
    df_preprocess = add_vix_data(df_preprocess, vix_data)
    # fill the missing values at the beginning
    df_final = df_preprocess.bfill()
    return df_final

def add_turbulence(df):
    """
    add turbulence index from a precalcualted dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    turbulence_index = calcualte_turbulence(df)
    df = df.merge(turbulence_index, on='datadate')
    df = df.sort_values(['datadate','tic']).reset_index(drop=True)
    return df



def calcualte_turbulence(df):
    """calculate turbulence index based on dow 30"""
    # can add other market assets
    
    df_price_pivot=df.pivot(index='datadate', columns='tic', values='adjcp')
    unique_date = df.datadate.unique()
    # start after a year
    start = 252
    turbulence_index = [0]*start
    #turbulence_index = [0]
    count=0
    for i in range(start,len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index ]]
        cov_temp = hist_price.cov()
        current_temp=(current_price - np.mean(hist_price,axis=0))
        temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
        if temp>0:
            count+=1
            if count>2:
                turbulence_temp = temp[0][0]
            else:
                #avoid large outlier because of the calculation just begins
                turbulence_temp=0
        else:
            turbulence_temp=0
        turbulence_index.append(turbulence_temp)
    
    
    turbulence_index = pd.DataFrame({'datadate':df_price_pivot.index,
                                     'turbulence':turbulence_index})
    return turbulence_index










