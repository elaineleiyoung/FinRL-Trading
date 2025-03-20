import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
from config import config
from scipy.optimize import minimize

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
    data = df[(df['datadate'] >= start) & (df['datadate'] < end)]
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
    calcualte technical indicators
    use stockstats package to add technical inidactors
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    stock = Sdf.retype(df.copy())

    stock['close'] = stock['adjcp']
    unique_ticker = stock.tic.unique()

    macd = pd.DataFrame()
    rsi = pd.DataFrame()
    cci = pd.DataFrame()
    dx = pd.DataFrame()

    #temp = stock[stock.tic == unique_ticker[0]]['macd']
    for i in range(len(unique_ticker)):
        ## macd
        temp_macd = stock[stock.tic == unique_ticker[i]]['macd']
        temp_macd = pd.DataFrame(temp_macd)
        macd = pd.concat([macd,temp_macd])
        ## rsi
        temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
        temp_rsi = pd.DataFrame(temp_rsi)
        rsi = pd.concat([rsi, temp_rsi])
        ## cci
        temp_cci = stock[stock.tic == unique_ticker[i]]['cci_30']
        temp_cci = pd.DataFrame(temp_cci)
        cci = pd.concat([cci, temp_cci])
        ## adx
        temp_dx = stock[stock.tic == unique_ticker[i]]['dx_30']
        temp_dx = pd.DataFrame(temp_dx)
        dx = pd.concat([dx, temp_dx])


    df['macd'] = macd
    df['rsi'] = rsi
    df['cci'] = cci
    df['adx'] = dx

    return df

def load_vix_data(file_name: str) -> pd.DataFrame:
    """
    Load historical VIX data and format it correctly.
    """
    vix_data = pd.read_csv(file_name)
    # depends on VIX data format
    vix_data = vix_data.rename(columns={'observation_date': 'datadate', 'VIXCLS': 'VIX'})
    # vix_data['datadate'] = pd.to_datetime(vix_data['datadate']).dt.strftime('%Y%m%d').astype(int)
    vix_data['datadate'] = pd.to_datetime(vix_data['datadate'])
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

def get_price_data(start_date):
    price_df = pd.read_csv('data_processor_update/sp500_price_199601_202502.csv')
    price_df.rename(columns={'date':'datadate','adj_close_q':'adjcp','openprc':'open',
                                'askhi':'high','bidlo':'low','vol':'volume'}, inplace=True)
    price_df = price_df[['datadate', 'tic', 'adjcp', 'open', 'high', 'low', 'volume']]
    fundamental_df = pd.read_csv('data_processor_update/sp500_fundamental_199601_202502.csv')
    fundamental_df.drop_duplicates(subset=['gvkey'], inplace=True)
    fundamental_df = fundamental_df[['gvkey','tic']]
    price_df = pd.merge(price_df, fundamental_df, on = 'tic')
    price_df.drop(columns='tic', inplace=True)
    price_df.rename(columns={'gvkey':'tic'}, inplace=True)
    price_df['datadate'] = pd.to_datetime(price_df['datadate'].str[:10], format="%Y-%m-%d")
    price_df = price_df[price_df['datadate']>=start_date]
    return price_df

def get_selected_stock(date, if_single = False):
    df = pd.read_csv('stock_selected.csv')
    df.rename(columns={'gvkey':'tic','trade_date':'datadate'}, inplace=True)
    df = df[['tic','datadate']]
    df = df.sort_values(['datadate','tic'], ignore_index=True)
    df['datadate'] = pd.to_datetime(df['datadate'], format="%Y-%m-%d")
    if if_single:
        df = df[df['datadate'] == date]
    else:
        df = df[df['datadate'] >= date]
    return df

def preprocess_selected_stock(df, start_date):
    report_date = (list(df.datadate.unique()))
    report_date.sort()
    price_df = get_price_data(start_date)
    trading_df = pd.DataFrame()
    for i in range(len(report_date)-1):
        temp_df = price_df[
            (price_df['datadate'] < report_date[i+1]) &
            (price_df['tic'].isin(df[df['datadate'] == report_date[i]]['tic']))
        ]
        trading_df = pd.concat([trading_df, temp_df])
    trading_df.drop_duplicates(subset=['datadate', 'tic'], inplace = True)
    trading_df = trading_df.sort_values(['datadate','tic'], ignore_index=True)
    return trading_df

def preprocess_data(if_vix = False, selected_stocks = False, stock_selected_df = None, datadate = False, start_date = 20010101):
    """data preprocessing pipeline"""
    if selected_stocks:
        df_preprocess = preprocess_selected_stock(df = stock_selected_df, start_date = start_date)
    else:
        df = load_dataset(file_name=config.TRAINING_DATA_FILE)
        # get data after 2009
        df = df[df.datadate>=20090000]
        # calcualte adjusted price
        df_preprocess = calcualte_price(df)
        # add technical indicators using stockstats
    df_preprocess=add_technical_indicator(df_preprocess) # add ROE etc
    # df_preprocess.to_csv('pre_with_ti.csv')
    # df_preprocess = pd.read_csv('pre_with_ti.csv', index_col=0)
    # df_preprocess['datadate'] = pd.to_datetime(df_preprocess['datadate'], format="%Y-%m-%d")
    if if_vix:
        vix_data = load_vix_data("DRL-for-Trading\data\VIXCLS.csv")
        df_preprocess = add_vix_data(df_preprocess, vix_data)
    else:
        df_preprocess = add_turbulence(df_preprocess)
    # fill the missing values at the beginning
    df_preprocess.fillna(method='bfill',inplace=True)
    return df_preprocess

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

def get_training_data(df, stock_selected_df, start_date, val_start_date, trade_start_date, trade_end_date):
    stock_selected_df = get_selected_stock(trade_start_date, if_single=True)
    stock_list = set(stock_selected_df['tic'])

    df = df[(df['datadate'] < trade_end_date) & (df['tic'].isin(stock_list))]
    df = df.sort_values(['datadate', 'tic'], ignore_index=True)

    # all_dates = df['datadate'].unique()
    # all_tics = list(stock_list)

    # full_index = pd.MultiIndex.from_product([all_dates, all_tics], names=['datadate', 'tic'])
    # df = df.set_index(['datadate', 'tic']).reindex(full_index).reset_index()

    # df.fillna(0, inplace=True)
    total_days = df['datadate'].nunique() 
    tic_counts = df.groupby('tic')['datadate'].nunique()  
    valid_tics = tic_counts[tic_counts == total_days].index 
    df = df[df['tic'].isin(valid_tics)]
    df = df.sort_values(['datadate', 'tic'], ignore_index=True)  

    train_set = data_split(df, start=start_date, end=val_start_date)
    val_set = data_split(df, start=val_start_date, end=trade_start_date)
    trade_set = data_split(df, start=trade_start_date, end=trade_end_date)

    return len(valid_tics), df.tic.unique().tolist(), train_set, val_set, trade_set

def calculate_mean_variance(start_date, trade_start_date, stock_pool, initial = 1000000, risk_aversion=1.0):
    price_df = get_price_data(start_date)
    price_df = price_df[price_df['datadate'] < trade_start_date]
    # df = get_selected_stock(trade_start_date, if_single=True)
    tickers = stock_pool
    price_df = price_df[price_df['tic'].isin(tickers)]
    
    price_pivot = price_df.pivot(index='datadate', columns='tic', values='adjcp')
    returns = price_pivot.pct_change().dropna()

    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    num_assets = len(tickers)

    def objective(weights):
        return - (weights.T @ mean_returns) + risk_aversion * (weights.T @ cov_matrix @ weights)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.ones(num_assets) / num_assets
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_weights = result.x

    final_allocations = optimal_weights * initial
    last_prices = price_pivot.iloc[-1].values
    num_shares = (final_allocations / last_prices) // 100 * 100 
    remaining_cash = initial - np.sum(num_shares * last_prices)
    df_result = pd.DataFrame({'tic': tickers, 'num_shares': num_shares, 'invested_amount': num_shares * last_prices})
    df_result = df_result.sort_values(['tic'], ignore_index=True)

    return df_result, remaining_cash

def update_portfolio(stock_pool, new_stock_pool, last_state):
    stock_dim = len(stock_pool)
    balance = last_state[0]
    last_prices = np.array(last_state[1:stock_dim + 1])
    last_shares = np.array(last_state[stock_dim + 1:stock_dim * 2 + 1])

    sell_mask = np.isin(stock_pool, new_stock_pool, invert=True)
    balance += np.sum(last_prices[sell_mask] * last_shares[sell_mask])

    num_shares_dict = dict(zip(stock_pool, last_shares))
    num_shares_list = [num_shares_dict.get(tic, 0) for tic in new_stock_pool]

    return balance, num_shares_list