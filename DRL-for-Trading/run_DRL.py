# common library
import pandas as pd
import numpy as np
import time
from stable_baselines3.common.vec_env import DummyVecEnv

# preprocessor
from preprocessing.preprocessors import *
# config
from config.config import *
# model
from model.models_0318 import *
import os

def run_model() -> None:
    """Train the model."""
    start_date = pd.to_datetime('2012-09-04', format='%Y-%m-%d')
    val_start_date = pd.to_datetime('2018-09-04', format='%Y-%m-%d') #
    trade_start_date = pd.to_datetime('2018-12-03', format='%Y-%m-%d')
    stock_selected_df = get_selected_stock(start_date)
    report_date = stock_selected_df.datadate.unique().tolist()

    preprocessed_path = 'DRL-for-Trading\done_data_0318.csv'
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)
        data["datadate"] = pd.to_datetime(data["datadate"])
    else:
        data = preprocess_data(if_vix = True, selected_stocks = True,
                               stock_selected_df = stock_selected_df,
                               start_date = start_date)
        data.to_csv(preprocessed_path)

    print(data.head())
    print(data.size)

    # count_path = "DRL-for-Trading\preprocessing\count.csv"
    # count_df = pd.read_csv(count_path, index_col=0)
    # # start_date = count_df.iloc[0, 0]
    # end_date = data.iloc[-1, 0]
    # count_df.loc[count_df.shape[0]] = [end_date,0,0]
    # # count_df[['datadate','stock_count','days_count']] = count_df[['datadate','stock_count','days_count']].astype(int)

    # unique_trade_date = data[(data.datadate > val_start_date)&(data.datadate <= end_date)].datadate.unique()
    # print(unique_trade_date)

    # stock_selected_df['trade_date'] = pd.to_datetime(stock_selected_df['trade_date'])
    # stock_selected_df['trade_date'] = stock_selected_df['trade_date'].apply(lambda x: int(x.strftime("%Y%m%d")))
    
    # vix_data = load_vix_data("data/VIXCLS.csv")
    ## Ensemble Strategy
    run_ensemble_strategy(df=data, 
                          report_date = report_date,
                          start_date = start_date,
                          val_start_date = val_start_date,
                          stock_selected_df = stock_selected_df)

    #_logger.info(f"saving model version: {_version}")

if __name__ == "__main__":
    run_model()
