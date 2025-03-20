# common library
import pandas as pd
import numpy as np
import time
import gym
import gc

# Stable Baselines3 RL models
from stable_baselines3 import A2C, PPO, DDPG, TD3
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from preprocessing.preprocessors import *
from config import config

# customized env
from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_validation import StockEnvValidation
from env.EnvMultipleStock_trade import StockEnvTrade


def train_A2C(env_train, model_name, timesteps=25000):
    """A2C model"""

    start = time.time()
    model = A2C('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model

# def train_ACER(env_train, model_name, timesteps=25000):
#     start = time.time()
#     model = ACER('MlpPolicy', env_train, verbose=0)
#     model.learn(total_timesteps=timesteps)
#     end = time.time()

#     model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
#     print('Training time (A2C): ', (end - start) / 60, ' minutes')
#     return model


def train_DDPG(env_train, model_name, timesteps=10000):
    """DDPG model"""

    # add the noise objects for DDPG
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    start = time.time()
    model = DDPG('MlpPolicy', env_train, action_noise=action_noise, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (DDPG): ', (end-start)/60,' minutes')
    return model

def train_PPO(env_train, model_name, timesteps=50000):
    """PPO model"""

    start = time.time()
    model = PPO('MlpPolicy', env_train, ent_coef=0.005, batch_size=64, verbose=0)

    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model

# def train_GAIL(env_train, model_name, timesteps=1000):
#     """GAIL Model"""
#     #from stable_baselines.gail import ExportDataset, generate_expert_traj
#     start = time.time()
#     # generate expert trajectories
#     model = SAC('MLpPolicy', env_train, verbose=1)
#     generate_expert_traj(model, 'expert_model_gail', n_timesteps=100, n_episodes=10)

#     # Load dataset
#     dataset = ExpertDataset(expert_path='expert_model_gail.npz', traj_limitation=10, verbose=1)
#     model = GAIL('MLpPolicy', env_train, dataset, verbose=1)

#     model.learn(total_timesteps=1000)
#     end = time.time()

#     model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
#     print('Training time (PPO): ', (end - start) / 60, ' minutes')
#     return model


def DRL_prediction(df,
                   model,
                   name,
                   last_state,
                   iter_num,
                   turbulence_threshold,
                   initial,
                   stock_dim,
                   initial_arrangement,
                   initial_balance,
                   new_balance,
                   arrangement
                   ):
    ### make a prediction based on trained model###

    ## trading env
    # trade_stock_count = count_df.iloc[iter_num - 1,1]
    # trade_data = data_split(df, start=count_df.iloc[iter_num - 1,0], end=count_df.iloc[iter_num,0])
    env_trade = DummyVecEnv([lambda: StockEnvTrade(df,
                                                   stock_dim = stock_dim,
                                                   turbulence_threshold=turbulence_threshold,
                                                   initial=initial,
                                                   previous_state=last_state,
                                                   model_name=name,
                                                   iteration=iter_num,
                                                   initial_arrangement = initial_arrangement,
                                                   initial_balance = initial_balance,
                                                   new_balance = new_balance,
                                                   arrangement = arrangement)])
    obs_trade = env_trade.reset()

    for i in range(len(df.index.unique())):
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
        if i == (len(df.index.unique()) - 2):
            # last_state = env_trade.render()
            last_state = env_trade.envs[0].render()
    pd.DataFrame({"last_state": last_state}).to_csv(
        f"DRL-for-Trading/results/last_state_{name}_{iter_num}.csv", index=False
    )
    return last_state



def DRL_validation(model, test_data, test_env, test_obs) -> None:
    ###validation process###
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)


def get_validation_sharpe(iteration):
    ###Calculate Sharpe ratio based on validation results###
    df_total_value = pd.read_csv('DRL-for-Trading/results/account_value_validation_{}.csv'.format(iteration), index_col=0)
    df_total_value.columns = ['account_value_train']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / \
             df_total_value['daily_return'].std()
    return sharpe


def run_ensemble_strategy(df, report_date, start_date, val_start_date, stock_selected_df) -> None:
    """Ensemble Strategy that combines PPO, A2C and DDPG"""
    print("============Start Ensemble Strategy============")
    # for ensemble model, it's necessary to feed the last state
    # of the previous model to the current model as the initial state
    last_state_ensemble = []
    last_state_a2c = []
    last_state_ppo = []
    last_state_ddpg = []

    # last_state_ensemble = pd.read_csv('DRL-for-Trading/results/last_state_ensemble_63.csv')['last_state'].tolist()
    # last_state_a2c = pd.read_csv('DRL-for-Trading/results/last_state_a2c_63.csv')['last_state'].tolist()
    # last_state_ppo = pd.read_csv('DRL-for-Trading/results/last_state_ppo_63.csv')['last_state'].tolist()
    # last_state_ddpg = pd.read_csv('DRL-for-Trading/results/last_state_ddpg_63.csv')['last_state'].tolist()
    initial_arrangement = []
    initial_balance = 0
    
    ppo_sharpe_list = []
    ddpg_sharpe_list = []
    a2c_sharpe_list = []

    new_balance_ensemble, arrangement_ensemble = [], []
    new_balance_a2c, arrangement_a2c = [], []
    new_balance_ppo, arrangement_ppo = [], []
    new_balance_ddpg, arrangement_ddpg = [], []

    model_use = []
    vix_df = load_vix_data("DRL-for-Trading\data\VIXCLS.csv")
    vix_df['datadate'] = pd.to_datetime(vix_df['datadate'])
    vix_df = vix_df[(vix_df.datadate<val_start_date)& (vix_df.datadate>=start_date)] 
    VIX_threshold = np.quantile(vix_df.VIX.dropna().values, .90)
    # insample_turbulence = df[(df.datadate<val_start_date)& (df.datadate>=start_date)] 
    # insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    # insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)

    start = time.time()
    # start_idx = count_df[count_df['datadate'] == val_start_date].index[0]
    start_idx = report_date.index(val_start_date)
    # prev_stock_count, prev_stock_pool, prev_train_set, prev_val_set, prev_trade_set = get_training_data(df, stock_selected_df, start_date, report_date[start_idx-1], report_date[start_idx], report_date[start_idx+1])
    # stock_pool = start_state['tic'].tolist()
    for i in range(start_idx + 2, len(report_date)):
        end_date_index = df.index[df["datadate"] < report_date[i - 2]].to_list()[-1] #end of test?
        # start_date_index = end_date_index - report_date[i - 3]*count_df.iloc[i - 3, 1] + 1 # 30 stocks, I'll have a table for you too, maybe a dataframe?
        start_date_index = df.index[df["datadate"] >= report_date[i - 3]].to_list()[0]
        historical_VIX = df.iloc[start_date_index:(end_date_index + 1), :]

        historical_VIX = historical_VIX.drop_duplicates(subset=['datadate'])
        historical_VIX_mean = np.mean(historical_VIX.VIX.dropna().values)

        if historical_VIX_mean > VIX_threshold:
            VIX_threshold = VIX_threshold
        else:
            VIX_threshold = np.quantile(vix_df.VIX.dropna().values, 1)
        print("VIX_threshold: ", VIX_threshold)
        stock_count, stock_pool, train_set, val_set, trade_set = get_training_data(df, stock_selected_df, start_date, report_date[i - 2], report_date[i - 1], report_date[i])
        
        # if i == 26:
        if i == start_idx + 2:
            initial = True
            start_state, initial_balance= calculate_mean_variance(start_date, report_date[start_idx + 1], stock_pool)
            initial_arrangement = start_state['num_shares'].tolist()
            prev_stock_pool =  stock_pool.copy()
        else:
            initial = False
            # new_stock_pool = stock_pool
            new_balance_ensemble, arrangement_ensemble = update_portfolio(prev_stock_pool, stock_pool, last_state_ensemble)
            new_balance_a2c, arrangement_a2c = update_portfolio(prev_stock_pool, stock_pool, last_state_a2c)
            new_balance_ppo, arrangement_ppo = update_portfolio(prev_stock_pool, stock_pool, last_state_ppo)
            new_balance_ddpg, arrangement_ddpg = update_portfolio(prev_stock_pool, stock_pool, last_state_ddpg)
            prev_stock_pool = stock_pool.copy()

        
        ############## Environment Setup starts ##############
        ## training env
        # train = data_split(df, start=start_date, end=count_df.iloc[i - 2, 0])
        env_train = DummyVecEnv([lambda: StockEnvTrain(train_set, stock_dim = stock_count)])
        # train = data_split(df, start=start_date, end=unique_trade_date[i - rebalance_window - validation_window])
        # env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

        ## validation env
        # validation = data_split(df, start=count_df.iloc[i - 2, 0],
        #                         end=count_df.iloc[i - 1, 0])
        # val_stock_count = count_df.iloc[i - 2,1]
        # val_day_count = count_df.iloc[i - 2,2]
        # validation = data_split(df, start=unique_trade_date[i - rebalance_window - validation_window],
        #                         end=unique_trade_date[i - rebalance_window])
        env_val = DummyVecEnv([lambda: StockEnvValidation(val_set,stock_dim = stock_count,
                                                          turbulence_threshold=VIX_threshold,
                                                          iteration=i)])
        print(f"validation shape: {val_set.shape}")

        obs_val = env_val.reset()
        ############## Environment Setup ends ##############

        ############## Training and Validation starts ##############
        print("======Model training from: ", start_date, "to ",
              report_date[i - 2])
        # print("training: ",len(data_split(df, start=20090000, end=test.datadate.unique()[i-rebalance_window]) ))
        # print("==============Model Training===========")
        print("======A2C Training========")
        model_a2c = train_A2C(env_train, model_name="A2C_30k_dow_{}".format(i), timesteps=30000)
        print("======A2C Validation from: ", report_date[i - 2], "to ",
              report_date[i - 1])
        DRL_validation(model=model_a2c, test_data=val_set, test_env=env_val, test_obs=obs_val)
        sharpe_a2c = get_validation_sharpe(i)
        print("A2C Sharpe Ratio: ", sharpe_a2c)

        print("======PPO Training========")
        model_ppo = train_PPO(env_train, model_name="PPO_100k_dow_{}".format(i), timesteps=100000)
        print("======PPO Validation from: ", report_date[i - 2], "to ",
              report_date[i - 1])
        DRL_validation(model=model_ppo, test_data=val_set, test_env=env_val, test_obs=obs_val)
        sharpe_ppo = get_validation_sharpe(i)
        print("PPO Sharpe Ratio: ", sharpe_ppo)

        print("======DDPG Training========")
        # gc.collect()
        model_ddpg = train_DDPG(env_train, model_name="DDPG_10k_dow_{}".format(i), timesteps=10000)
        #model_ddpg = train_TD3(env_train, model_name="DDPG_10k_dow_{}".format(i), timesteps=20000)
        print("======DDPG Validation from: ", report_date[i - 2], "to ",
              report_date[i - 1])
        DRL_validation(model=model_ddpg, test_data=val_set, test_env=env_val, test_obs=obs_val)
        sharpe_ddpg = get_validation_sharpe(i)

        ppo_sharpe_list.append(sharpe_ppo)
        a2c_sharpe_list.append(sharpe_a2c)
        ddpg_sharpe_list.append(sharpe_ddpg)

        # Model Selection based on sharpe ratio
        if (sharpe_ppo >= sharpe_a2c) & (sharpe_ppo >= sharpe_ddpg):
            model_ensemble = model_ppo
            model_use.append('PPO')
        elif (sharpe_a2c > sharpe_ppo) & (sharpe_a2c > sharpe_ddpg):
            model_ensemble = model_a2c
            model_use.append('A2C')
        else:
            model_ensemble = model_ddpg
            model_use.append('DDPG')
        ############## Training and Validation ends ##############

        ############## Trading starts ##############
        print("======Trading from: ", report_date[i - 1], "to ", report_date[i])
        #print("Used Model: ", model_ensemble)
        last_state_ensemble = DRL_prediction(df=trade_set, model=model_ensemble, name="ensemble",
                                             last_state=last_state_ensemble, iter_num=i,
                                             turbulence_threshold=VIX_threshold,
                                             initial=initial, stock_dim = stock_count,
                                             initial_arrangement = initial_arrangement,
                                             initial_balance = initial_balance,
                                             new_balance = new_balance_ensemble,
                                             arrangement = arrangement_ensemble)
        # print("============Trading Done============")

        last_state_a2c = DRL_prediction(
            trade_set,
            model_a2c,
            "a2c",
            last_state_a2c,
            i,
            VIX_threshold,
            initial,
            stock_count,
            initial_arrangement,
            initial_balance,
            new_balance_a2c,
            arrangement_a2c
        )
        last_state_ppo = DRL_prediction(
            trade_set,
            model_ppo,
            "ppo",
            last_state_ppo,
            i,
            VIX_threshold,
            initial,
            stock_count,
            initial_arrangement,
            initial_balance,
            new_balance_ppo,
            arrangement_ppo
        )
        last_state_ddpg = DRL_prediction(
            trade_set,
            model_ddpg,
            "ddpg",
            last_state_ddpg,
            i,
            VIX_threshold,
            initial,
            stock_count,
            initial_arrangement,
            initial_balance,
            new_balance_ddpg,
            arrangement_ddpg
        )
        del model_a2c, model_ppo, model_ddpg, model_ensemble
        del env_train, env_val
        del train_set, val_set, trade_set
        gc.collect()

    pd.DataFrame(ppo_sharpe_list).to_csv("DRL-for-Trading/ppo_sharpe_list.csv")
    pd.DataFrame(a2c_sharpe_list).to_csv("DRL-for-Trading/a2c_sharpe_list.csv")
    pd.DataFrame(ddpg_sharpe_list).to_csv("DRL-for-Trading/ddpg_sharpe_list.csv")
    pd.DataFrame(model_use).to_csv("DRL-for-Trading/model_use.csv")
        ############## Trading ends ##############

    end = time.time()
    print("Ensemble Strategy took: ", (end - start) / 60, " minutes")
