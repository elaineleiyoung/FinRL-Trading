import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from config import config

HMAX_NORMALIZE = 100
INITIAL_ACCOUNT_BALANCE = 1000000
STOCK_DIM = 30
REWARD_SCALING = 1e-4

class StockEnvValidation(gym.Env):
    """
    Validation environment with VIX-based transaction costs and a penalty-based reward.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, stock_dim, day = 0, turbulence_threshold=140, iteration='',
                 if_mvo=False, initial_arrangement = None, initial_balance = None):
        #super(StockEnv, self).__init__()
        #money = 10 , scope = 1
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        # action_space normalization and shape is STOCK_DIM
        self.action_space = spaces.Box(low = -1, high = 1,shape = (self.stock_dim,)) 
        # Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30] 
        # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (self.stock_dim*6+2,))
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day,:]
        self.terminal = False     
        self.turbulence_threshold = turbulence_threshold

        self.if_mvo = if_mvo
        if self.if_mvo:
            self.initial_arrangement = initial_arrangement
            self.initial_balance = initial_balance
            self.state = [self.initial_balance] + \
                          self.data.adjcp.values.tolist() + \
                          self.initial_arrangement + \
                          self.data.macd.values.tolist() + \
                          self.data.rsi.values.tolist()  + \
                          self.data.cci.values.tolist()  + \
                          self.data.adx.values.tolist() 
            init_total_asset = self.initial_balance+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            self.asset_memory = [init_total_asset]
        else:

        # initalize state

            self.state = [INITIAL_ACCOUNT_BALANCE] + \
                        self.data.adjcp.values.tolist() + \
                        [0]*self.stock_dim + \
                        self.data.macd.values.tolist() + \
                        self.data.rsi.values.tolist() + \
                        self.data.cci.values.tolist() + \
                        self.data.adx.values.tolist() + \
                        [self.data.VIX.values[0]]
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.rewards_memory = []
        self.iteration = iteration

        self._seed()

    def _get_dynamic_transaction_fee(self, vix):
        base_fee = 0.001
        delta = 0.0001
        return base_fee + (vix * delta)

    def _sell_stock(self, index, action):
        if abs(action) < 1:
            return  # Skip small trades
        
        vix = self.state[-1]
        transaction_fee = self._get_dynamic_transaction_fee(vix)

        current_shares = self.state[index + self.stock_dim + 1]
        if current_shares <= 0:
            return # No shares to sell
        
        # perform sell action based on the sign of the action
        if self.turbulence<self.turbulence_threshold:
            shares_sold = min(abs(action), current_shares)
        else:
            shares_sold = current_shares
        stock_price = self.state[index + 1]
        self.state[0] += stock_price * shares_sold * (1 - transaction_fee)
        self.state[index + self.stock_dim + 1] -= shares_sold
        self.cost += stock_price * shares_sold * transaction_fee
        self.trades += 1
        self.volume += shares_sold
    
    def _buy_stock(self, index, action):
        if abs(action) < 1:
            return  # Skip small trades
        vix = self.state[-1]
        transaction_fee = self._get_dynamic_transaction_fee(vix)
        # perform buy action based on the sign of the action
        if self.turbulence< self.turbulence_threshold:
            current_price = self.state[index + 1]
            available_amount = self.state[0] // current_price
            shares_bought = min(abs(action), available_amount)
            # print('available_amount:{}'.format(available_amount))
            
            #update balance
            self.state[0] -= current_price * shares_bought * (1 + transaction_fee)

            self.state[index+self.stock_dim+1] += shares_bought
            
            self.cost+=current_price * shares_bought * transaction_fee
            self.trades+=1
            self.volume += shares_bought
        else:
            pass

    def step(self, actions):
        self.terminal = self.day >= (len(self.df.index.unique()) - 1)

        if self.terminal:
            plt.plot(self.asset_memory,'r')
            plt.savefig(config.RESULTS_DIR + '/account_value_validation_{}.png'.format(self.iteration))
            plt.close()
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv(config.RESULTS_DIR + '/account_value_validation_{}.csv'.format(self.iteration))
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            #print("previous_total_asset:{}".format(self.asset_memory[0]))           

            #print("end_total_asset:{}".format(end_total_asset))
            #print("total_reward:{}".format(self.state[0]+sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):61]))- self.asset_memory[0] ))
            #print("total_cost: ", self.cost)
            #print("total trades: ", self.trades)

            df_total_value.columns = ['account_value']
            df_total_value['daily_return']=df_total_value.pct_change(1)
            sharpe = (4**0.5)*df_total_value['daily_return'].mean()/ \
                  df_total_value['daily_return'].std()
            #print("Sharpe: ",sharpe)
            
            #df_rewards = pd.DataFrame(self.rewards_memory)
            #df_rewards.to_csv('results/account_rewards_trade_{}.csv'.format(self.iteration))
            
            # print('total asset: {}'.format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))))
            #with open('obs.pkl', 'wb') as f:  
            #    pickle.dump(self.state, f)
            
            return self.state, self.reward, self.terminal,{}

        else:
            # reset volume/trades
            self.volume = 0
            self.trades = 0

            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1:(self.stock_dim+1)]) *
                np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)])
            )

            actions = actions * HMAX_NORMALIZE
            #actions = (actions.astype(int))
            if self.turbulence>=self.turbulence_threshold:
                actions=np.array([-HMAX_NORMALIZE]*self.stock_dim)
            begin_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            #print("begin_total_asset:{}".format(begin_total_asset))
            
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for idx in sell_index:
                self._sell_stock(idx, actions[idx])
            for idx in buy_index:
                self._buy_stock(idx, actions[idx])

            # next day
            self.day += 1
            self.data = self.df.loc[self.day,:]         
            self.turbulence = self.data['turbulence'].values[0]
            #print(self.turbulence)
            #load next state
            # print("stock_shares:{}".format(self.state[29:]))
            self.state =  [self.state[0]] + \
                    self.data.adjcp.values.tolist() + \
                    list(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]) + \
                    self.data.macd.values.tolist() + \
                    self.data.rsi.values.tolist() + \
                    self.data.cci.values.tolist() + \
                    self.data.adx.values.tolist() + \
                [self.data.VIX.values[0]]
            
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            self.asset_memory.append(end_total_asset)

            volume = self.volume
            num_trades = self.trades
            vix = self.state[-1]

            self.reward = self.compute_reward(end_total_asset, begin_total_asset, volume, num_trades, vix)
            self.rewards_memory.append(self.reward)
            self.reward *= REWARD_SCALING

            return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.turbulence = 0
        self.volume = 0
        self.trades = 0
        self.cost = 0
        self.reward = 0
        self.terminal = False

        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        #initiate state
        if self.if_mvo:
            self.state = [INITIAL_ACCOUNT_BALANCE] + \
                        self.data.adjcp.values.tolist() + \
                        [0]*self.stock_dim + \
                        self.data.macd.values.tolist() + \
                        self.data.rsi.values.tolist()  + \
                        self.data.cci.values.tolist()  + \
                        self.data.adx.values.tolist()  + \
              [self.data.VIX.values[0]]
            init_total_asset = self.initial_balance+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            self.asset_memory = [init_total_asset]
        else:
            self.state = [INITIAL_ACCOUNT_BALANCE] + \
                        self.data.adjcp.values.tolist() + \
                        [0]*self.stock_dim + \
                        self.data.macd.values.tolist() + \
                        self.data.rsi.values.tolist()  + \
                        self.data.cci.values.tolist()  + \
                        self.data.adx.values.tolist()  + \
                          [self.data.VIX.values[0]]
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        return self.state

    def render(self, mode='human', close=False):
        return self.state
    
    def _get_dynamic_transaction_fee(self, vix):
        base_fee = 0.001
        delta = 0.0001
        return base_fee + (vix * delta)

    def compute_reward(self, end_total_asset, begin_total_asset, volume, num_trades, vix):
        base_reward = end_total_asset - begin_total_asset
        alpha_volume = 0.01 * (volume / 1e6)
        alpha_trades = 0.05 * num_trades
        alpha_vix = 0.05 * (vix / 20)
        penalty = alpha_volume + alpha_trades + alpha_vix
        adjusted_reward = base_reward - penalty * abs(base_reward)
        return adjusted_reward

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def seed(self, seed=None):
        return self._seed(seed)
