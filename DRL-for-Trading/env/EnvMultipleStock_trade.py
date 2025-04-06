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

class StockEnvTrade(gym.Env):
    """
    A stock trading environment for OpenAI gym (trade mode),
    with dynamic VIX-based transaction fees.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, stock_dim, day = 0,turbulence_threshold=140
                 ,initial=True, previous_state=[], model_name='', iteration='',
                 initial_arrangement = None, initial_balance = None,
                 new_balance = None, arrangement = None, if_fix = False, if_mvo = True):
        #super(StockEnv, self).__init__()
        #money = 10 , scope = 1
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.if_mvo = if_mvo
        self.initial = initial
        self.if_fix = if_fix
        if self.initial:
            self.initial_arrangement = initial_arrangement
            self.initial_balance = initial_balance
        else:
            self.new_balance = new_balance
            self.arrangement = arrangement
        self.previous_state = previous_state\
        
        self.turbulence_threshold = turbulence_threshold
        self.model_name = model_name
        self.iteration = iteration

        # action_space normalization and shape is STOCK_DIM
        self.action_space = spaces.Box(low = -1, high = 1,shape = (self.stock_dim,)) 
        # Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30] 
        # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (self.stock_dim*6+2,))
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day,:]
        self.terminal = False     
        self.turbulence_threshold = turbulence_threshold
        # initalize state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                      self.data.adjcp.values.tolist() + \
                      [0]*self.stock_dim + \
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist() + \
                      self.data.cci.values.tolist() + \
                      self.data.adx.values.tolist()
        
        # if 'VIX' in columns, append it once
        if 'VIX' in self.data.columns:
            self.state += [self.data.VIX.values[0]]

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        # memorize all the total balance change
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        #self.reset()
        self._seed()
        self.model_name=model_name        
        self.iteration=iteration

    def _get_dynamic_transaction_fee(self, vix):
        base_fee = 0.001
        delta = 0.0001
        return base_fee + (vix * delta)

    def _sell_stock(self, index, action):
        if abs(action) < 1:
            return

        # perform sell action based on the sign of the action
        if len(self.state) > 182:  # means we have VIX appended
            vix = self.state[-1]
            transaction_fee = self._get_dynamic_transaction_fee(vix)
        else:
            transaction_fee = 0.001
        if self.turbulence<self.turbulence_threshold:
            if self.state[index+self.stock_dim+1] > 0:
                #update balance
                self.state[0] += \
                self.state[index+1]*min(abs(action),self.state[index+self.stock_dim+1]) * \
                 (1- transaction_fee)
                
                self.state[index+self.stock_dim+1] -= min(abs(action), self.state[index+self.stock_dim+1])
                self.cost +=self.state[index+1]*min(abs(action),self.state[index+self.stock_dim+1]) * \
                 transaction_fee
                self.trades+=1
            else:
                pass
        else:
            # if turbulence goes over threshold, just clear out all positions 
            if self.state[index+self.stock_dim+1] > 0:
                #update balance
                self.state[0] += self.state[index+1]*self.state[index+self.stock_dim+1]* \
                              (1- transaction_fee)
                self.state[index+self.stock_dim+1] =0
                self.cost += self.state[index+1]*self.state[index+self.stock_dim+1]* \
                              transaction_fee
                self.trades+=1
            else:
                pass
    
    def _buy_stock(self, index, action):
        if abs(action) < 1:
            return
        if len(self.state) > 182:  # means we have VIX appended
            vix = self.state[-1]
            transaction_fee = self._get_dynamic_transaction_fee(vix)
        else:
            transaction_fee = 0.001
        # perform buy action based on the sign of the action    
        if self.turbulence< self.turbulence_threshold:
            available_amount = self.state[0] // self.state[index+1]
            # print('available_amount:{}'.format(available_amount))
            
            #update balance
            self.state[0] -= self.state[index+1]*min(available_amount, action)* \
                              (1+ transaction_fee)

            self.state[index+self.stock_dim+1] += min(available_amount, action)
            
            self.cost+=self.state[index+1]*min(available_amount, action)* \
                              transaction_fee
            self.trades+=1
        else:
            pass

    def compute_reward(self, end_total_asset, begin_total_asset, volume, num_trades, vix):
        base_reward = end_total_asset - begin_total_asset
        alpha_volume = 0.01 * (volume / 1e6)
        alpha_trades = 0.05 * num_trades
        alpha_vix = 0.05 * (vix / 20)
        penalty = alpha_volume + alpha_trades + alpha_vix
        adjusted_reward = base_reward - penalty * abs(base_reward)
        return adjusted_reward

    def step(self, actions):
        self.terminal = (self.day >= (len(self.df.index.unique()) - 1))
        
        if self.terminal:
            plt.plot(self.asset_memory,'r')
            plt.savefig(config.RESULTS_DIR + '/account_value_trade_{}_{}.png'.format(self.model_name, self.iteration))
            plt.close()
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv(config.RESULTS_DIR + '/account_value_trade_{}_{}.csv'.format(self.model_name, self.iteration))
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            print("previous_total_asset:{}".format(self.asset_memory[0]))           

            print("end_total_asset:{}".format(end_total_asset))
            print("total_reward:{}".format(self.state[0]+sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))- self.asset_memory[0] ))
            print("total_cost: ", self.cost)
            print("total trades: ", self.trades)

            df_total_value.columns = ['account_value']
            df_total_value['daily_return']=df_total_value.pct_change(1)
            sharpe = (4**0.5)*df_total_value['daily_return'].mean()/ \
                  df_total_value['daily_return'].std()
            print("Sharpe: ",sharpe)
            
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.to_csv(config.RESULTS_DIR + '/account_rewards_trade_{}_{}.csv'.format(self.model_name, self.iteration))
            
            # print('total asset: {}'.format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))))
            #with open('obs.pkl', 'wb') as f:  
            #    pickle.dump(self.state, f)
            
            return self.state, self.reward, self.terminal,{}

            
        else:
            self.volume = 0
            self.trades = 0

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
                    self.data.adx.values.tolist()
            if 'VIX' in self.data.columns:
                self.state += [self.data.VIX.values[0]]
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            self.asset_memory.append(end_total_asset)

            vix = self.state[-1] if len(self.state) == 182 else 0
            self.reward = self.compute_reward(end_total_asset, begin_total_asset, self.volume, self.trades, vix)
            self.rewards_memory.append(self.reward)
            self.reward *= REWARD_SCALING

            return self.state, self.reward, self.terminal, {}

    def reset(self):
            self.day = 0
            self.data = self.df.loc[self.day, :]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.terminal = False
            self.rewards_memory = []
            if self.initial:
              if self.if_mvo:
              self.state = [self.initial_balance] + \
                            self.data.adjcp.values.tolist() + \
                            self.initial_arrangement + \
                            self.data.macd.values.tolist() + \
                            self.data.rsi.values.tolist()  + \
                            self.data.cci.values.tolist()  + \
                            self.data.adx.values.tolist() 
                if 'VIX' in self.data.columns:
                    self.state += [self.data.VIX.values[0]]
             else:
                self.state = [INITIAL_ACCOUNT_BALANCE] + \
                              self.data.adjcp.values.tolist() + \
                              [0]*self.stock_dim + \
                              self.data.macd.values.tolist() + \
                              self.data.rsi.values.tolist()  + \
                              self.data.cci.values.tolist()  + \
                              self.data.adx.values.tolist()
                if 'VIX' in self.data.columns:
                    self.state += [self.data.VIX.values[0]]
                self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        else:
            if self.if_fix:
                previous_total_asset = self.previous_state[0]+ \
                sum(np.array(self.previous_state[1:(self.stock_dim+1)])*np.array(self.previous_state[(self.stock_dim+1):(self.stock_dim*2+1)]))

                self.state = [ self.previous_state[0]] + \
                            self.data.adjcp.values.tolist() + \
                            self.previous_state[(self.stock_dim+1):(self.stock_dim*2+1)]+ \
                            self.data.macd.values.tolist() + \
                            self.data.rsi.values.tolist()  + \
                            self.data.cci.values.tolist()  + \
                            self.data.adx.values.tolist() 
                if 'VIX' in self.data.columns:
                    self.state += [self.data.VIX.values[0]]
                self.asset_memory = [previous_total_asset]

            else:
                self.state = [self.new_balance] + \
                            self.data.adjcp.values.tolist() + \
                            self.arrangement+ \
                            self.data.macd.values.tolist() + \
                            self.data.rsi.values.tolist()  + \
                            self.data.cci.values.tolist()  + \
                            self.data.adx.values.tolist() 
                if 'VIX' in self.data.columns:
                    self.state += [self.data.VIX.values[0]]
                previous_total_asset = self.new_balance+ \
                sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
                self.asset_memory = [previous_total_asset]
        return self.state

    def render(self, mode='human', close=False):
        return self.state

    def seed(self, seed=None):
        return self._seed(seed)
