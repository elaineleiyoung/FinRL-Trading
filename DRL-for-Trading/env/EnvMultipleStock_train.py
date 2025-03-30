import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# constants
HMAX_NORMALIZE = 100
INITIAL_ACCOUNT_BALANCE = 1000000
STOCK_DIM = 30
REWARD_SCALING = 1e-4

class StockEnvTrain(gym.Env):
    """A stock trading environment for OpenAI gym (training mode)."""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, day=0):
        """
        Args:
            df (pd.DataFrame): Preprocessed stock data with columns for adjcp, macd, rsi, cci, adx, VIX, etc.
            day (int): Starting index.
        """
        super(StockEnvTrain, self).__init__()

        self.day = day
        self.df = df
        
        # track volume (shares traded) & number of trades each step
        self.volume = 0
        self.trades = 0
        self.cost = 0
        self.reward = 0

        # define spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))
        # We want 1 (cash) + 30 (prices) + 30 (shares) + 30 (macd) + 30 (rsi) + 30 (cci) + 30 (adx) + 1 (VIX) = 182
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(182,))

        # load the first day’s data
        self.data = self.df.loc[self.day, :]
        self.terminal = False

        # initialize state with 1+30+30+30+30+30+30+30+1 = 182
        self.state = [
            INITIAL_ACCOUNT_BALANCE
        ] + self.data.adjcp.values.tolist() + \
            [0]*STOCK_DIM + \
            self.data.macd.values.tolist() + \
            self.data.rsi.values.tolist() + \
            self.data.cci.values.tolist() + \
            self.data.adx.values.tolist() + \
            [self.data.VIX.values[0]]  # appended exactly once here

        # track asset value & rewards
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []

        self._seed()

    # ============== Transaction Fee (VIX-Based) ==============
    def _get_dynamic_transaction_fee(self, vix):
        base_fee = 0.001
        delta = 0.0001
        return base_fee + (vix * delta)

    # ============== Sell ==============
    def _sell_stock(self, index, action):
        """Execute a sell action based on the sign of action."""
        vix = self.state[-1]  # last element is VIX
        transaction_fee = self._get_dynamic_transaction_fee(vix)

        current_shares = self.state[index + STOCK_DIM + 1]
        if current_shares > 0:
            shares_sold = min(abs(action), current_shares)
            self.state[0] += self.state[index+1] * shares_sold * (1 - transaction_fee)
            self.state[index + STOCK_DIM + 1] -= shares_sold
            self.cost += self.state[index+1] * shares_sold * transaction_fee
            self.trades += 1
            self.volume += shares_sold

    # ============== Buy ==============
    def _buy_stock(self, index, action):
        vix = self.state[-1]
        transaction_fee = self._get_dynamic_transaction_fee(vix)

        current_price = self.state[index+1]
        available_amount = self.state[0] // current_price
        shares_bought = min(available_amount, action)

        self.state[0] -= current_price * shares_bought * (1 + transaction_fee)
        self.state[index + STOCK_DIM + 1] += shares_bought

        self.cost += current_price * shares_bought * transaction_fee
        self.trades += 1
        self.volume += shares_bought
        
    # ============== Step ==============
    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            # end of episode
            plt.plot(self.asset_memory, 'r')
            plt.savefig('results/account_value_train.png')
            plt.close()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1:(STOCK_DIM+1)]) *
                np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory, columns=['account_value'])
            df_total_value['daily_return'] = df_total_value['account_value'].pct_change(1)
            df_total_value.to_csv('results/account_value_train.csv', index=False)
            
            return self.state, self.reward, self.terminal, {}
        else:
            # reset volume/trades for this step
            self.volume = 0
            self.trades = 0

            # 1) compute initial portfolio value
            begin_total_asset = (
                self.state[0] +
                sum(
                    np.array(self.state[1:(STOCK_DIM+1)]) *
                    np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)])
                )
            )

            # 2) scale actions
            actions = actions * HMAX_NORMALIZE

            # 3) sell first, then buy
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            # 4) executes sells
            for idx in sell_index:
                self._sell_stock(idx, actions[idx])

            # 5) executes buys
            for idx in buy_index:
                self._buy_stock(idx, actions[idx])

            # 6) move to next day
            self.day += 1
            self.data = self.df.loc[self.day, :]

            # 7) rebuild state => 1+30+30+30+30+30+30+30+1=182
            self.state = [
                self.state[0]
            ] + self.data.adjcp.values.tolist() + \
                list(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]) + \
                self.data.macd.values.tolist() + \
                self.data.rsi.values.tolist() + \
                self.data.cci.values.tolist() + \
                self.data.adx.values.tolist() + \
                [self.data.VIX.values[0]]

            # 8) compute new asset
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1:(STOCK_DIM+1)]) *
                np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)])
            )
            self.asset_memory.append(end_total_asset)

            # 9) custom reward logic
            volume = self.volume
            num_trades = self.trades
            vix = self.state[-1]
            self.reward = self.compute_reward(end_total_asset, begin_total_asset, volume, num_trades, vix)
            self.rewards_memory.append(self.reward)

            # 10) scale reward
            self.reward *= REWARD_SCALING

            return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.volume = 0
        self.trades = 0
        self.cost = 0
        self.terminal = False
        self.reward = 0
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []

        # build initial state => 182
        self.state = (
            [INITIAL_ACCOUNT_BALANCE] +
            self.data.adjcp.values.tolist() +
            [0]*STOCK_DIM +
            self.data.macd.values.tolist() +
            self.data.rsi.values.tolist() +
            self.data.cci.values.tolist() +
            self.data.adx.values.tolist() +
            [self.data.VIX.values[0]]
        )
        return self.state

    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_dynamic_transaction_fee(self, vix):
        """
        Compute transaction cost dynamically based on VIX index.
        """
        base_fee = 0.001
        delta = 0.0001
        return base_fee + (vix * delta)

    def compute_reward(self, end_total_asset, begin_total_asset, volume, num_trades, vix):
        """
        Example reward with penalty for volume/trades/VIX.
        """
        base_reward = end_total_asset - begin_total_asset
        alpha_volume = 0.01 * (volume / 1e6)
        alpha_trades = 0.05 * num_trades
        alpha_vix = 0.02 * (vix / 20)
        penalty = alpha_volume + alpha_trades + alpha_vix
        adjusted_reward = base_reward - penalty * abs(base_reward)
        return adjusted_reward
