import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

    def __init__(
        self, df, day=0, turbulence_threshold=140,
        initial=True, previous_state=[], model_name='', iteration=''
    ):
        super(StockEnvTrade, self).__init__()

        self.day = day
        self.df = df
        self.initial = initial
        self.previous_state = previous_state
        self.turbulence_threshold = turbulence_threshold
        self.model_name = model_name
        self.iteration = iteration

        # define spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))
        # 1 + 30 + 30 + 30 + 30 + 30 + 30 + 30 + 1 = 182
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(182,))

        # load first day’s data
        self.data = self.df.loc[self.day, :]
        self.terminal = False

        # build initial state => 182
        self.state = (
            [INITIAL_ACCOUNT_BALANCE] +
            self.data.adjcp.values.tolist() +
            [0]*STOCK_DIM +
            self.data.macd.values.tolist() +
            self.data.rsi.values.tolist() +
            self.data.cci.values.tolist() +
            self.data.adx.values.tolist()
        )
        # if 'VIX' in columns, append it once
        if 'VIX' in self.data.columns:
            self.state += [self.data.VIX.values[0]]

        # track reward, cost, trades, volume
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.volume = 0

        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []

        self._seed()

    def _get_dynamic_transaction_fee(self, vix):
        base_fee = 0.001
        delta = 0.0001
        return base_fee + (vix * delta)

    def _sell_stock(self, index, action):
        if len(self.state) > 182:  # means we have VIX appended
            vix = self.state[-1]
            transaction_fee = self._get_dynamic_transaction_fee(vix)
        else:
            transaction_fee = 0.001

        if self.turbulence < self.turbulence_threshold:
            current_shares = self.state[index + STOCK_DIM + 1]
            if current_shares > 0:
                shares_sold = min(abs(action), current_shares)
                self.state[0] += self.state[index+1] * shares_sold * (1 - transaction_fee)
                self.state[index + STOCK_DIM + 1] -= shares_sold
                self.cost += self.state[index+1] * shares_sold * transaction_fee
                self.trades += 1
                self.volume += shares_sold
        else:
            current_shares = self.state[index + STOCK_DIM + 1]
            if current_shares > 0:
                shares_sold = current_shares
                self.state[0] += self.state[index+1] * shares_sold * (1 - transaction_fee)
                self.state[index + STOCK_DIM + 1] = 0
                self.cost += self.state[index+1] * shares_sold * transaction_fee
                self.trades += 1
                self.volume += shares_sold

    def _buy_stock(self, index, action):
        if len(self.state) > 182:
            vix = self.state[-1]
            transaction_fee = self._get_dynamic_transaction_fee(vix)
        else:
            transaction_fee = 0.001

        if self.turbulence < self.turbulence_threshold:
            current_price = self.state[index+1]
            available_amount = self.state[0] // current_price
            shares_bought = min(available_amount, action)

            self.state[0] -= current_price * shares_bought * (1 + transaction_fee)
            self.state[index + STOCK_DIM + 1] += shares_bought
            self.cost += current_price * shares_bought * transaction_fee
            self.trades += 1
            self.volume += shares_bought
        else:
            pass

    def compute_reward(self, end_total_asset, begin_total_asset, volume, num_trades, vix):
        base_reward = end_total_asset - begin_total_asset
        alpha_volume = 0.01 * (volume / 1e6)
        alpha_trades = 0.05 * num_trades
        alpha_vix = 0.02 * (vix / 20)
        penalty = alpha_volume + alpha_trades + alpha_vix
        adjusted_reward = base_reward - penalty * abs(base_reward)
        return adjusted_reward

    def step(self, actions):
        self.terminal = (self.day >= (len(self.df.index.unique()) - 1))
        
        if self.terminal:
            # end of episode
            plt.plot(self.asset_memory, 'r')
            plt.savefig(f'results/account_value_trade_{self.model_name}_{self.iteration}.png')
            plt.close()

            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv(f'results/account_value_trade_{self.model_name}_{self.iteration}.csv')

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1:(STOCK_DIM+1)]) *
                np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)])
            )
            print("previous_total_asset:", self.asset_memory[0])
            print("end_total_asset:", end_total_asset)
            print("total_reward:", end_total_asset - self.asset_memory[0])
            print("total_cost:", self.cost)
            print("total trades:", self.trades)

            df_total_value.columns = ['account_value']
            df_total_value['daily_return'] = df_total_value['account_value'].pct_change(1)
            sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
            print("Sharpe:", sharpe)

            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.to_csv(f'results/account_rewards_trade_{self.model_name}_{self.iteration}.csv')

            return self.state, self.reward, self.terminal, {}
        else:
            self.volume = 0
            self.trades = 0

            begin_total_asset = (
                self.state[0] +
                sum(
                    np.array(self.state[1:(STOCK_DIM+1)]) *
                    np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)])
                )
            )

            actions = actions * HMAX_NORMALIZE
            if self.turbulence >= self.turbulence_threshold:
                actions = np.array([-HMAX_NORMALIZE] * STOCK_DIM)

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for idx in sell_index:
                self._sell_stock(idx, actions[idx])
            for idx in buy_index:
                self._buy_stock(idx, actions[idx])

            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.turbulence = self.data['turbulence'].values[0]

            # rebuild state => 182
            self.state = [
                self.state[0]
            ] + self.data.adjcp.values.tolist() + \
                list(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]) + \
                self.data.macd.values.tolist() + \
                self.data.rsi.values.tolist() + \
                self.data.cci.values.tolist() + \
                self.data.adx.values.tolist()

            if 'VIX' in self.data.columns:
                self.state += [self.data.VIX.values[0]]

            end_total_asset = (
                self.state[0] +
                sum(
                    np.array(self.state[1:(STOCK_DIM+1)]) *
                    np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)])
                )
            )
            self.asset_memory.append(end_total_asset)

            vix = self.state[-1] if len(self.state) == 182 else 0
            self.reward = self.compute_reward(end_total_asset, begin_total_asset, self.volume, self.trades, vix)
            self.rewards_memory.append(self.reward)
            self.reward *= REWARD_SCALING

            return self.state, self.reward, self.terminal, {}

    def reset(self):
        """
        Reset environment at the beginning of an episode or if 'initial' is True.
        """
        if self.initial:
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
            self.day = 0
            self.data = self.df.loc[self.day, :]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.terminal = False
            self.rewards_memory = []

            self.state = (
                [INITIAL_ACCOUNT_BALANCE] +
                self.data.adjcp.values.tolist() +
                [0]*STOCK_DIM +
                self.data.macd.values.tolist() +
                self.data.rsi.values.tolist() +
                self.data.cci.values.tolist() +
                self.data.adx.values.tolist()
            )
            if 'VIX' in self.data.columns:
                self.state += [self.data.VIX.values[0]]

        else:
            previous_total_asset = (
                self.previous_state[0] +
                sum(
                    np.array(self.previous_state[1:(STOCK_DIM+1)]) *
                    np.array(self.previous_state[(STOCK_DIM+1):(STOCK_DIM*2+1)])
                )
            )
            self.asset_memory = [previous_total_asset]
            self.day = 0
            self.data = self.df.loc[self.day, :]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.terminal = False
            self.rewards_memory = []

            self.state = [
                self.previous_state[0]
            ] + self.data.adjcp.values.tolist() + \
                self.previous_state[(STOCK_DIM+1):(STOCK_DIM*2+1)] + \
                self.data.macd.values.tolist() + \
                self.data.rsi.values.tolist() + \
                self.data.cci.values.tolist() + \
                self.data.adx.values.tolist()

            if 'VIX' in self.data.columns:
                self.state += [self.data.VIX.values[0]]

        return self.state

    def render(self, mode='human', close=False):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

