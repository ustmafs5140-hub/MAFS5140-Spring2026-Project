import pandas as pd

"""
STUDENT INSTRUCTIONS:
1. This Strategy class is where you will implement your own trading strategy.
2. The current implementation is just a SIMPLE EXAMPLE (Moving Average Trend Following) 
   provided for your reference. Please modify this class to build your own strategy.
3. You may create new Python scripts and import them into this file if you 
   want to organize your code. 
4. IMPORTANT: Do NOT modify any other existing scripts in the backtest 
   framework. Changing core engine files may break the backtester and cause 
   evaluation errors.
"""
import pandas as pd
import numpy as np
import cvxpy as cp

class Strategy:
    def __init__(self):
        # 最优参数（基于验证集搜索）
        self.lookback = 20
        self.top_n = 44
        self.rebalance_freq = 39
        self.w_momentum = 0.6
        self.w_lowvol = 0.2
        self.w_volume = 0.2
        
        self.price_history = pd.DataFrame()
        self.volume_history = pd.DataFrame()
        self.bar_counter = 0
        self.last_rebalance_bar = 0
        self.current_weights = None
        
    def step(self, current_market_data):
        self.bar_counter += 1
        current_prices = current_market_data['close']
        current_volumes = current_market_data['volume']
        
        self.price_history = pd.concat([self.price_history, current_prices.to_frame().T], ignore_index=True)
        self.volume_history = pd.concat([self.volume_history, current_volumes.to_frame().T], ignore_index=True)
        
        need_rebalance = (self.current_weights is None) or (self.bar_counter - self.last_rebalance_bar >= self.rebalance_freq)
        
        if need_rebalance and len(self.price_history) >= self.lookback + 1:
            self.current_weights = self._rebalance()
            self.last_rebalance_bar = self.bar_counter
        elif self.current_weights is None:
            n = len(current_prices)
            self.current_weights = pd.Series(1.0 / n, index=current_prices.index)
            self.last_rebalance_bar = self.bar_counter
            
        return self.current_weights
    
    def _rebalance(self):
        prices = self.price_history.iloc[-self.lookback-1:]
        volumes = self.volume_history.iloc[-self.lookback-1:]
        
        ret = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
        pct_chg = prices.pct_change().dropna()
        vol = pct_chg.std()
        avg_vol = volumes.iloc[:-1].mean()
        vol_ratio = volumes.iloc[-1] / avg_vol
        
        ret_norm = (ret - ret.min()) / (ret.max() - ret.min()) if ret.max() > ret.min() else pd.Series(0.5, index=ret.index)
        vol_norm = 1 - (vol - vol.min()) / (vol.max() - vol.min()) if vol.max() > vol.min() else pd.Series(0.5, index=vol.index)
        vol_ratio_norm = (vol_ratio - vol_ratio.min()) / (vol_ratio.max() - vol_ratio.min()) if vol_ratio.max() > vol_ratio.min() else pd.Series(0.5, index=vol_ratio.index)
        
        score = (self.w_momentum * ret_norm + self.w_lowvol * vol_norm + self.w_volume * vol_ratio_norm)
        selected = score.nlargest(self.top_n).index
        
        if len(selected) == 0:
            return pd.Series(0.0, index=score.index)
        
        selected_returns = pct_chg[selected]
        Sigma = selected_returns.cov().values
        Sigma += np.eye(Sigma.shape[0]) * 1e-8
        
        n = len(selected)
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, Sigma))
        constraints = [w >= 0, cp.sum(w) == 1]
        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=False)
        
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            optimal_weights = np.ones(n) / n
        else:
            optimal_weights = w.value
        
        # 处理微小负数并归一化
        optimal_weights = np.clip(optimal_weights, 0, None)
        optimal_weights = optimal_weights / optimal_weights.sum()
        
        weights = pd.Series(0.0, index=score.index)
        weights[selected] = optimal_weights
        return weights
