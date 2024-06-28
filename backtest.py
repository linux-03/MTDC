import pandas as pd
import numpy as np
import dclassifier
from time import sleep


def w_avg(weights: list, rev_points: list):
    """
    Function to calculate the trend reversal point given the weights and reversl points for the single DC strategies

    Parameters
    ----------
    weights : list
        A list containing the weights for the strategies
    
    rev_points : list
        A list containing the predicted trend reversal points by a single strategie

    Return
    ------
    int
        predicted trend reversal point
    """
    weights = np.array(weights)
    rev_points = np.array(rev_points)

    #print("w_avg: ", weights, rev_points)
    try:
        W = np.dot(weights, rev_points)/np.sum(weights)
        #print('W: ', W)
    except ZeroDivisionError:
        print('The sum of the weights are not allowed to be zero!')
    
    return W


class Optimize_eps:
    # we init with a few parameters: 100 000 1 0.03 0.0083 0.1 0.03
    def __init__(self, data, thresholds, event_handlers: list, capital=100000, fix_fee=0, var_fee=0.00, borrow_costs=0.000, invest_rate=0.1, rf_rate=0.03):
        # here we set up all relevant backtesting variables like trading fees (fix + variable) and starting capital
        self.data = data
        self.weights = [0]*len(thresholds)
        
        self.thresh_event_handlers = event_handlers        
        
        self.base = capital
        self.other = 0
        self.wealth = 0
        self.exch = 0
        self.budget = capital
        self.fix_fee = fix_fee
        self.var_fee = var_fee
        self.borrow_costs = borrow_costs
        self.max_invest_rate = invest_rate
        self.rf_rate = rf_rate
        self.Y = None
        self.last_Up_price = 0
        self.TRP_list={}
        self.hist = []

    def reset(self):
        self.other = 0
        self.wealth = 0
        self.exch = 0
        self.last_Up_price = 0
        self.TRP_list={}
        self.hist = []
        self.base = self.budget
        self.other = 0
        for i in range(len(self.thresh_event_handlers)):
            self.thresh_event_handlers[i].reset()

    def sell(self, price):
        if self.base > 0:
            self.base = self.base - self.fix_fee
            self.other = self.base * price
            self.base = 0
            self.last_Up_price = price
    
    def buy(self, price):
        if self.other > 0 and price < self.last_Up_price:
            self.other = self.other - self.fix_fee
            self.base = self.other/price
            self.other = 0
    

    def hist(self):
        return self.hist

    def check_status(self, price, time):
        """
        This function is written for one specific strategy and executes trades

        Parameters
        ----------
        price : price for checking
        time : timepoint

        Return
        ------
        current portfolio value combined
        """
        TRP = 0
        TP_list_s = [0]*len(self.thresh_event_handlers)
        TP_list_b = [0]*len(self.thresh_event_handlers)
        W_s = 0 #weights for buying
        W_b = 0 #weights for selling

        #print('checkpoint')
        for i in range(len(self.thresh_event_handlers)):
            #print(self.thresh_event_handlers[i].step(price, time))
            even_data = self.thresh_event_handlers[i].step(price, time)
            os_length = even_data[0]
            trend = even_data[1]
            #print('Check:' ,os_length, ' , ', trend)
            # if we are alpha we skip
            if os_length == -1:
                continue

            #print('checkpoint')

            
            if trend ==  'Up':
                TP_list_s[i] = int(time + os_length)
                #print('Up', i, time, os_length)
                #print('TP_list', i, TP_list_s)
                #print('TRP', i, int(time + os_length))
                W_s += self.weights[i]
            else:
                TP_list_b[i] = int(time + os_length)
                #print('TP_list', i, TP_list_b)
                #print('TRP', i, int(time + os_length))
                #print('Down', i, time, os_length)
                W_b += self.weights[i]  

            
        if W_s >= W_b:
            TRP = w_avg(self.weights, TP_list_s)
            self.TRP_list[TRP] = 'sell'
        
        
        if W_b > W_s:
            TRP = w_avg(self.weights, TP_list_b)
            self.TRP_list[TRP] = 'buy'

        #print(self.TRP_list)
        if time in self.TRP_list:
            if self.TRP_list[time] == 'buy':
                self.buy(price)
            else:
                self.sell(price)
    
    
    def sharpe_ratio(self, ret, vol):
        """
        Returns the sharpe ratio
        of the given portfolio
        :param ret: annualized return of asset
        :param vol: annualized volatility of asset
        :return sharpe ratio : float
        """

        return (ret - self.rf_rate)/vol

    def annualize_vol(self, price, periods_per_year=252):
        """
        Annualizes the vol of a set of returns given the periods per year
        :param price: daily price time series of asset
        :return annualized volatility : float
        """
        r = price.pct_change().dropna()
        return r.std()*(periods_per_year**0.5)

    def annualize_rets(self, price, periods_per_year=252):
        """
        Annualizes a set of returns given the periods per year
        :param price: daily price time series of asset
        :return annualized return : float
        """
        r = price.pct_change().dropna()
        compounded_growth = (1+r).prod()
        n_periods = r.shape[0]
        #print("growth:", compounded_growth)
        if compounded_growth < 0:
            return (-1)*(pow((-1)*compounded_growth, (periods_per_year/n_periods))-1)
        return pow(compounded_growth, (periods_per_year/n_periods))-1

    def var_historic(self, data: pd.DataFrame, level = 99, price=False, agg=False):
        """
        Takes a time series of asset returns, returns the historic
        Value at Risk at a specified confidence level
        :param data: price data time series of asset
        :retun VaR : float
        """
        data = data.copy().pct_change().dropna()

        return -np.percentile(data, (100 - level))

    def evaluate(self, weights):
        """
        Evaluates the startegie in given data for given weights

        Parameters
        ----------
        weights:
            list of weights sorted accordingly to 

        Return
        ------
        sharpe ratio
        """
        self.reset()
        self.weights = weights
        self.last_Up_price = 0
        self.base = self.budget
        self.TRP_list = {}
        count = 0
        for el in self.data:
            self.check_status(el, count)
            if self.other > 0:
                self.hist.append((self.other - self.fix_fee)/el)
            else:
                self.hist.append(self.base)
            
            count += 1
        
        if self.other > 0:
            self.other = self.other - self.fix_fee
            self.base = self.other/self.data.iloc[-1]
            self.other = 0


        self.wealth = self.base - self.budget
        ret = self.wealth/self.budget

        #hist_sec = pd.Series(hist_sec)
        self.hist = pd.Series(self.hist)


        vol = self.hist.pct_change().dropna().std()
        sharpe = self.sharpe_ratio(self.annualize_rets(self.hist), self.annualize_vol(self.hist))
        print("vol", vol, "sharpe", sharpe, "ann_ret:", self.annualize_rets(self.hist), "return", ret, "capital", self.budget)


        # [sharpe, ret, vol]
        return ret