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
    try:
        W = np.dot(weights, rev_points)/np.sum(weights)

    except ZeroDivisionError:
        print('The sum of the weights are not allowed to be zero!')
    
    return W


class Optimize_eps:
    # we init with a few parameters: 100 000 1 0.03 0.0083 0.1 0.03
    def __init__(self, data, thresholds, classifiers: list, regressors:list, capital=100000, fix_fee=0, var_fee=0.00, borrow_costs=0.000, invest_rate=0.1, rf_rate=0.03):
        # here we set up all relevant backtesting variables like trading fees (fix + variable) and starting capital
        self.data = data
        self.weights = [0]*len(thresholds)
        self.regressors = regressors
        self.thresh_event_handlers = []
        
        # initialize DC_EVENT_HANDLERS
        for j in thresholds:
            self.thresh_event_handlers.append(dclassifier.DC_EVENT_HANDLER(thresholds[j], classifiers[j]))
        
        
        self.capital = capital
        self.fix_fee = fix_fee
        self.var_fee = var_fee
        self.borrow_costs = borrow_costs
        self.max_invest_rate = invest_rate
        self.rf_rate = rf_rate
        self.Y = None

    def reset(self):
        self.long_pos = {}
        self.short_pos = {}
        self.short = False
        self.long = False     

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
        TP_list_l = [0]*len(self.thresh_event_handlers)
        W_s = 0 #weights for buying
        W_b = 0 #weights for selling


        for i in range(len(self.thresh_event_handlers)):
            os_length, trend  = self.thresh_event_handlers[i].step(price, time)

            # if we are alpha we skip
            if not os_length:
                continue

            
            if trend ==  'Up':
                TP_list_s[i] = int(time + os_length)
                W_b += self.weights[i]
            else:
                TP_list_l[i] = int(time + os_length)
                W_s += self.weights[i]  

        if W_s > W_b:
            TRP = w_avg(self.weights, TP_list_s)
        

        # wenn wir prognostizieren zu steigen gehen wir eine  long position ein
        if wap_2 > wap_1:
        #    self.enter_long(time, price_1)
            
            #maximales capital das wir investieren wollen
            usable_cap = self.capital*self.max_invest_rate

            #gesmatkapital sinkt da wir kaufen
            self.capital -= usable_cap

            #kaufen maximale share anzahl mit fix_fees und varfees includiert
            shares = (usable_cap - self.fix_fee - usable_cap*self.var_fee)/price_1
            #self.capital += ((price_2/price_1))*self.capital*self.max_invest_rate

            
            # ACHTUNG! Wir addieren nie unseren Gewinn direkt zum Capital, da wir eigentlich mehrere Positionen gleichzeitig eingehen,
            # daher müssen zuerst alle positionen eingegangen werden, bevor wir das kapital mit unseren gewinnen/verlusten erhöhen/senken
            return shares*price_2 - self.fix_fee - shares*price_2*self.var_fee

        # wenn wir prognostizieren zu fallen gehen wir eine short position ein
        if wap_2 < wap_1:
            #print(1 - (price_2/price_1))
            #self.capital -= ((price_2/price_1) - 1 )*self.capital*self.max_invest_rate

            #maximales capital das wir investieren wollen
            usable_cap = self.capital*self.max_invest_rate
            #self.capital -= usable_cap
            self.capital -= usable_cap*self.var_fee - self.fix_fee
            # here is the amount of shares
            shares = usable_cap/price_1
            usable_cap -= usable_cap*self.var_fee - self.fix_fee
            #if shares >100000 :
            #    print(shares, price_1, price_2)
            need_cap = shares*price_2 + self.fix_fee + shares*price_2*self.var_fee + self.borrow_costs

            # ACHTUNG! Wir addieren nie unseren Gewinn direkt zum Capital, da wir eigentlich mehrere Positionen gleichzeitig eingehen,
            # daher müssen zuerst alle positionen eingegangen werden, bevor wir das kapital mit unseren gewinnen/verlusten erhöhen/senken
            return usable_cap - need_cap

        return 0
    
    
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

    def evaluate(self, X):
        """
        :param X: Market Data of asset
        """
        # als erstes teilen wir unser Datenset in die einzelnen Tage ein
        X = X.groupby("date_id")
        keys = list(X.groups.keys())

        # Als zweites Teilen wir das Datenset in train_test_split (80% der tage zu 20% der Tage)
        index = int(len(keys)*0.8)

        hist_sec = []
        hist_d = []


        # Danach gehen wir durch jeden Tag unseres Test sets durch
        for dat in keys[index:]:
            print(dat)


            # hier nehmen wir uns einen Tag heraus
            self.Y = X.get_group(dat)
            self.unique = self.Y["stock_id"].unique()
            print(self.unique)

            # jeden Tag haben wir 54 Zeitshritte von jeweils 10 sekunden
            for i in range(54):

                #print(i)
                # wir suchen uns für Zeitpunkt i nun die am meisten versprechenden Stock_ids
                stocks = self.max(i)

                #print(stocks)

                #für jede stock_id berechnen wir uns den profit
                profit = 0
                #print(stocks)
                for stock in stocks:
                    
                    # Hier nehmen wir uns an dem Tag nur einträge für den einen stock (der Frame ist automatisch geordnet)
                    Z = self.Y[self.Y["stock_id"] == stock]
                    # wap_i+1 den wir vorhergesagen
                    wap_2 = Z.iloc[i+1]["predicted_wap"]
                    # wap_i den wir wissen
                    wap_1 = Z.iloc[i]["predicted_wap"]

                    #wir gehen jetz davon aus das wir dann schon zum Zeitpunkt i+1 sind und unseren verkauspreis kennen
                    # price_i
                    price_1 = Z.iloc[i]["reference_price"]
                    # price_i+1
                    price_2 = Z.iloc[i+1]["reference_price"]

                    # jetzt evaluieren wir unseren Trade
                    profit += self.check_status(price_1, price_2, wap_1, wap_2)
                
                self.capital += profit
                # nachdem wir unsere Trades für den Zeitpunkt abgeschlossen haben, speichern wir unseren Vermögensstand in der historie
                hist_sec.append(self.capital)
            
            # tägliche historie 
            hist_d.append(self.capital)
            
            #hist.append(self.check_status(Y.iloc[i]["seconds_in_bucket"], Y.iloc[i]["reference_price"], 0, Y.iloc[-1]["predicted_wap"], exit=True))

        # jetzt können wir unseren return, volatilität und sharpe ration berechnen
        ret = (hist_sec[-1] - hist_sec[0])/hist_sec[0]

        hist_sec = pd.Series(hist_sec)
        hist_d = pd.Series(hist_d)


        vol = hist_sec.pct_change().dropna().std()
        sharpe = self.sharpe_ratio(self.annualize_rets(hist_d), self.annualize_vol(hist_d))
        print("vol", vol, "sharpe", sharpe, "ann_ret:", self.annualize_rets(hist_d), "return", ret, "capital", self.capital)


        
        #return [sharpe, ret, vol]