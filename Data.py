import pandas as pd

class Data:

    def __init__(self):
        self._prices = None

    @property
    def prices(self):
        if self._prices is None:
            self._prices = self._get_prices()

        return self._prices

    def _get_prices(self):
        currency_pairs = ['AUDUSD', 'EURCHF', 'EURJPY', 'EURUSD', 'USDCAD', 'USDCHF', 'USDJPY']

        data = pd.concat([
            pd.read_csv(f'data/{pair}-2000-2020-15m.csv', parse_dates=['DATE_TIME']).assign(CURRENCY_PAIR=pair)
            for pair in currency_pairs
        ])

        pivoted_data = data.pivot_table(index='DATE_TIME', columns='CURRENCY_PAIR', values='CLOSE', aggfunc='first')
        index = pd.date_range(start=pivoted_data.index.min(), end=pivoted_data.index.max(), freq='15min')
        pivoted_data = pivoted_data.reindex(index)
        pivoted_data.columns = pivoted_data.columns.rename('DATE_TIME')

        # A day with no data is assumed to either be a weekend or a holiday, and so won't be considered as missing data
        all_nan_days = pivoted_data.isna().groupby(pivoted_data.index.to_period('D')).agg('all').any(axis='columns')
        bdays_data = pivoted_data[~pivoted_data.index.to_period('D').isin(all_nan_days[all_nan_days].index)]

        # Return prices for months with no missing data
        is_all_nan_interval = bdays_data.isna().any(axis='columns')
        max_missing_interval = is_all_nan_interval.groupby(is_all_nan_interval.index.to_period('M')).apply(lambda x: x.groupby((~x).cumsum()).sum().max())
        is_month_with_data = max_missing_interval[max_missing_interval == 0].index

        return bdays_data[bdays_data.index.to_period('M').isin(is_month_with_data)]
