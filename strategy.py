import pandas as pd
import numpy as np
# global price_data,volume_data,trade_actions_df, initial_cash,num_stocks,buy_period,recent_days,sell_WM_threshold, rebuy_WM_threshold, sell_RATIO_threshold, rebuy_RATIO_threshold, sell_charges, buy_charges
trade_actions_df = pd.read_excel('TRADE_ACTIONS.xlsx')

def clean_data(df, threshold=0.8):
    min_required = int(threshold * len(df))
    df_cleaned = df.dropna(axis=1, thresh=min_required)
    df_cleaned = df_cleaned.interpolate(method='time')
    return df_cleaned

def determine_current_state(WM, Ratio, Previous_State, sell_WM_threshold, rebuy_WM_threshold, sell_RATIO_threshold, rebuy_RATIO_threshold):
    # global sell_WM_threshold, rebuy_WM_threshold, sell_RATIO_threshold, rebuy_RATIO_threshold
    if WM < sell_WM_threshold:
        wm_category = "WM<sell_WM_threshold"
    elif sell_WM_threshold <= WM < rebuy_WM_threshold:
        wm_category = 'sell_WM_threshold<WM<rebuy_WM_threshold'
    elif WM >= rebuy_WM_threshold:
        wm_category = "WM>rebuy_WM_threshold"
    else:
        return None
    if Ratio < sell_RATIO_threshold:
        ratio_category = "RATIO<sell_RATIO_threshold"
    elif sell_RATIO_threshold <= Ratio < rebuy_RATIO_threshold:
        ratio_category = "sell_RATIO_threshold<RATIO<rebuy_RATIO_threshold"
    elif Ratio >= rebuy_RATIO_threshold:
        ratio_category = "RATIO>rebuy_RATIO_threshold"
    else:
        return None
    filtered_df = trade_actions_df[
        (trade_actions_df[wm_category] == 1) &
        (trade_actions_df[ratio_category] == 1) &
        ((trade_actions_df["PREVIOUS STATE"] == Previous_State) | trade_actions_df['PREVIOUS STATE'].isna())]
    if not filtered_df.empty:
        return filtered_df["CURRENT STATE"].values[0]
    else:
        return "No Matching State Found"

def first_day_trade(price_data,volume_data, initial_cash,num_stocks,
                    buy_period,recent_days, 
                    rebuy_RATIO_threshold, sell_charges, buy_charges):
    # global initial_cash,num_stocks,buy_period,recent_days,sell_WM_threshold, rebuy_WM_threshold, sell_RATIO_threshold, rebuy_RATIO_threshold, sell_charges, buy_charges
    p1 = price_data[(price_data['Date'] <= pd.to_datetime('29-06-2015'))].tail(buy_period)
    p1.set_index('Date', inplace=True)
    p1 = clean_data(p1)
    p1.reset_index(inplace=True)
    x1 = p1.melt(
        id_vars='Date',
        var_name='Stock',
        value_name='stock_price'
        )
    x1['yearly_momentum'] = x1.groupby('Stock')['stock_price'].transform(lambda x: x.pct_change().rolling(window=buy_period-1).mean())
    x1['weekly_momentum'] = x1.groupby('Stock')['stock_price'].transform(lambda x: x.pct_change().rolling(window=6).mean())
    x2 = x1.dropna().sort_values('yearly_momentum', ascending=False).head(num_stocks)
    v1 = volume_data[(volume_data['Date'] <= pd.to_datetime('29-06-2015'))].tail(recent_days)
    v1.dropna(axis=1, inplace=True)
    y1 = v1.melt(
        id_vars='Date',
        var_name='Stock',
        value_name='volume'
        )
    y1['avg_volume'] = y1.groupby('Stock')['volume'].transform(lambda x: x.rolling(window=recent_days).mean())
    y2 = y1.dropna()
    z = y2.merge(x2, on=['Date','Stock'])
    z = z[['Date', 'Stock', 'stock_price','yearly_momentum', 'weekly_momentum','volume', 'avg_volume']]
    z['max_buyable_amount'] = z['avg_volume']/rebuy_RATIO_threshold
    z['cash_allocated_per_stock'] = np.nan
    z['ratio_for_selling'] = np.nan
    z['previous_state_before_selling'] = None
    z['current_state_for_selling'] = None
    z['selling_date'] = None
    z['selling_price'] = np.nan
    z['reason_for_selling'] = None
    z['quantity_to_sell'] = np.nan
    z['transaction_cost_selling'] = z['selling_price'] * z['quantity_to_sell'] * sell_charges
    z['total_transaction_cost_sellling'] = sum(z['transaction_cost_selling'])
    z['cash_held_after_selling'] = z['selling_price'] * z['quantity_to_sell'] * (1-sell_charges)
    z['total_cash_held_after_all_selling'] = sum(z['cash_held_after_selling'])
    z['cash_allocated_per_opening'] = initial_cash/15
    z['ratio_for_buying'] = z['avg_volume']/z['cash_allocated_per_opening']
    z['current_state_for_buying'] = z.apply(
        lambda row: determine_current_state(row['weekly_momentum'], row['ratio_for_buying'], row['current_state_for_selling']), 
        axis=1)
    z['buying_date'] = z.apply(lambda row: row['Date'] if row['current_state_for_buying'] in ['BUY','BUY_10','REBUY','REBUY_10'] 
                               else None, axis=1)
    z['buying_price'] = z.apply(lambda row: row['stock_price'] 
                                if row['current_state_for_buying'] in ['BUY','BUY_10','REBUY','REBUY_10'] 
                                else np.nan, axis=1)
    z['quantity_bought'] = z.apply(
        lambda row: min(row['max_buyable_amount'],row['cash_allocated_per_opening'])*(1-buy_charges)//row['buying_price'], 
        axis=1)
    z['transaction_cost_buying'] = z.apply(
        lambda row: row['quantity_bought']*row['buying_price']*buy_charges
        if row['current_state_for_buying'] in ['BUY','BUY_10','REBUY','REBUY_10'] 
        else 0, axis=1)
    z['cash_held_in_stock_after_buying'] = z.apply(
        lambda row: row['cash_allocated_per_opening'] - row['quantity_bought']*row['buying_price'] - row['transaction_cost_buying']
        if row['current_state_for_buying'] in ['BUY','BUY_10','REBUY','REBUY_10'] 
        else row['cash_allocated_per_opening'], axis=1)
    z.sort_values('Stock').head()
    z['daily_return'] = 0
    z['profit/loss'] = 0
    z['current_investment'] = z.apply(lambda row: row['quantity_bought']*row['stock_price'], axis=1)
    z['total_investment_cost'] = sum(z[~z['current_investment'].isna()]['current_investment'])
    z['total_cash_held_after_buying'] = sum(z['cash_held_in_stock_after_buying'])
    z['total_portfolio_value'] = z['total_investment_cost']+z['total_cash_held_after_buying']
    z.sort_values('Stock',inplace=True)
    z.reset_index(drop=True, inplace=True)
    return z

def today_trade(previous_day, today_date,  price_data, volume_data,
                num_stocks,buy_period,recent_days,
                rebuy_RATIO_threshold, buy_charges, sell_charges):
    # global price_data, volume_data, initial_cash,num_stocks,buy_period,recent_days,sell_WM_threshold, rebuy_WM_threshold, sell_RATIO_threshold, rebuy_RATIO_threshold, sell_charges, buy_charges
    sell_date = today_date
    p1 = price_data[(price_data['Date'] <= sell_date)].tail(buy_period)
    p1.set_index('Date', inplace=True)
    p1 = clean_data(p1)
    p1.reset_index(inplace=True)
    x1 = p1.melt(
        id_vars='Date',
        var_name='Stock',
        value_name='stock_price')
    x1['yearly_momentum'] = x1.groupby('Stock')['stock_price'].transform(
        lambda x: x.pct_change().rolling(window=(buy_period-1 - x.isna().sum())).mean())
    x1['weekly_momentum'] = x1.groupby('Stock')['stock_price'].transform(
        lambda x: x.pct_change().rolling(window=6, min_periods=5).mean())
    x2 = x1.dropna().sort_values('yearly_momentum', ascending=False).head(num_stocks)
    v1 = volume_data[(volume_data['Date'] <= pd.to_datetime(sell_date))].tail(recent_days)
    v1.dropna(axis=1, inplace=True)
    y1 = v1.melt(
        id_vars='Date',
        var_name='Stock',
        value_name='volume')
    y1['avg_volume'] = y1.groupby('Stock')['volume'].transform(lambda x: x.rolling(window=recent_days).mean())
    y2 = y1.dropna()
    z = y2.merge(x2, on=['Date','Stock'])
    z = z[['Date', 'Stock', 'stock_price','yearly_momentum', 'weekly_momentum','volume', 'avg_volume']]
    z['max_buyable_amount'] = z['avg_volume']/rebuy_RATIO_threshold
    z['cash_allocated_per_stock'] = 0
    z['ratio_for_selling'] = np.nan
    z['previous_state_before_selling'] = None
    z['current_state_for_selling'] = None
    z['selling_date'] = None
    z['selling_price'] = np.nan
    z['reason_for_selling'] = None
    z['quantity_to_sell'] = z.merge(previous_day[['Stock','quantity_bought']], on='Stock', how='left')['quantity_bought']
    z['cash_allocated_per_stock'] = z['quantity_to_sell'] * z['stock_price']
    z['ratio_for_selling'] = z.apply(
        lambda row: row['avg_volume']/row['cash_allocated_per_stock']
        if row['cash_allocated_per_stock'] != 0
        else np.nan, axis=1)
    z['previous_state_before_selling'] = z.merge(
        previous_day[['Stock','current_state_for_buying']], on='Stock', how='left')['current_state_for_buying']
    z['current_state_for_selling'] = z.apply(
        lambda row: determine_current_state(row['weekly_momentum'], row['ratio_for_selling'], row['previous_state_before_selling']) 
        if row['previous_state_before_selling'] in ['BUY','BUY_10','REBUY','REBUY_10','HOLD']
        else row['previous_state_before_selling'], axis=1)
    z['selling_date'] = z.apply(
        lambda row: row['Date'] 
        if isinstance(row['current_state_for_selling'], str) and 'sell' in row['current_state_for_selling'].lower() 
        else None, axis=1)
    z['selling_price'] = z.apply(
        lambda row: row['stock_price'] 
        if isinstance(row['current_state_for_selling'], str) and 'sell' in row['current_state_for_selling'].lower() 
        else None, axis=1)
    z['reason_for_selling'] = z['current_state_for_selling'].apply(
        lambda x: 'weekly_momentum_drop' 
        if x=='CASH/SELL_WEEKLY' 
        else 'ratio_drop' 
        if x=='CASH/SELL_VOLUME' 
        else 'weekly_volume_drop' 
        if x=='CASH/SELL_WEEKLY_VOLUME' 
        else None)
    z['transaction_cost_selling'] = z['selling_price'] * z['quantity_to_sell'] * sell_charges
    z['total_transaction_cost_sellling'] = sum(z['transaction_cost_selling'])
    z['cash_held_after_selling'] = z['selling_price'] * z['quantity_to_sell'] * (1-sell_charges)
    stocks_to_sell_today = list(previous_day[~previous_day['Stock'].isin(z['Stock'])]['Stock'])
    total_cash_held_YM_selling = 0
    for stock in stocks_to_sell_today:
        previous_day.loc[previous_day['Stock'] == stock, 'current_state_for_selling'] = 'SELL'
        previous_day.loc[previous_day['Stock'] == stock, 'selling_date'] = sell_date
        previous_day.loc[previous_day['Stock'] == stock, 'selling_price'] = p1.loc[p1['Date'] == sell_date, stock].values[0]
        previous_day.loc[previous_day['Stock'] == stock, 'reason_for_selling'] = 'yearly_momentum_drop'
        if pd.isna(previous_day.loc[previous_day['Stock'] == stock, 'quantity_bought'].iloc[0]):
            previous_day.loc[previous_day['Stock'] == stock, 'quantity_bought'] = 0
        cash_after_selling = previous_day.loc[
        previous_day['Stock'] == stock, 'selling_price'
        ].iloc[0] * previous_day.loc[
        previous_day['Stock'] == stock, 'quantity_bought'
        ].iloc[0] * (1-sell_charges)
        total_cash_held_YM_selling += cash_after_selling
    z['total_cash_held_after_all_selling'] = sum(z[~z['cash_held_after_selling'].isna()]['cash_held_after_selling']) + total_cash_held_YM_selling
    number_of_all_openings = sum(~z['current_state_for_selling'].isin(['BUY','BUY_10','REBUY','REBUY_10','HOLD']))
    total_cash_held_till_yesterday = previous_day['total_cash_held_after_buying'].iloc[0]
    if not z.empty and 'total_cash_held_after_all_selling' in z.columns:
        cash_allocated_per_opening = (z['total_cash_held_after_all_selling'].iloc[0] + total_cash_held_till_yesterday) / number_of_all_openings
    else:
        cash_allocated_per_opening = 0 
    z['cash_allocated_per_opening'] = z.apply(
        lambda row: cash_allocated_per_opening 
        if (pd.isna(row['current_state_for_selling']) or 
            (isinstance(row['current_state_for_selling'], str) and 
             'cash' in row['current_state_for_selling'].lower()))
        else 0, axis=1)
    
    z['ratio_for_buying'] = z.apply(
        lambda row: row['avg_volume']/row['cash_allocated_per_opening'] 
        if row['cash_allocated_per_opening']!=0 
        else np.nan, axis=1)
    
    z['current_state_for_buying'] = z.apply(
        lambda row: determine_current_state(row['weekly_momentum'], row['ratio_for_buying'], row['current_state_for_selling']) 
        if row['current_state_for_selling'] in ['CASH_WEEKLY','CASH_VOLUME','CASH_WEEKLY_VOLUME',
                                                'CASH/SELL_WEEKLY','CASH/SELL_VOLUME','CASH/SELL_WEEKLY_VOLUME',
                                                np.nan] 
        else row['current_state_for_selling'], axis=1)
    
    z['buying_date'] = z.apply(
        lambda row: row['Date'] 
        if row['current_state_for_buying'] in ['BUY', 'BUY_10', 'REBUY', 'REBUY_10']
        else previous_day.loc[previous_day['Stock'] == row['Stock'], 'Date'].iloc[0] 
        if not previous_day.loc[previous_day['Stock'] == row['Stock'], 'Date'].empty and 
        row['current_state_for_buying'] not in ['CASH_WEEKLY','CASH_VOLUME','CASH_WEEKLY_VOLUME']
        else None, axis=1)
    
    z['buying_price'] = z.apply(
        lambda row: row['stock_price'] 
        if row['current_state_for_buying'] in ['BUY', 'BUY_10', 'REBUY', 'REBUY_10']
        else previous_day.loc[previous_day['Stock'] == row['Stock'], 'buying_price'].iloc[0]
        if not previous_day.loc[previous_day['Stock'] == row['Stock'], 'buying_price'].empty and 
        row['current_state_for_buying'] not in ['CASH_WEEKLY','CASH_VOLUME','CASH_WEEKLY_VOLUME']
        else None, axis=1)
    
    z['quantity_bought'] = z.apply(
        lambda row: min(row['max_buyable_amount'],row['cash_allocated_per_opening'])*(1-buy_charges)//row['buying_price'] 
        if row['current_state_for_buying'] in ['BUY', 'BUY_10', 'REBUY', 'REBUY_10']
        else previous_day.loc[previous_day['Stock'] == row['Stock'], 'quantity_bought'].iloc[0] 
        if not previous_day.loc[previous_day['Stock'] == row['Stock'], 'quantity_bought'].empty and 
        row['current_state_for_buying'] not in ['CASH_WEEKLY',
                                                'CASH_VOLUME',
                                                'CASH_WEEKLY_VOLUME',
                                                'CASH/SELL_WEEKLY',
                                                'CASH/SELL_VOLUME',
                                                'CASH/SELL_WEEKLY_VOLUME']
        
        else 0, axis=1)
    
    z['transaction_cost_buying'] = z.apply(
        lambda row: row['quantity_bought']*row['buying_price']*buy_charges
        if row['current_state_for_buying'] in ['BUY','BUY_10','REBUY','REBUY_10'] 
        else 0, axis=1)
    
    z['cash_held_in_stock_after_buying'] = z.apply(
        lambda row: row['cash_allocated_per_opening'] - row['quantity_bought']*row['buying_price'] - row['transaction_cost_buying']
        if row['current_state_for_buying'] in ['BUY','BUY_10','REBUY','REBUY_10'] 
        else cash_allocated_per_opening 
        if row['current_state_for_buying'] in ['CASH/SELL_WEEKLY','CASH/SELL_VOLUME','CASH/SELL_WEEKLY_VOLUME'] 
        else row['cash_allocated_per_opening'], axis=1)
    
    z.sort_values('Stock').head()
    z['daily_return'] = 0
    z['profit/loss'] = z.apply(
        lambda row: row['quantity_bought']*(row['stock_price']-row['buying_price']) 
        if row['current_state_for_buying']=='HOLD' 
        else np.nan, axis=1)
    
    z['current_investment'] = z.apply(
        lambda row: row['quantity_bought']*row['stock_price'], axis=1)
    
    z['total_investment_cost'] = sum(z[~z['current_investment'].isna()]['current_investment'])
    z['total_cash_held_after_buying'] = sum(z['cash_held_in_stock_after_buying'])
    z['total_portfolio_value'] = z['total_investment_cost'] + z['total_cash_held_after_buying']
    z.sort_values(['Date','Stock'],inplace=True)
    z.reset_index(drop=True, inplace=True)
    return z


def backtest(price_data, 
             volume_data, 
             initial_cash,
             num_stocks,
             buy_period,
             recent_days,
             sell_WM_threshold,
             rebuy_WM_threshold,
             sell_RATIO_threshold,
             rebuy_RATIO_threshold,
             sell_charges,
             buy_charges,
             num_of_days_for_backtesting=15, 
             save_backtesting_excel_by_name='backtesting_v1.xlsx'):
    print('Hi')
    backtest = price_data[price_data['Date']>pd.to_datetime('29-06-2015')]['Date'].iloc[:num_of_days_for_backtesting]
    today = first_day_trade(price_data,volume_data,trade_actions_df, initial_cash,num_stocks,buy_period,recent_days,sell_WM_threshold, rebuy_WM_threshold, sell_RATIO_threshold, rebuy_RATIO_threshold, sell_charges, buy_charges)
    HISTORICAL = today.copy()
    print(f"DAY {int(len(HISTORICAL)/num_stocks)} : {'2015-06-29'} -- Portfolio Value: {round(HISTORICAL['total_portfolio_value'].iloc[-1],2)}")
    for date in backtest:
        today = today_trade(HISTORICAL.tail(num_stocks), date)
        HISTORICAL = pd.concat([HISTORICAL,today])
        HISTORICAL.drop_duplicates(subset='Date', keep='last')
        print(f"DAY {int(len(HISTORICAL)/num_stocks)} : {date.date()} -- Portfolio Value: {round(HISTORICAL['total_portfolio_value'].iloc[-1],2)}")
    HISTORICAL.to_excel(save_backtesting_excel_by_name)