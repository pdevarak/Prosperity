def calculate_ideal_size(edge, min_edge, min_size, max_size,slope):
    if edge <= min_edge:
        return 0
    else:
        return min(min_size + slope * (edge - min_edge),max_size)
    

def take_liquidity(theo,limit,bid1_price,bid2_price,bid3_price,bid1_size,bid2_size,bid3_size,ask1_price,ask2_price,ask3_price,ask1_size,ask2_size,ask3_size,min_edge,min_size,max_size,slope,position,MCR_multiplier):
    if (theo - ask1_price >= min_edge):
        edge = theo - ask1_price
        size = calculate_ideal_size(edge,min_edge,min_size,max_size,slope)
        size1 = min(min(size,limit-position),ask1_size)
        position+=size1
        theo-=size1*MCR_multiplier*min_edge/limit
        if (size1 == ask1_size):
            if (theo - ask2_price >= min_edge):
                edge = theo - ask1_price
                size = calculate_ideal_size(edge,min_edge,min_size,max_size,slope)
                size2 = min(min(size,limit-position),ask2_size)
                position+=size2
                theo-=size2*MCR_multiplier*min_edge/limit
                if (size2 == ask2_size):
                    if (theo - ask3_price >= min_edge):
                        edge = theo - ask3_price
                        size = calculate_ideal_size(edge,min_edge,min_size,max_size,slope)
                        size3 = min(min(size,limit-position),ask3_size)
                        position+=size3
                        theo-=size3*MCR_multiplier*min_edge/limit
    elif (bid1_price - theo >= min_edge):
        edge = bid1_price - theo
        size = calculate_ideal_size(edge,min_edge,min_size,max_size,slope)
        size1 = min(min(size,position+limit),bid1_size)
        position-=size1
        theo+=size2*MCR_multiplier*min_edge/limit
        if (size1 == bid1_size):
            if (bid2_price -theo >= min_edge):
                edge = bid2_price - theo
                size = calculate_ideal_size(edge,min_edge,min_size,max_size,slope)
                size2 = min(min(size,limit+position),bid2_size)
                position-=size2
                theo+=size2*MCR_multiplier*min_edge/limit
                if (size2 == bid2_size):
                    if (bid3_price - theo >= min_edge):
                        edge = bid3_price - theo
                        size = calculate_ideal_size(edge,min_edge,min_size,max_size,slope)
                        size3 = min(min(size,limit-position),bid3_size)
                        position-=size3
                        theo+=size3*MCR_multiplier*min_edge/limit

def backtest_market_making_strategy(df, capital, leverage, participation_rate, min_edge, min_size, max_size, slope,MCR_multiplier, trading_fee = 0.0015):
    df['Theo'] = 0.0  # Theoretical price
    df['mcr'] = 0.0  # Market Change Rate
    df['inv'] = 0.0  # Inventory
    df['inv_price'] = 0.0 # How much we spent or received to acqure this position
    df['edge_collected'] = 0.0
    df['edge_wo_mcr'] = 0.0
    df['trade_size'] = 0.0 # Wether or not we traded, and the side if we did
    df['ideal_size'] = 0.0
    df['PNL'] = 0.0  # Profit and Loss
    df['total_size_traded'] = 0.0
    
    inventory = 0.0
    inventory_price = 0.0
    edge_collected = 0.0
    edge_wo_mcr = 0.0
    tot = 0.0
    for index, row in df.iterrows():
        mcr = (inventory / (capital/row['price'])) * MCR_multiplier * min_edge
        theo = row['MKT_Theo'] + (row['MKT_Theo'] * (row['y_hat']/100.0)) - mcr
        df.at[index, 'mcr'] = mcr
        df.at[index, 'Theo'] = theo
        trade_size = abs(row['size'])
        if row['size'] > 0.0:  # BUY trade
            if row['price'] >= theo + min_edge:
                edge = row['price'] - theo
                ideal_size = calculate_ideal_size(edge, min_edge, min_size, max_size, slope)
                df.at[index,'ideal_size'] = 0 - ideal_size
                size = min(ideal_size, participation_rate * trade_size)
                tot+=size*leverage
                edge_collected+= size*(edge - row['price']*trading_fee)*leverage
                edge_wo_mcr+= size*(edge-mcr-row['price']*trading_fee)*leverage
                inventory -= size*leverage
                inventory_price -= size*row['price']*(1-trading_fee)*leverage
                df.at[index,'trade_size'] = 0.0 - leverage*size
        elif row['size'] < 0.0:  # SELL trade
            if row['price'] <= theo - min_edge:
                edge = theo - row['price']
                ideal_size = calculate_ideal_size(edge, min_edge, min_size, max_size, slope)
                df.at[index,'ideal_size'] = ideal_size
                size = min(ideal_size, participation_rate * trade_size)
                tot+=size*leverage
                edge_wo_mcr+= size*(edge + mcr-row['price']*trading_fee)*leverage
                edge_collected+= size*(edge-row['price']*trading_fee)*leverage
                inventory += size*leverage
                inventory_price += size*row['price']*(1+trading_fee)*leverage
                df.at[index,'trade_size'] = size*leverage
        df.at[index,'total_size_traded'] = tot
        df.at[index,'edge_collected'] = edge_collected
        df.at[index,'edge_wo_mcr'] = edge_wo_mcr
        df.at[index, 'inv'] = inventory
        df.at[index, 'inv_price'] = inventory_price
    df['PNL'] = df['inv']*df['price'] - df['inv_price']
    df['PctEdgeRetained'] = df['PNL']/df['edge_collected']
    df['PctEdgeRetained'].fillna(0.0,inplace = True)
    df['TradingCosts'] = (df['price']*np.abs(df['trade_size'])*trading_fee).cumsum()
    df['PNL_wo_fees'] = df['PNL'] + df['TradingCosts']
    
    return df
