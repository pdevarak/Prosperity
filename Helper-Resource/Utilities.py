import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.preprocessing import StandardScaler
import numpy as np
import scipy as sp
from scipy.stats import zscore

import itertools
import IPython.display

import math
from math import floor
import functools
import seaborn as sns
import h5py
import copy
import openpyxl
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error



#from plotnine import ggplot, aes, geom_line, geom_hline, labs, theme, element_text, geom_area, scale_fill_manual, geom_bar

import os
from datetime import datetime
from datetime import datetime
from dateutil.relativedelta import relativedelta

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


from enum import Enum

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def process_and_set_index(df, column_name):
    """
    First checks if rows that would have the same index based on the specified column have approximately 
    the same values across all columns using numpy.isclose. Then, it checks if the specified column in the 
    DataFrame contains only unique values, sorts the DataFrame by that column, and sets the column as the 
    DataFrame's index if it is unique.
    
    Parameters:
    - df: pandas.DataFrame to process
    - column_name: str, the name of the column to check for uniqueness and set as index
    
    Returns:
    - The processed DataFrame with the column set as index if unique, after ensuring rows with potential 
      duplicated indices have approximately the same values across all columns.
    """

    df = df.drop_duplicates(subset=[column_name], keep='last')

    # Check if the column is unique
    is_unique = df[column_name].is_unique
    print(f"Is the {column_name} column unique?", is_unique)
    
    # If the column is not unique, find and print the duplicated values
    if not is_unique:
        duplicated_values = df[df[column_name].duplicated(keep=False)]
        print(f"Duplicated values in the {column_name} column:")
        print(duplicated_values[column_name].unique())

    # Sort the DataFrame by the specified column
    df_sorted = df.sort_values(by=column_name)
    
    # If unique, set the column as the index
    if is_unique:#is_unique:
        df_sorted.set_index(column_name, inplace=True)
        print("Column set as index.")
    else:
        print("Column contains duplicates; not set as index.")
    
    return df_sorted

def process_dataframe_and_adjust_size(df, index_columns):
    """
    Processes the DataFrame by setting a multi-index, sorting by the multi-index, adjusting the 'size' column
    based on the 'side' column, and then removing the 'side' column.
    
    Parameters:
    - df: pandas.DataFrame to process
    - index_columns: list of str, column names to set as multi-index
    
    Returns:
    - The processed DataFrame with a multi-index, sorted by the multi-index, and with adjusted 'size' column.
    """
    # Set the multi-index
    df.set_index(index_columns, inplace=True)

    # Adjust the 'size' column based on the 'side' column
    # Assuming 'side' column contains numeric values that indicate multiplication factor
    df['size'] = df['size'] * np.sign(df['side'])

    # Remove the 'side' column
    df.drop(columns=['side'], inplace=True)

    # Sort the DataFrame by the new multi-index
    df.sort_index(inplace=True)
    
    return df

def check_dataframe_integrity(df, set_0 = False, display_df = False):
    """
    Performs integrity checks on a pandas DataFrame, including checks for excessively large numbers,
    NaN values, and more. Results of checks are printed along with details on excessively large numbers
    and NaN values. Finally, all NaN values are filled with 0.0.

    Parameters:
    - df: pandas.DataFrame to check
    """
    tmp = df.index.is_unique
    print(f"Check for uniqueness:                {tmp}")
    # Flag to indicate if any excessively large numbers were found
    found_excessive_numbers = False
    
    # Check for excessively large numbers and print them
    for column in df.select_dtypes(include=[np.number]).columns:
        excessive_values = df[df[column] > 1e9][column]
        if not excessive_values.empty:
            found_excessive_numbers = True
            print(f"Excessively large numbers found in column '{column}':")
            print(excessive_values.to_string(), "\n")
    
    if not found_excessive_numbers:
        print("Check for excessively large numbers: Passed")
    
    # Initial check for any NaN values
    total_nan_count = df.isnull().sum().sum()
    if total_nan_count > 0:
        print("Check for NaN values: Found. Detailed report follows.")
    else:
        print("Check for NaN values:                Passed")

    # Detailed NaN values report
    for column in df.columns:
        nan_count = df[column].isnull().sum()
        if nan_count > 0:
            percent_nan = (nan_count / len(df)) * 100
            print(f"Column '{column}' has {nan_count} NaN value(s), which is {percent_nan:.2f}% of the rows.")
    
    if set_0:
        # Fill all NaN values with 0.0
        print("Filling all NaN values with 0.0")
        df.fillna(0.0, inplace=True)
    if display_df == True:
        display(df)
    
def split_and_filter_dataframes(book_df, trades_df, name, test_date_start, display_data=False):
    # Split the book DataFrame based on the test_date_start
    book_train = book_df.loc[book_df.index.get_level_values(0) < test_date_start].copy()
    book_test = book_df.loc[book_df.index.get_level_values(0) >= test_date_start].copy()

    # Split the trades DataFrame based on the test_date_start
    trades_train = trades_df.loc[trades_df.index.get_level_values(0) < test_date_start].copy()
    trades_test = trades_df.loc[trades_df.index.get_level_values(0) >= test_date_start].copy()

    # Integrity Checks
    # 1. Index Integrity Check
    assert book_test.index.min() > book_train.index.max(), "Overlap or gap between book test and train datasets."
    assert trades_test.index.get_level_values(0).min() > trades_train.index.get_level_values(0).max(), "Overlap or gap between trades test and train datasets."

    # 2. Data Integrity Check
    total_rows_before_split = len(book_df) + len(trades_df)
    total_rows_after_split = len(book_test) + len(book_train) + len(trades_test) + len(trades_train)
    assert total_rows_before_split == total_rows_after_split, "Data loss in splitting process."

    if display_data:
        # Display test and train splits for both book and trades
        print(f"{name} Book Train Dataset:")
        display(book_train)
        print(f"{name} Book Test Dataset:")
        display(book_test)
        print(f"{name} Trades Train Dataset:")
        display(trades_train)
        print(f"{name} Trades Test Dataset:")
        display(trades_test)
        
        # Display integrity check results
        print("Integrity checks passed. No data loss, overlaps, or gaps.")
        print()

    return book_test, book_train, trades_test, trades_train

def construct_size_df(book_df, trade_df=15000):
    # Perform a full outer join on the two DataFrames using their indices
    size_df = book_df.join(trade_df['size'], how='outer', rsuffix='_trade')
    
    # Reduce size_df to only include the 'size' column
    size_df = size_df[['size']]
    
    return size_df

def Calculate_Trade_Flow(trade_data, flow_halflife = 15000):
    """
    Adds a 'flow' column to the input DataFrame, calculated as the exponentially weighted moving sum
    of the 'size' column. This version of the function adjusts for uneven time intervals between observations
    using a dynamic decay based on a half-life specified in milliseconds. The 'halflife' parameter provides an
    intuitive measure for the decay, indicating the time it takes for the influence of an observation to reduce
    to half its initial value.

    The 'times' parameter in the ewm method leverages actual timestamps, allowing for the exponential weighting
    to dynamically adjust based on the real-time passage between observations. This method is particularly useful
    for time series data that features irregular intervals between data points.

    Parameters:
    - trade_data: DataFrame containing trade data, expected to have a datetime index and a 'size' column.
    - trade_halflife: The half-life period, specified as a number of milliseconds, over which the weight
                             of an observation reduces to half its original value.

    Returns:
    - The DataFrame with an added 'flow' column, showing the dynamically adjusted exponentially weighted moving
      sum of trade sizes based on the specified 'halflife'.
    """
    # Ensure the DataFrame's index is in datetime format for time-based calculation
    trade_data.index = pd.to_datetime(trade_data.index)
    
    # Convert halflife from milliseconds to a pandas-compatible string format
    halflife_str = f'{flow_halflife}ms'
    
    # Use the datetime index as the 'times' argument and apply the 'halflife' parameter for the decay rate
    trade_data['flow'] = trade_data['size'].ewm(halflife=halflife_str, ignore_na=True, times=trade_data.index).mean()

    trade_data['flow'].fillna(0, inplace=True)

    return trade_data

def calc_abs_volume(df, volume_halflife = 15000):
    """
    Adds a 'volume' column to the DataFrame, calculated as the exponentially weighted moving average
    of the absolute values of the 'size' column. This implementation uses the 'halflife' parameter,
    specified in milliseconds, to define the decay rate, allowing for dynamic adjustment based on the 
    actual timing of observations. The half-life specifies the time period over which the influence of
    an observation reduces to half its original value, providing an intuitive measure for handling
    temporal decay in time series data.

    Parameters:
    - df: DataFrame containing trade data. Expected to have a datetime index and a 'size' column.
    - halflife_milliseconds: The period over which the weight of an observation in the calculation reduces
                             to half its original value, specified in milliseconds as a numeric value.

    Returns:
    - The DataFrame with an added 'volume' column, representing the dynamically adjusted exponentially 
      weighted moving average of trade sizes, using the 'halflife' parameter for decay.
    """
    # Ensure the DataFrame's index is in datetime format for time-based calculation
    df.index = pd.to_datetime(df.index)
    
    # Convert halflife from milliseconds to a pandas-compatible string format
    halflife_str = f'{volume_halflife}ms'
    
    # Apply the ewm method with the 'halflife' parameter for dynamically adjusted decay based on timing,
    # using the absolute values of the 'size' column. 'adjust=False' is used to emphasize the relevance 
    # of more recent observations without normalizing weights, and 'ignore_na=True' to handle missing values
    # by ignoring them in the calculation while still considering the timing of observations.
    df['volume'] = df['size'].abs().ewm(halflife=halflife_str, ignore_na=True, times=df.index).mean()

    df['volume'].fillna(0, inplace=True)

    return df



def add_volume_and_flow_to_book(book_df, size_df):
    """
    Adds 'volume' and 'flow' columns from the size DataFrame to the corresponding book DataFrame.

    Parameters:
    - book_df: DataFrame representing the book data.
    - size_df: DataFrame containing 'size', 'volume', and 'flow' columns.
    
    Returns:
    - The book DataFrame with added 'volume' and 'flow' columns from the size DataFrame.
    """

    # Remove duplicates in size_df, keeping only the last entry for each index
    filtered_size_df = size_df[~size_df.index.duplicated(keep='last')]

    # Select only the 'volume' and 'flow' columns to be added to book_df
    columns_to_add = filtered_size_df[['volume', 'flow']]

    # Join the selected columns with book_df based on index
    updated_book_df = book_df.join(columns_to_add, how='left')
    
    return updated_book_df

def add_cross_volume_and_flow_to_book(book_df, size_df):
    """
    Adds 'cross_volume' and 'cross_flow' columns from the size DataFrame to the corresponding book DataFrame.

    Parameters:
    - book_df: DataFrame representing the book data.
    - size_df: DataFrame containing 'size', 'volume', and 'flow' columns.
    
    Returns:
    - The book DataFrame with added 'volume' and 'flow' columns from the size DataFrame.
    """

    size_df.rename(columns={
        'volume': 'cross_volume',
        'flow': 'cross_flow',
        # Add more columns as needed
    }, inplace=True)

    # Remove duplicates in size_df, keeping only the last entry for each index
    filtered_size_df = size_df[~size_df.index.duplicated(keep='last')]

    # Select only the 'volume' and 'flow' columns to be added to book_df
    columns_to_add = filtered_size_df[['cross_volume', 'cross_flow']]

    # Join the selected columns with book_df based on index
    updated_book_df = book_df.join(columns_to_add, how='left')

    return updated_book_df

def plot_acf_pacf(dataframe, pair_name, column='mid', lags=20):
    """
    Plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)
    for the specified column of a dataframe, including a title with the pair name.

    Parameters:
    - dataframe: pandas DataFrame containing the data.
    - pair_name: str, the name of the currency pair to include in the title.
    - column: str, the column for which to plot ACF and PACF. Defaults to 'mid'.
    - lags: int, the number of lags to include in the plots.
    """
    # Ensure the column exists in the dataframe to avoid KeyError
    if column not in dataframe.columns:
        print(f"Column '{column}' not found in the dataframe.")
        return
    
    # Extract the column data
    series = dataframe[column]

    # Plotting ACF
    plt.figure(figsize=(10, 4))
    plt.subplot(121) # 1 row, 2 columns, 1st subplot
    plot_acf(series, lags=lags, ax=plt.gca(), title=f'Autocorrelation Function - {pair_name}')
    
    # Plotting PACF
    plt.subplot(122) # 1 row, 2 columns, 2nd subplot
    plot_pacf(series, lags=lags, ax=plt.gca(), title=f'Partial Autocorrelation Function - {pair_name}')
    
    plt.suptitle(f'ACF and PACF for {pair_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def calculate_time_deltas(book_raw, trades_raw, pair_name):
    print()
    print(pair_name + ":")
    
    # Book - Raw Data
    time_deltas = book_raw['timestamp_utc_nanoseconds'].diff()
    print(f"Book: Average time delta: {time_deltas.mean()}")
    print(f"Book: Median time delta: {time_deltas.median()}")

    # Trades - Raw Data
    time_deltas = trades_raw['timestamp_utc_nanoseconds'].diff()
    print(f"Trades: Average time delta: {time_deltas.mean()}")
    print(f"Trades: Median time delta: {time_deltas.median()}")

    # Book - Unique Timestamps
    timestamps = pd.Series(book_raw['timestamp_utc_nanoseconds'].unique())
    timestamps = pd.to_datetime(timestamps, unit='ns')
    time_deltas = timestamps.diff()
    print(f"Book: Average time delta between unique instances: {time_deltas.mean()}")
    print(f"Book: Median time delta between unique instances: {time_deltas.median()}")

    # Trades - Unique Timestamps
    timestamps = pd.Series(trades_raw['timestamp_utc_nanoseconds'].unique())
    timestamps = pd.to_datetime(timestamps, unit='ns')
    time_deltas = timestamps.diff()
    print(f"Trades: Average time delta between unique instances: {time_deltas.mean()}")
    print(f"Trades: Median time delta between unique instances: {time_deltas.median()}")

def reindex_dataframe_to_uniform_intervals(dataframe, interval_milliseconds):
    """
    Reindexes a dataframe to have uniform time intervals in milliseconds.

    Parameters:
    - dataframe: pd.DataFrame, the dataframe with a DateTime index.
    - interval_milliseconds: int, the desired time interval in milliseconds.

    Returns:
    - A new dataframe reindexed to the specified uniform time intervals in milliseconds.
    """
    # Convert interval_milliseconds to a Pandas frequency string
    freq = f'{interval_milliseconds}ms'

    # Ensure the dataframe's index is a DateTimeIndex
    if not isinstance(dataframe.index, pd.DatetimeIndex):
        raise ValueError("Dataframe index must be a pd.DatetimeIndex")

    # Make the index unique by taking the last value for each timestamp
    # This step is necessary to ensure we can reindex with a method like 'ffill'
    unique_df = dataframe.groupby(dataframe.index).last()

    # Creating a new uniform time index starting from the minimum to the maximum existing index
    start, end = unique_df.index.min(), unique_df.index.max()
    new_index = pd.date_range(start=start, end=end, freq=freq)
    
    reindexed_df = unique_df.reindex(new_index, method='ffill')

    return reindexed_df

def percent_change(df, column_name):
    df[column_name+"_%"] = df[column_name].pct_change()
    df.fillna(0,inplace = True)
    return df



def add_lags_and_dep_var(df, column_name, num_lags):
    """
    Adds a dependent variable column 'y' which is the next value of the specified column,
    and a specified number of lagged columns for the given column in the DataFrame.
    
    Parameters:
    - df: pandas DataFrame
    - column_name: the name of the column to create lags for
    - num_lags: the number of lagged columns to include
    
    Returns:
    - The DataFrame with the added 'y' and lagged columns
    """
    # Adding the dependent variable column 'y'
    df['y'] = df[column_name].shift(-1)
    if num_lags != 0:
        # Adding the specified number of lagged columns
        for lag in range(1, num_lags + 1):
            lag_col_name = f"{column_name}-{lag}"
            df[lag_col_name] = df[column_name].shift(lag)
    df.fillna(0,inplace = True)
    return df

def normalize(df, column_names):
    """
    Apply Z-score normalization on specified columns of the DataFrame.

    Parameters:
    - df: pandas DataFrame
    - column_names: list of column names to be normalized

    Returns:
    - DataFrame with specified columns normalized
    """
    df['one'] = 1.0
    for col in column_names:
        mean = df[col].cumsum()/(df['one'].cumsum())
        std_dev = ((df[col]**2).cumsum()/(df['one'].cumsum()) - mean**2)**0.5
        df[col] = (df[col] - mean)/std_dev
    df = df.drop('one',axis = 1)
    df.dropna(inplace=True)
    return df

def rolling_normalize(df, column_names,window_size):
    """
    Apply Z-score normalization on specified columns of the DataFrame.

    Parameters:
    - df: pandas DataFrame
    - column_names: list of column names to be normalized

    Returns:
    - DataFrame with specified columns normalized
    """
    if len(column_names) == 0:
        return df
    
    rolling_mean = df[column_names].rolling(window=window_size).mean()
    rolling_std = df[column_names].rolling(window=window_size).std()

    # Calculate the Z-score for each data point using the rolling mean and rolling standard deviation
    z_score = (df[column_names] - rolling_mean) / rolling_std
    df[column_names] = z_score
    return df

def calc_beta(df, df_name, print_results = False):
    # Assuming 'y' is the dependent variable and all others are independent
    X = df.drop('y', axis=1)  # Independent variables
    y = df['y']  # Dependent variable
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    model = sm.OLS(y, X).fit()
    if print_results:
        # Print header and summary for in-sample results
        print(f"Regression Results for {df_name}")
        print(model.summary())
    
    return model  # Return the model object for further use

def add_predicted_values_single(dataframe, betas, y_mean, y_std):
    """
    Adds and unnormalizes predicted values to a single dataframe based on the provided betas.

    Parameters:
    - dataframe: A Pandas DataFrame to add predictions to.
    - betas: A list of regression betas (coefficients) for the dataframe. Assumes first beta is the intercept.
    - y_mean: Mean of the dependent variable `y` before normalization.
    - y_std: Standard deviation of the dependent variable `y` before normalization.

    Modifies the input dataframe in-place by adding a 'y_hat' column with the regression-based predictions,
    then unnormalizes these predictions.
    """
    dataframe = dataframe.dropna()
    intercept = betas[0]
    coefficients = betas[1:]
    
    X = dataframe.drop('y', axis=1)
    # Calculate predicted values
    predicted_values = np.dot(X, coefficients) + intercept
    
    # Unnormalize the predicted values
    unnormalized_predicted_values = (predicted_values * y_std) + y_mean
    
    # Add the unnormalized predicted values to the dataframe
    dataframe['y_hat'] = unnormalized_predicted_values

    return dataframe


def merge_test_with_trades(test_df, trades_df):
    """
    Merges test data frames with a trades data frame based on the closest index values,
    ensuring that the result only contains trade indices. Uses forward fill to handle missing values.

    Parameters:
    - test_df: pd.DataFrame, the test book data frame with a DateTime index or similar.
    - trades_df: pd.DataFrame, the trades data frame with a DateTime index or similar.

    Returns:
    - A new merged DataFrame indexed by the trade indices.
    """
    # Ensure both dataframes are sorted by index to optimize the merge_asof operation
    test_df = test_df.sort_index()
    trades_df = trades_df.sort_index()
    trades_df.index = trades_df.index - pd.Timedelta('1ns')
    
    # Merge the test data frame with the trades data frame using merge_asof
    # This approach allows for matching the closest key in 'test_df' with keys in 'trades_df'
    # and using 'ffill' to forward-fill missing values from 'test_df'
    merged_df = pd.merge_asof(trades_df, test_df, left_index=True, right_index=True, direction='backward')
    merged_df.dropna(inplace =True)

    return merged_df

def calculate_ideal_size(edge, min_edge, min_size, max_size,slope):
    if edge <= min_edge:
        return 0
    else:
        return min(min_size + slope * (edge - min_edge),max_size)

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

def regularize_backtest_data(backtest_data, book_data, ETH_df = None):
    backtest_data['T_pre'] = backtest_data.index

    # Round end time up to the nearest second
    start_time = backtest_data.index.min()#.floor('S')
    end_time = backtest_data.index.max()#.ceil('S')

    # Creating a regular index DataFrame with one-second intervals
    reg_index = pd.date_range(start=start_time, end=end_time, freq='S')
    
    # Initialize reg_df with reg_index
    reg_df = pd.DataFrame(index=reg_index)

    # Remove indices from reg_df that are already in backtest_data
    reg_df = reg_df[~reg_df.index.isin(backtest_data.index)]

    # Adjust the index's on the order book data
    trade_irr_index = backtest_data.index
    book_irr_index = book_data.index
    mid_pre_union = trade_irr_index.unique().union(book_irr_index.union(reg_index))

    #continuous_mid = book_data.reindex(book_irr_index.union(reg_index)).fillna(method  = 'ffill')


    Adj_book_df = book_data.reindex(mid_pre_union).fillna(method  = 'ffill')

    # Concatenate backtest_data with the updated reg_df
    adj_trade_df = pd.concat([backtest_data, reg_df], axis=0).sort_index()

    # Fill 'trade_size' with 0 specifically
    adj_trade_df['trade_size'] = adj_trade_df['trade_size'].fillna(0.0)

    # Forward fill the rest
    adj_trade_df = adj_trade_df.fillna(method='ffill')
    

    #Adj_book_df = Adj_book_df.fillna(method  = 'bfill')
    #adj_trade_df = adj_trade_df.fillna(method = 'bfill')

    T_pre_index = pd.Index(adj_trade_df.loc[reg_index,'T_pre'])

    # calulates the change in PNL between the last trade and the next Regularly spaced time step
    tmp_1 = adj_trade_df.loc[reg_index, 'inv']
    tmp_2 = Adj_book_df.loc[reg_index,"mid"].values
    tmp_3 = Adj_book_df.loc[T_pre_index, 'mid'].values
    PNL_delta = adj_trade_df.loc[reg_index, 'inv'] * (Adj_book_df.loc[reg_index,"mid"].values - Adj_book_df.loc[T_pre_index, 'mid'].values)

    # adjusts PNL at regularly spaced time stamp by adding the PNL from the time inbetween the last trade and the regular time stamp
    adj_trade_df.loc[reg_index, 'PNL'] += PNL_delta.values

    # calculate the final PNL including the last trade closing at the last 'mid' from book_data
    #adj_trade_df['PNL'].iloc[-1] = (adj_trade_df["curr_pos"].iloc[-1] * book_data["mid"].iloc[-1]) - adj_trade_df["pos_cost"].iloc[-1]

    adj_trade_df['mid'] = Adj_book_df.loc[adj_trade_df.index, 'mid']

    # for ETH BTC case convert PNL into dollars at the end
    if ETH_df is not None:
        ETH_df = ETH_df.reindex(reg_index, method = 'ffill')
        display(adj_trade_df)
        adj_trade_df['PNL'] = adj_trade_df['PNL'] * ETH_df.loc[reg_index,'mid']
        adj_trade_df['PNL'] = adj_trade_df['PNL'].fillna(method = 'ffill')
        adj_trade_df['PNL'].fillna(0)
        adj_trade_df['mid'].fillna(method = 'bfill')
        display(adj_trade_df)



    return adj_trade_df.iloc[-1]["PNL"], adj_trade_df, adj_trade_df.loc[reg_index], Adj_book_df, Adj_book_df.loc[reg_index]


def plot_backtest_results(backtest_data, title='Backtest Results, Trades, PNL, Position, and More'):
    # Ensure the index is in datetime format and sorted
    backtest_data.index = pd.to_datetime(backtest_data.index)
    backtest_data = backtest_data.sort_index()
    
    # Compute cumulative maximum of the absolute inventory
    backtest_data['cummax_abs_inv'] = backtest_data['inv'].abs().cummax()
    
    # Compute the new metric for traded over cumulative max inventory
    backtest_data['traded_over_cummaxinv'] = backtest_data['total_size_traded'] / backtest_data['cummax_abs_inv']
    
    # Creating subplots for each metric, now including additional subplots for the new metrics
    fig, axs = plt.subplots(8, 1, figsize=(15, 28), sharex=True)
    
    # Plot 1: Mid price with trade markers
    axs[0].plot(backtest_data.index, backtest_data['mid'], label='Mid Price', color='tab:blue')
    axs[0].set_ylabel('Mid Price')
    axs[0].legend(loc='upper left')
    axs[0].grid(True)

    # Plot 2: PNL
    axs[1].plot(backtest_data.index, backtest_data['PNL'], label='PNL', color='tab:red')
    axs[1].set_ylabel('PNL')
    axs[1].legend(loc='upper left')
    axs[1].grid(True)
    
    # Plot 3: Inventory
    axs[2].plot(backtest_data.index, backtest_data['inv'], label='Inventory', color='tab:orange')
    axs[2].set_ylabel('Inventory')
    axs[2].legend(loc='upper left')
    axs[2].grid(True)

    # Plot 4: Edge Collected
    axs[3].plot(backtest_data.index, backtest_data['edge_collected'], label='Edge Collected', color='tab:purple')
    axs[3].set_ylabel('Edge Collected')
    axs[3].legend(loc='upper left')
    axs[3].grid(True)

    # Plot 5: Percent Edge Retained
    axs[4].plot(backtest_data.index, backtest_data['PctEdgeRetained'], label='Percent Edge Retained', color='tab:pink')
    axs[4].set_ylabel('Percent Edge Retained')
    axs[4].set_xlabel('Time')
    axs[4].legend(loc='upper left')
    axs[4].grid(True)
    
    # Plot 6: total_size_traded / cumulative_max(abs(inventory))
    axs[5].plot(backtest_data.index, backtest_data['traded_over_cummaxinv'], label='Traded/Cumulative Max Inv', color='tab:green')
    axs[5].set_ylabel('Traded/CumMaxInv')
    axs[5].legend(loc='upper left')
    axs[5].grid(True)

    # Plot 7: PNL without fees
    axs[6].plot(backtest_data.index, backtest_data['PNL_wo_fees'], label='PNL without Fees', color='tab:grey')
    axs[6].set_ylabel('PNL without Fees')
    axs[6].legend(loc='upper left')
    axs[6].grid(True)

    # Title and layout adjustments for the full plot
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjusted the rect for the added subplots

    plt.show()








def performance_summary(return_series):

    summary_stats = pd.Series()

    summary_stats['Mean'] = return_series.mean() 
    summary_stats['Annualized_mean'] = summary_stats['Mean']*31536000
    summary_stats['Volatility'] = return_series.std()
    summary_stats['Annualized_Volatility'] = summary_stats['Volatility']*np.sqrt(31536000)
    summary_stats['Sharpe_Ratio'] = summary_stats['Mean'] / summary_stats['Volatility']
    summary_stats['Annualized_Sharpe'] = summary_stats['Sharpe_Ratio']*np.sqrt(31536000)
    summary_stats['Skewness'] = return_series.skew()
    summary_stats['Excess_Kurtosis'] = return_series.kurtosis()

    summary_stats['VaR (0.05)'] = return_series.quantile(0.05)
    summary_stats['CVaR (0.05)'] = return_series[return_series <= summary_stats['VaR (0.05)']].mean()
    summary_stats['Min'] = return_series.min()
    summary_stats['Max'] = return_series.max()

    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks

    summary_stats['Max Drawdown'] = drawdowns.min()
    summary_stats['Peak'] = previous_peaks.idxmax()
    summary_stats['Bottom'] = drawdowns.idxmin()

    recovery_date = wealth_index[drawdowns.idxmin():].ge(previous_peaks).idxmax()
    summary_stats['Recovery'] = recovery_date
    formatted_series = summary_stats.map('{:.5e}'.format)

    return formatted_series

def three_way_split_by_date(book_df, trades_df, name, split_dates, display_data=False):
    """
    Splits the given dataframes into three parts (train, hyper_test, test) based on the split_dates.
    split_dates should be a tuple or list with two dates: (date_for_hyper_test_start, date_for_test_start)
    """
    # Unpack the split dates
    date_for_hyper_test_start, date_for_test_start = split_dates
    
    # Split for train
    book_train = book_df.loc[book_df.index.get_level_values(0) < date_for_hyper_test_start].copy()
    trades_train = trades_df.loc[trades_df.index.get_level_values(0) < date_for_hyper_test_start].copy()
    
    # Split for hyper_test
    book_hyper_test = book_df.loc[(book_df.index.get_level_values(0) >= date_for_hyper_test_start) & (book_df.index.get_level_values(0) < date_for_test_start)].copy()
    trades_hyper_test = trades_df.loc[(trades_df.index.get_level_values(0) >= date_for_hyper_test_start) & (trades_df.index.get_level_values(0) < date_for_test_start)].copy()
    
    # Split for test
    book_test = book_df.loc[book_df.index.get_level_values(0) >= date_for_test_start].copy()
    trades_test = trades_df.loc[trades_df.index.get_level_values(0) >= date_for_test_start].copy()
    
    if display_data:
        # Display data for each split
        print(f"{name} Book Train Dataset:")
        display(book_train.tail())
        print(f"{name} Book Hyper Test Dataset:")
        display(book_hyper_test.head())
        print(f"{name} Book Test Dataset:")
        display(book_test.head())
        
        print(f"{name} Trades Train Dataset:")
        display(trades_train.tail())
        print(f"{name} Trades Hyper Test Dataset:")
        display(trades_hyper_test.head())
        print(f"{name} Trades Test Dataset:")
        display(trades_test.head())
        
    return book_train, book_hyper_test, book_test, trades_train, trades_hyper_test, trades_test

def calc_out_of_sample_MSE(backtest_df):
    # `backtest_df` already contains the predicted values (`y_hat`) and the actual values (`y`)
    
    # Extract the actual and predicted values
    y_test = backtest_df['y']  # Actual values
    y_pred = backtest_df['y_hat']  # Predicted values
    
    # Calculate out-of-sample MSE
    out_of_sample_MSE = mean_squared_error(y_test, y_pred)
    out_of_sample_MSE/(backtest_df['y'].std()**2)
    return out_of_sample_MSE

def calc_indicators(trade_data_train, trade_data_test, book_data_train, book_data_test, cross_trade_data_train, cross_trade_data_test, name ="ETH_USD", volume_halflife = 15000, flow_halflife = 15000, T = 5, rolling_z_score_window = 50, round = 2):
    book_data_test_mkt_theo = book_data_test["MKT_Theo"].copy(deep = True)

    # CALC SIZE DF
    size_train = construct_size_df(book_data_train, trade_data_train)
    size_cross_train = construct_size_df(book_data_train, cross_trade_data_train)
    size_test = construct_size_df(book_data_test, trade_data_test)
    size_cross_test = construct_size_df(book_data_test, cross_trade_data_test)

    # CALC ABSOLUTE VOLUME
    size_train = calc_abs_volume(size_train, volume_halflife)
    size_cross_train = calc_abs_volume(size_cross_train, volume_halflife)
    size_test = calc_abs_volume(size_test, volume_halflife)
    size_cross_test = calc_abs_volume(size_cross_test, volume_halflife)

    # CALC TRADE FLOW
    size_train = Calculate_Trade_Flow(size_train, flow_halflife)
    size_cross_train = Calculate_Trade_Flow(size_cross_train, flow_halflife)
    size_test = Calculate_Trade_Flow(size_test, flow_halflife)
    size_cross_test = Calculate_Trade_Flow(size_cross_test, flow_halflife)

    # ADD INDICATORS TO BOOK DF
    book_data_train = add_volume_and_flow_to_book(book_data_train, size_train)
    book_data_train = add_cross_volume_and_flow_to_book(book_data_train, size_cross_train)
    book_data_test = add_volume_and_flow_to_book(book_data_test, size_test)
    book_data_test = add_cross_volume_and_flow_to_book(book_data_test, size_cross_test)

    # REINDEX DATA FRAME TO REGULAR INTERVALS
    book_data_train_reindexed = reindex_dataframe_to_uniform_intervals(book_data_train, T)

    # CALC % CHANGE IN MKT_Theo
    column_name = "MKT_Theo"
    book_data_test = percent_change(book_data_test, column_name)
    book_data_train_reindexed = percent_change(book_data_train_reindexed, column_name)

    # ADD IN LAGS
    book_data_test = add_lags_and_dep_var(book_data_test, "MKT_Theo_%", num_lags=0)
    book_data_train_reindexed = add_lags_and_dep_var(book_data_train_reindexed, "MKT_Theo_%", num_lags=0)

    # REDUCE COLUMNS
    columns_to_keep = ["volume", "flow", "share_imbalance", "y", "MKT_Theo_%", 'cross_volume', 'cross_flow']
    book_data_test = book_data_test[columns_to_keep]
    book_data_train_reindexed = book_data_train_reindexed[columns_to_keep]

    # Z-SCORE NORMALIZATION
    columns_to_normalize = ['y', 'volume', 'flow','cross_flow','cross_volume', 'MKT_Theo_%']
    book_data_test = normalize(book_data_test, columns_to_normalize)
    book_data_train_reindexed = normalize(book_data_train_reindexed, columns_to_normalize)

    # ROLLING Z-SCORE NORMALIZATION
    columns_to_roling_normalize = []  # Assuming this is intended to be empty or specify columns if needed.
    book_data_test = rolling_normalize(book_data_test, columns_to_roling_normalize, rolling_z_score_window)
    book_data_train_reindexed = rolling_normalize(book_data_train_reindexed, columns_to_roling_normalize, rolling_z_score_window)

    # RUN REGRESSIONS
    beta_list = []

    model = calc_beta(book_data_train_reindexed, name)
    beta_list.append(model.params)

    # CALC PREDICTIONS
    book_data_test = add_predicted_values_single(book_data_test, beta_list[0],book_data_train_reindexed['y'].mean(),book_data_train_reindexed['y'].std())  # Assuming this function exists

    book_data_test['MKT_Theo'] = book_data_test_mkt_theo.round(round)

    columns_to_keep_final = ["MKT_Theo", "y_hat",'y']
    book_data_test_final = book_data_test[columns_to_keep_final]

    # MERGE WITH TRADES
    backtest_df = merge_test_with_trades(book_data_test_final, trade_data_test)  # Assuming this function exists

    out_of_sample_MSE = calc_out_of_sample_MSE(backtest_df)
    backtest_df.drop(columns=['y'], inplace=True)

    return out_of_sample_MSE, backtest_df

def indicator_hyperparameter_tuner(trade_data_train, trade_data_test, book_data_train, book_data_test, cross_trade_data_train, cross_trade_data_test, name,
                                     volume_halflife_list, flow_halflife_list, t_list, rolling_z_score_window_list, round_p = 2):
    
    trade_data_train_cpy = trade_data_train.copy(deep = True)
    trade_data_test_cpy = trade_data_test.copy(deep = True)
    book_data_train_cpy = book_data_train.copy(deep = True)
    book_data_test_cpy = book_data_test.copy(deep = True)
    cross_trade_data_train_cpy = cross_trade_data_train.copy(deep = True)
    cross_trade_data_test_cpy = cross_trade_data_test.copy(deep = True)
    # Initialize variables to track the best results
    best_MSE = 9999999999.9
    best_params = {}
    best_backtest_df = None
    MSE_list = []  # List to store all R-squared values
    
    # Generate all combinations of hyperparameters
    all_combinations = itertools.product(volume_halflife_list, flow_halflife_list, t_list, rolling_z_score_window_list)
    counter = 0
    # Iterate through each combination
    for combination in all_combinations:
        display(combination)
        counter += 1
        display(counter)
        volume_halflife, flow_halflife, T, rolling_z_score_window = combination
        '''
        trade_data_train = trade_data_train_cpy
        trade_data_test = trade_data_test_cpy
        book_data_train = book_data_train_cpy
        book_data_test = book_data_test_cpy
        cross_trade_data_train = cross_trade_data_train_cpy
        cross_trade_data_test = cross_trade_data_test_cpy
        '''
        # Call your calc_indicators function with the current set of hyperparameters
        MSE, backtest_df = calc_indicators(
            trade_data_train, trade_data_test, book_data_train, book_data_test,
            cross_trade_data_train, cross_trade_data_test, name,
            volume_halflife, flow_halflife, T, rolling_z_score_window, round_p
        )

        # Store the R-squared value in the list
        MSE_list.append(MSE)
        
        # Update the best results if the current R-squared is higher
        if MSE < best_MSE:
            best_MSE = MSE
            best_params = {"volume_halflife":volume_halflife, "flow_halflife":flow_halflife, "T":T, "rolling_z_score_window":rolling_z_score_window}
            best_backtest_df = backtest_df
        display(MSE)
        display(best_MSE)
        display()
        display()
    # Return the best set of hyperparameters, the associated backtest_df, the best R-squared, and all R-squared values
    return best_params, best_backtest_df, best_MSE, MSE_list

def run_backtest(backtest_df, book_test_cpy,
                 capital=100000, leverage=2.0, participation_rate=0.03, 
                 min_edge=5, min_size=0.02, max_size=1.5, slope=0.001,MCR_mult = 10, ETH_df=None):
    """
    Runs a backtest for market making strategy on given data.

    :param backtest_df: DataFrame containing backtest data.
    :param book_test_cpy: DataFrame containing book test data.
    :param ETH_df: Optional DataFrame containing additional ETH data.
    :param capital: Initial capital for the strategy.
    :param leverage: Leverage to use in the strategy.
    :param participation_rate: Participation rate in the market.
    :param min_edge: Minimum edge required to make a trade.
    :param min_size: Minimum trade size.
    :param max_size: Maximum trade size.
    :param slope: Slope parameter for the strategy.
    :return: Tuple containing final PnL, adjusted total backtest data, adjusted regular backtest data,
             adjusted total book test data, and adjusted regular book test data.
    """
    print(min_edge)
    print(min_size)
    print(max_size)
    print(slope)
    print(MCR_mult)
    backtest_data = backtest_market_making_strategy(
        backtest_df, capital=capital, leverage=leverage, participation_rate=participation_rate,
        min_edge=min_edge, min_size=min_size, max_size=max_size, slope=slope,MCR_multiplier=MCR_mult
    )
    # Remove rows with duplicate index values in backtest_data
    backtest_data = backtest_data[~backtest_data.index.duplicated(keep='last')]
    if ETH_df is None:
        
        # Regularize backtest data without ETH_df
        final_pnl, adj_total_backtest_data, adj_reg_backtest_data, adj_total_book_test, adj_reg_book_test = regularize_backtest_data(
            backtest_data, book_test_cpy
        )
        
    return final_pnl, adj_total_backtest_data, adj_reg_backtest_data, adj_total_book_test, adj_reg_book_test


def calculate_sharpe_ratio(adj_reg_backtest_data, capital=100000.0):
    """
    Calculates the Sharpe Ratio for given returns.
    """
    return_data = adj_reg_backtest_data['PNL'].diff()/capital
    return_data.mean()
    return float(performance_summary(return_data).iloc[4])

def strategy_hyperparameter_tuner(backtest_df, book_test_cpy, trading_cost, capital, ETH_df = None):
    # Define the hyperparameter grid
    min_edge_multipliers = [1,3]#np.linspace(start=1, stop=3, num=2)  # Example range for multiplier of trading_cost for min_edge
    min_size_multipliers = [0.0001, 0.001]#np.linspace(0.0001, 0.001, num=2)  # 0.01% to 0.1% of capital for min_size
    max_size_multipliers = [0.005, 0.03]#np.linspace(0.005, 0.03, num=2)  # 0.5% to 3% of capital for max_size
    mcr_multipliers = [25.0]#np.linspace(5, 50, num=3)  # Example: from 5 to 50, in steps of 5
    slope_values = []  # Will be dynamically calculated based on min_size/min_edge

    best_sharpe_ratio = -np.inf
    best_params = None
    best_performance = None
    counter = 0

    for min_edge_multiplier in min_edge_multipliers:
        for min_size_multiplier in min_size_multipliers:
            for max_size_multiplier in max_size_multipliers:
                for mcr_multiplier in mcr_multipliers:
                    # Calculate dependent hyperparameters
                    min_edge = min_edge_multiplier * (trading_cost * 2144.16)
                    min_size = (min_size_multiplier * capital) / 2144.16 # divide by the inital price of ETH
                    max_size = (max_size_multiplier * capital) / 2144.16
                    initial_slope = min_size / min_edge
                    # Extend or reduce slope around the initial value
                    for slope_adjustment in [0.5,1.0,1.5]:#np.linspace(0.8, 1.2, num=2):
                        slope = initial_slope * slope_adjustment
                        # Run the strategy backtest with the current set of parameters
                        final_pnl, adj_total_backtest_data, adj_reg_backtest_data, adj_total_book_test, adj_reg_book_test = run_backtest(
                            backtest_df=backtest_df, book_test_cpy=book_test_cpy,
                            ETH_df = ETH_df,
                            capital=capital, leverage=2.0, participation_rate=0.03,
                            min_edge=min_edge, min_size=min_size, max_size=max_size, slope=slope,MCR_mult=mcr_multiplier
                        )
                        # Calculate the Sharpe Ratio for this set of parameters
                        sharpe_ratio = calculate_sharpe_ratio(adj_reg_backtest_data)
                        print("sharp:")
                        display(sharpe_ratio)
                        counter+=1
                        print(counter)
                        print()
                        # Update the best parameters if this set performs better
                        if sharpe_ratio > best_sharpe_ratio:
                            best_sharpe_ratio = sharpe_ratio
                            best_params = {
                                'min_edge': min_edge,
                                'min_size': min_size,
                                'max_size': max_size,
                                'mcr_multiplier': mcr_multiplier,
                                'slope': slope
                            }
                            best_performance = (final_pnl, adj_total_backtest_data, adj_reg_backtest_data, adj_total_book_test, adj_reg_book_test)

    return best_params, best_performance

def full_backtest_process(trade_data_train, trade_data_test, book_data_train, book_data_test,
                          cross_trade_data_train, cross_trade_data_test,
                          strat_params, ind_params,
                          trading_cost=0.0015, capital=100000, leverage=2.0, participation_rate=0.03, ETH_df = None):
    
    # Extract strategy parameters
    min_edge = strat_params['min_edge']
    min_size = strat_params['min_size']
    max_size = strat_params['max_size']
    slope = strat_params['slope']
    MCR_mult = strat_params['mcr_multiplier']
    # Extract indicator parameters
    volume_halflife = ind_params['volume_halflife']
    flow_halflife = ind_params['flow_halflife']
    T = ind_params['T']
    rolling_z_score_window = ind_params['rolling_z_score_window']
    
    if ETH_df is None:
        name = 'ETH/USD'
        round_p = 2
    else:
        name = 'ETH/BTC'
        round_p = 5

    # Calculate indicators
    MSE, backtest_df = calc_indicators(
        trade_data_train, trade_data_test, book_data_train, book_data_test,
        cross_trade_data_train, cross_trade_data_test, name,
        volume_halflife, flow_halflife, T, rolling_z_score_window, round_p
    )
    
    
    # Run backtest
    final_pnl, adj_total_backtest_data, adj_reg_backtest_data, adj_total_book_test, adj_reg_book_test = run_backtest(
        backtest_df=backtest_df, book_test_cpy=book_data_test.copy(),
        capital=capital, leverage=leverage, participation_rate=participation_rate,
        min_edge=min_edge, min_size=min_size, max_size=max_size, slope=slope,MCR_mult=MCR_mult,ETH_df=ETH_df
    )
    
    # Print the best final PnL
    print("Best Final PnL:", final_pnl)
    
    return final_pnl, adj_total_backtest_data, adj_reg_backtest_data, adj_total_book_test, adj_reg_book_test


def create_enhanced_scatterplot(df, x_col, y_col, title='Scatter Plot', xlabel='X-axis', ylabel='Y-axis', figsize=(10, 8), style='seaborn-darkgrid', point_color='blue', alpha=0.5, edgecolor='w', linewidth=0.5):
    """
    Create an enhanced scatter plot from a DataFrame using matplotlib.

    Parameters:
    - df: DataFrame containing the data to plot.
    - x_col: The name of the column to use for the x-axis.
    - y_col: The name of the column to use for the y-axis.
    - title: The title of the plot.
    - xlabel: The label for the x-axis.
    - ylabel: The label for the y-axis.
    - figsize: Size of the figure (width, height).
    - style: Style of the plot (e.g., 'seaborn-darkgrid').
    - point_color: Color of the scatter points.
    - alpha: Transparency of the scatter points.
    - edgecolor: Edge color of the scatter points.
    - linewidth: Line width of the edge of the scatter points.
    """
    # Apply the selected style
    #plt.style.use(style)

    # Create a figure and axis with the specified size
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot using the specified parameters
    scatter = ax.scatter(x=df[x_col],
                         y=df[y_col],
                         c=point_color, 
                         alpha=alpha, 
                         edgecolors=edgecolor, 
                         linewidth=linewidth)

    # Set title and labels with font settings
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # Customize tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Enable grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Display the plot
    plt.show()
