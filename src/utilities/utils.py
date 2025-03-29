import os
import sys
import glob
import re
import gzip
import pickle
import yaml
import feather

import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime, timedelta
from jinja2 import Template
from loguru import logger

from itertools import product
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX


def save(data, filename):
    folders = os.path.dirname(filename)
    if folders:
        os.makedirs(folders, exist_ok=True)

    fl = filename.lower()
    if fl.endswith(".gz"):
        if fl.endswith(".feather.gz") or fl.endswith(".fthr.gz"):
            # Since feather doesn't support writing to the file handle, we
            # can't easily point it to gzip.
            raise NotImplementedError(
                "Saving to compressed .feather not currently supported."
            )
        else:
            fp = gzip.open(filename, "wb")
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        if fl.endswith(".feather") or fl.endswith(".fthr"):
            if str(type(data)) != "<class 'pandas.core.frame.DataFrame'>":
                raise TypeError(
                    ".feather format can only be used to save pandas "
                    "DataFrames"
                )
            feather.write_dataframe(data, filename)
        else:
            fp = open(filename, "wb")
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load(filename):
    """
    Loads data saved with save() (or just normally saved with pickle).
    Autodetects gzip if filename ends in '.gz'
    Also reads feather files denoted .feather or .fthr.

    Parameters
    ----------
    filename -- String with the relative filename of the pickle/feather
    to load.
    """
    fl = filename.lower()
    if fl.endswith(".gz"):
        if fl.endswith(".feather.gz") or fl.endswith(".fthr.gz"):
            raise NotImplementedError("Compressed feather is not supported.")
        else:
            fp = gzip.open(filename, "rb")
            return pickle.load(fp)
    else:
        if fl.endswith(".feather") or fl.endswith(".fthr"):
            import feather

            return feather.read_dataframe(filename)
        else:
            fp = open(filename, "rb")
            return pickle.load(fp)
        

def read_yaml(filename, render=False, **kwargs):
    """
    Read yaml configuation and returns the dict

    Parameters
    ----------
    filename: string
        Path including yaml file name
    render: Boolean, default = False
        Template rendering
    **kwargs:
        Template render args to be passed
    """
    if render:
        yaml_text = Template(open(filename, "r").read())
        yaml_text = yaml_text.render(**kwargs)
        config = yaml.safe_load(yaml_text)
    else:
        with open(filename) as f:
            config = yaml.safe_load(f)

    return config

def preprocess_df(df, outlier):
    # Filter by outlier type
    df = df[df['Outlier'] == outlier]

    # Group by 'Inv Date (MMM-YYYY)' and calculate the average 'Total COGS Value'
    df_grouped = df.groupby(['Inv Date (MMM-YYYY)', 'Material Code', 'Storage Location Code'], as_index=False).agg(
        {
            'Material Group Code' : 'first',
            'Material Group Desc' : 'first',
            'Material Desc' : 'first',
            'Total COGS EA': 'mean',
            'Total COGS CTN': 'mean',
            'Total COGS Value': 'mean',
            'Outlier': 'first'  # Keeps the first Outlier flag (modify if needed)
        }  # Computes avg COGS and keeps first Outlier flag
    )

    # Rename columns
    df_grouped.rename(columns={
        'Total COGS EA': 'AVG Total EA',
        'Total COGS CTN': 'AVG Total CTN',
        'Total COGS Value': 'AVG Total RM'
    }, inplace=True)

    # Convert 'Inv Date (MMM-YYYY)' to datetime format
    df_grouped['Date'] = pd.to_datetime(df_grouped['Inv Date (MMM-YYYY)'], format='%b - %Y')

    # Sort DataFrame by date
    df_grouped = df_grouped.sort_values("Date")
    df_grouped.reset_index(drop=True, inplace=True)

    return df_grouped

def early_plot(df, column):
    # Column could be AVG Total RM, EA / CTN
    # Plot
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=df["Date"], y=df[column], marker='o', linestyle='-')

    # Labels & Title
    plt.xlabel("Invoice Date")
    plt.ylabel(column)
    plt.title(f"{column} Over Time")
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.grid(True)

    # Show plot
    plt.show()

def train_predict(df, column):
    # Column could be AVG Total RM, EA / CTN
    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Sort values to ensure correct order
    df_base_sample = df.sort_values("Date").reset_index(drop=True)

    # List of seasonal values to try (prioritizing 12 and 6, then smaller values if needed)
    seasonal_options = [12, 8, 6, 4, 3, 2]

    for seasonal_period in seasonal_options:
        try:
            # Define SARIMA model
            model = SARIMAX(df_base_sample[column], 
                            order=(1,1,1),       
                            seasonal_order=(1,1,1,seasonal_period),  
                            enforce_stationarity=False,
                            enforce_invertibility=False)

            model_fit = model.fit(disp=False)
            
            # Forecast next 12 months
            forecast = model_fit.forecast(steps=24)

            # Replace negative values with 0
            forecast[forecast < 0] = 0
            
            # Explosion check: If the forecast exceeds 5 * max actual value, it's unstable
            if (forecast.abs().max() > 3 * df_base_sample[column].max()):
                # print(f"⚠️ Forecast exploded with seasonal order {seasonal_period}, trying next...")
                continue
            
            # print(f"Model trained successfully with seasonal order {seasonal_period}")
            return forecast  # Return forecast if successful
        
        except Exception as e:
            logger.info(f"Model failed with seasonal order {seasonal_period}: {e}")
            continue

    logger.info("No valid SARIMAX model could be trained. Consider removing seasonality.")
    return None  # Return None if all attempts fail

def postprocess(df_base_sample, forecast, column, plot):
    # Column could be AVG Total RM, EA / CTN
    # Get the last date in the dataset
    last_date = df_base_sample['Date'].max()

    # Generate future dates for the next 12 months, keeping the day as 1
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),  # Start from next month
                                periods=24,  # Generate 12 months
                                freq='MS')   # 'MS' ensures the 1st day of each month


    if plot:
        # Plot actual data and forecast
        plt.figure(figsize=(10, 5))
        plt.plot(df_base_sample.Date, df_base_sample[column], label="Actual Data", marker='o')
        plt.plot(future_dates, forecast, label="Forecast", linestyle='dashed', marker='x', color='red')

        plt.xlabel("Date")
        plt.ylabel(column)
        plt.title(f"Corrected Forecast of {column}")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return future_dates

def all_storage_grouper(df):
    # Add this to switch Storage Location to = all
    df = df.groupby(['Inv Date (MMM-YYYY)', 'Material Code'], as_index=False).agg(
            {
                'Material Group Code' : 'first',
                'Material Group Desc' : 'first',
                'Material Desc' : 'first',
                'AVG Total EA': 'mean',
                'AVG Total CTN': 'mean',
                'AVG Total RM': 'mean',
                'Outlier': 'first'  # Keeps the first Outlier flag (modify if needed)
            }  # Computes avg COGS and keeps first Outlier flag
        )   
    
    # Convert 'Inv Date (MMM-YYYY)' to datetime format
    df['Date'] = pd.to_datetime(df['Inv Date (MMM-YYYY)'], format='%b - %Y')

    # Sort DataFrame by date
    df = df.sort_values("Date")
    df.reset_index(drop=True, inplace=True)
    
    return df