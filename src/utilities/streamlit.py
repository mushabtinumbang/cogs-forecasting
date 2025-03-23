import streamlit as st
import time
import pandas as pd
import numpy as np
import os
import glob
import plotly.express as px
import plotly.graph_objects as go

import src.utilities.utils as utils
from src.utilities.config_ import predicted_data_path, morgan_train_data_path, ferrero_train_data_path

def greet():
    st.toast('Hello!', icon='✅')
    time.sleep(1)
    st.toast('Welcome!', icon='✅')

def stspace(num):
    for j in range(num):
        st.write("")

def get_predicted_data(client):
    df = utils.load(os.path.join(predicted_data_path, f'{client}_predicted_.feather'))
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def get_client_data(client):
    # training data path mapping based on client value
    training_data = {
        "morgan" : morgan_train_data_path,
        "ferrero": ferrero_train_data_path
        # soon to be added
    }

    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(training_data[client], "*.csv"))

    # Read each CSV into a DataFrame and store them in a list
    dataframes = [pd.read_csv(file) for file in csv_files]

    # Optionally, concatenate all DataFrames into one
    df = pd.concat(dataframes, ignore_index=True).reset_index(drop=True)

    df["Inv Date"] = pd.to_datetime(df["Inv Date"])
    df["Year"] = df["Inv Date"].dt.year  # Extract Year for Filtering

    return df

def plot(baseline_df, outlier_df, cogs_column, year_option, df_predicted_filtered_baseline, df_predicted_filtered_outlier):
    # Create Plotly Line Chart
    fig = px.line()

    # Add Baseline (Blue Line)
    fig.add_scatter(x=baseline_df["Inv Date"], y=baseline_df[cogs_column], 
                    mode="lines+markers", name=f"Baseline ({'Total COGS RM' if cogs_column == 'Total COGS Value' else cogs_column})", line=dict(color="blue"))

    # Add Outliers (Red Line)
    fig.add_scatter(x=outlier_df["Inv Date"], y=outlier_df[cogs_column], 
                    mode="lines+markers", name="Outlier", line=dict(color="red"))

    # Add Outliers (Red Line)
    fig.add_scatter(x=df_predicted_filtered_baseline["Date"], y=df_predicted_filtered_baseline["COGS Value"], 
                    mode="lines+markers", name="Baseline Predicted", line=dict(color="orange"))

    # Add Outliers (Red Line)
    fig.add_scatter(x=df_predicted_filtered_outlier["Date"], y=df_predicted_filtered_outlier["COGS Value"], 
                    mode="lines+markers", name="Outlier Predicted", line=dict(color="black"))

    # Layout settings
    fig.update_layout(
        title=f"COGS Sales and Outliers - {year_option}",
        xaxis_title="Inv Date (MMM-YYYY)",
        yaxis_title="Avg COGS Sales",
        legend_title="Legend",
        template="plotly_white"
    )

    return fig

