import pandas as pd
import numpy as np
import os 
import sys
import click
import glob

from loguru import logger
from datetime import datetime, timedelta
from src.utilities.config_ import log_path, ConfigManager, config_path, morgan_train_data_path, predicted_data_path, ferrero_train_data_path
from itertools import product

import src.utilities.utils as utils

@click.command()
@click.option(
    "--client",
    required=True,
    type=str,
    help="Define which client that wants to be forecasted.",
)
@click.option(
    "--suffix",
    "-s",
    required=False,
    type=str,
    default="",
    help="Suffix for the output names.",
)
@click.option(
    "--sample",
    "-s",
    type=bool,
    help="Set to True to perform sampling (limit to ~10 material codes), or False to disable sampling."
)
def main_predict(
    client,
    suffix,
    sample
):
    container_date = datetime.now().strftime("%Y-%m-%d")
    logger.remove()
    logger.add(
        os.path.join(log_path, "run_forecasting_" + container_date + ".log"),
        format="<green>{time}</green> | <yellow>{name}</yellow> | {level} |"
        " <cyan>{message}</cyan>"
    )
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time}</green> | <yellow>{name}</yellow> | {level} |"
        " <cyan>{message}</cyan>"
    )

    # load some config
    params = utils.read_yaml(
        os.path.join(config_path, "main_config.yaml"), render=True, suffix=suffix
    )

    # set predicted feathername by mapping the input
    pred_feathername = f'{params["run_forecasting_params"][client]["predicted_feathername"]}'

    # training data path mapping based on client value
    training_data = {
        "morgan" : morgan_train_data_path,
        "ferrero" : ferrero_train_data_path
        # soon to be added
    }

    logger.info(
        "Sentiment Forecasting Params- \n"
        + f" Client: {client} |\n"
        + f" Suffix: {suffix} |\n"
        + f" Output File Name: {pred_feathername} |\n"
    )

    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(training_data[client], "*.csv"))

    # Read each CSV into a DataFrame and store them in a list
    dataframes = [pd.read_csv(file) for file in csv_files]

    # Optionally, concatenate all DataFrames into one
    df = pd.concat(dataframes, ignore_index=True).reset_index(drop=True)

    # Hyperparameters :)
    COLUMN = ["AVG Total RM", "AVG Total EA", "AVG Total CTN"]
    STORAGE_TYPE = ["All", "Specific"]
    OUTLIER = [True, False]

    # Store a new empty list for the final result
    results = []

    # Iterate over all parameter combinations
    for idx, (col, storage_type, outlier) in enumerate(product(COLUMN, STORAGE_TYPE, OUTLIER), start=1):
        logger.info(f"Running {idx}/12 → COLUMN={col}, STORAGE_TYPE={storage_type}, OUTLIER={outlier}")

        # Preprocess data once per OUTLIER setting
        df_grouped = utils.preprocess_df(df, outlier)
        
        # Apply grouping if storage_type is "All"
        if storage_type != "Specific":
            df_grouped = utils.all_storage_grouper(df_grouped)
            material_storage_pairs = ((m, None) for m in df_grouped["Material Code"].unique())  # Use (material, None) pairs
        else:
            material_storage_pairs = product(df_grouped["Material Code"].unique(), df_grouped["Storage Location Code"].unique())

        # Iterate over material-storage pairs (up to LOOP_VALUE)
        for i, (material, storage) in enumerate(material_storage_pairs):
            if sample:
                if i >= 10: 
                    break  # Stop early if LOOP_VALUE is reached

            # Efficient DataFrame filtering using .query()
            if storage:
                df_sample = df_grouped.query("`Material Code` == @material and `Storage Location Code` == @storage")
            else:
                df_sample = df_grouped.query("`Material Code` == @material")

            # Skip if DataFrame is empty
            if df_sample.empty:
                continue

            try:
                # Train and forecast
                forecast = utils.train_predict(df_sample, col)

                # Post-process and plot results
                future_dates = utils.postprocess(df_sample, forecast, col, False)

                # Determine COGS Type
                cogs_type = "RM" if col == "AVG Total RM" else "EA" if col == "AVG Total EA" else "CTN"

                # Store results
                results.extend([
                    {
                        "Date": future_date,
                        "Material Code": material,
                        "Storage Location Code": storage if storage else "All",
                        "COGS Type": cogs_type,
                        "COGS Value": value,
                        "Outlier": outlier
                    }
                    for future_date, value in zip(future_dates, forecast)
                ])

            except Exception as error:
                print(f"⚠️ Error for Material {material}: {error}")
                continue

    # Convert results list into a DataFrame
    df_results = pd.DataFrame(results)

    # Postprocess data type
    df_results["Storage Location Code"] = df_results["Storage Location Code"].astype(str)

    # save
    logger.info(f"Saving feather as {pred_feathername}...")
    utils.save(df_results, os.path.join(predicted_data_path, pred_feathername))


if __name__ == "__main__":
    main_predict()