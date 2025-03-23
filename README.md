# COGS Forecasting
-----------

## Project Description
This project is designed to forecast COGS Value over a certain period of time using SARIMAX algorithm. 

## Installation
### Creating an Environment
To ensure that all dependencies and libraries used later use the same version and do not produce any errors, we need to install the environment. To install the environment, the user can run this script in the terminal.
```bash
$ make create-env
$ conda activate forecast-env
```
-----------
## Forecasting Pipeline
#### Scraping, Predicting, and Summarizing News Data
The script below is used to run the main forecasting pipeline. Within this script, users can specify which client to forecast and define whether or not suffix or/and sampling is needed.


| Parameter  | Description |
|-----------|------------|
| `CLIENT`  | Specifies the client name for which the forecast is being generated. Example: `'ferrero'`. |
| `SUFFIX`  | Determines the suffix of the output file name. Example: `'test'`. |
| `SAMPLING` | Determines whether data sampling is enabled. Choose `'True'` to enable sampling and `'False'` to disable it. |

Here's an example of a script that can be run.
```bash
$ export CLIENT='ferrero' &&
export SUFFIX='test' &&
export SAMPLING=False &&
make forecast-client
```

#### Streamlit
To run the Streamlit for the Stock app, run this command.
```bash
make run-streamlit-stock
```
-----------


