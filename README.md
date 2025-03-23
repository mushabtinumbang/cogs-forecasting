# COGS Forecasting
-----------

## Project Description
This project is designed to forecast COGS Value over a certain period of time using SARIMAX algorithm. 

## Installation
#### Clone Repository
Make sure you have installed git lfs (Large File Storage) to be able to git clone the Machine Learning model.
```bash
$ git clone https://github.com/mushabtinumbang/market-sentiment-LLM.git
$ cd market-sentiment-LLM
$ git lfs install
$ git lfs pull
```
### Creating an Environment
The next step to run this program is to create a conda environment. This is to ensure that all dependencies and libraries used later use the same version and do not produce any errors. To install the environment, the user can run this script in the terminal.
```bash
$ make create-env
$ conda activate market-dash
```
### Setup BART for Summarizer
Since the quota for Git LFS is limited, we have to download the BART summarizer model manually. To download the model, run the code snippet below.
```bash
$ make setup-bart
```
-----------
## Stock Market Pipeline
#### Scraping, Predicting, and Summarizing News Data
The script below is used to run the main program for the stock market app. With this script, users can scrape stock-related news pages in real-time. Users can specify the date range for the news they wish to scrape.

After scraping, users can instantly predict the sentiment and summarize the content of the stock news. The prediction results are then processed and presented within the Streamlit interface.

| Command          | Description |
|-----------------|------------|
| DATE            | Determines the period of data to be processed. Example: 'latest' or for a date range, input it like this: "2020-01-02\|2020-01-03" (multiple dates separated by \|). |
| SUFFIX          | Determines the suffix of the output file name. |
| RUN_SCRAPER     | Determines whether to run the scraping function or not. Choose 'y' for yes and 'n' for no. |
| RUN_SUMMARIZER     | Determines whether to run the summarizer function or not. Choose 'y' for yes and 'n' for no. |
| RUN_PREDICTION  | Determines whether to run the prediction using the machine learning model or not. Choose 'y' for yes and 'n' for no. |
| PREPARE_STREAMLIT | Determines whether to further process the data for Streamlit or not. Choose 'y' for yes and 'n' for no. |

Based on these parameters, users can change the date, specify the news pages to scrape, and customize the overall pipeline. Here's an example of a script that can be run.
```bash
$ export DATE='06-10-2024|07-10-2024' && 
export STOCK='META' &&
export SUFFIX='test' &&
export RUN_SCRAPER='y' &&
export RUN_SUMMARIZER='y' &&
export RUN_PREDICTION='y' &&
export PREPARE_STREAMLIT='n' &&
make predict-stocks
```

#### Streamlit
To run the Streamlit for the Stock app, run this command.
```bash
make run-streamlit-stock
```
-----------
## Forex Market Pipeline
### Running the Program
#### Scraping and Predicting News Data
The script below is used to run the main program for forex app. With this script, users can perform real-time scraping on financial forexd news pages. Users can also set the date of the news they want to scrape.

After scraping, users can also immediately predict the sentiment of the news. The prediction results will then be processed and displayed in the Streamlit interface.

| Command          | Description |
|-----------------|------------|
| DATE            | Determines the period of data to be processed. Example: 'latest' or for a date range, input it like this: "2020-01-02\|2020-01-03" (multiple dates separated by \|). |
| DAILYFX         | Determines whether to scrape data from DailyFX or not. Choose 'y' for yes and 'n' for no. **Note: Currently, the page is not available for some network in Indonesia, please use VPN if you want to proceed to scrape DailyFX**|
| ECONTIMES       | Determines whether to scrape data from Economic Times or not. Choose 'y' for yes and 'n' for no. |
| FINANCIALTIMES       | Determines whether to scrape data from Financial Times or not. Choose 'y' for yes and 'n' for no. |
| SUFFIX          | Determines the suffix of the output file name. |
| RUN_SCRAPER     | Determines whether to run the scraping function or not. Choose 'y' for yes and 'n' for no. |
| RUN_PREDICTION  | Determines whether to run the prediction using the machine learning model or not. Choose 'y' for yes and 'n' for no. |
| PREPARE_STREAMLIT | Determines whether to further process the data for Streamlit or not. Choose 'y' for yes and 'n' for no. |

Based on these parameters, users can change the date, specify the news pages to scrape, and customize the overall pipeline. Here's an example of a script that can be run.

```bash
$ export DATE='29-06-2024|12-07-2024' &&
export DAILYFX='n' &&
export ECONTIMES='y' &&
export FINANCIALTIMES='y' &&
export SUFFIX='test' &&
export RUN_SCRAPER='y' &&
export RUN_PREDICTION='y' &&
export PREPARE_STREAMLIT='y' &&
make predict-sentiments
```

#### Streamlit
To run the Streamlit for the Forex app, run this command.
```bash
make run-streamlit
```
-----------

## Citation

This project was created by Mushab Tinumbang under [Universitas Padjadjaran](https://www.unpad.ac.id/).