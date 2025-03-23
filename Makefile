###########################################################################################################
## VARIABLES
###########################################################################################################
PYTHON=python
CONDA=conda
STREAMLIT=streamlit
CURRENT_DIR := $(PWD)
SRC_DIR=$(CURRENT_DIR)/src
MAIN_DIR=$(SRC_DIR)/main

###########################################################################################################
## SCRIPTS
###########################################################################################################
# Create conda env to run MM
create-env:
	$(CONDA) env update --file environment.yml

# Run Main Predict Pipeline
forecast-client:
	$(PYTHON) -m src.main.main_predict --client='$(CLIENT)' --suffix='$(SUFFIX)' --sample='$(SAMPLING)'

# Run Streamlit 
run-streamlit:
	$(STREAMLIT) run app.py