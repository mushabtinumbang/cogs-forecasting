import streamlit as st
import pandas as pd
import plotly.express as px

from src.utilities.streamlit import *
from streamlit_option_menu import option_menu

# Set Layout Config
st.set_page_config(page_title="Market Dashboard", page_icon=":chart_with_upwards_trend:",layout="wide")
     
# Add Font
streamlit_style = """
			<style>
			@import url('https://fonts.cdnfonts.com/css/neue-haas-grotesk-display-pro');    

			html, body, [class*="css"]  {
			font-family: 'neue haas grotesk display pro';
			}
			</style>
			"""

# Apply Font
st.markdown(streamlit_style, unsafe_allow_html=True)

if True:
    # Greet
    if "greet" not in st.session_state:
        greet()
        st.session_state.greet = True

    # Make a Centered Text
    col1, col2, col3, col4, col5, col6, col7 = st.columns((6, 6, 6, 14, 6, 6, 6))

    # Add whitespace
    stspace(3)

    with col1:
        client_option = st.selectbox("Client", ("Morgan", "Ferrero"))

        # Cache training df
        if "df_train" not in st.session_state:
            df_train = get_client_data(client_option.lower())
            st.session_state.df_train = df_train
        else:
            df_train = st.session_state.df_train

        # Cache predicted df
        if "df_predicted" not in st.session_state:
            df_predicted = get_predicted_data(client_option.lower())
            st.session_state.df_predicted = df_predicted
        else:
            df_predicted = st.session_state.df_predicted

        # update cache if client_option changed
        if "current_client" not in st.session_state:
            st.session_state.current_client = client_option

        if st.session_state != client_option:
            df_train = get_client_data(client_option.lower())
            st.session_state.df_train = df_train

            df_predicted = get_predicted_data(client_option.lower())
            st.session_state.df_predicted = df_predicted

        # Get unique values
        unique_sloc = df_train["Storage Location Code"].dropna().unique().tolist()
        unique_mat = df_train["Material Code"].dropna().unique().tolist()
        # Get unique years and sort in descending order
        unique_years = ["Whole"] + sorted(df_train["Year"].unique(), reverse=True)

    with col2:
        sloc_option = st.selectbox("Sloc", ["All"] + unique_sloc)

    with col3:
        mat_option = st.selectbox("Material Code", unique_mat)

    with col4:
        year_option = st.selectbox("Select Year", unique_years, index=0)  # Default: latest year

    with col7:
        COGS_opt = st.selectbox("COGS", ('RM', 'EA', 'CTN'))

    # Map COGS option to correct column
    cogs_column_mapping = {
        "RM": "Total COGS Value",
        "EA": "Total COGS EA",
        "CTN": "Total COGS CTN"
    }
    cogs_column = cogs_column_mapping[COGS_opt]

    # Apply Filters
    filtered_df = df_train[(df_train["Material Code"] == mat_option)]
    if year_option != "Whole":
        filtered_df = filtered_df[(filtered_df["Year"] == year_option)]

    if sloc_option != "All":
        filtered_df = filtered_df[filtered_df["Storage Location Code"] == sloc_option]

    # Sort Data
    filtered_df = filtered_df.sort_values("Inv Date")

    # Separate Baseline (Outlier=False) and Outlier (Outlier=True)
    baseline_df = filtered_df[filtered_df["Outlier"] == False]
    outlier_df = filtered_df[filtered_df["Outlier"] == True]

    # Groupby to get the AVG value
    baseline_df = baseline_df.groupby(['Inv Date (MMM-YYYY)', 'Material Code']).agg(
    {
        'Total COGS EA': 'mean',
        'Total COGS CTN': 'mean',
        'Total COGS Value': 'mean',
        'Outlier': 'first', # Keeps the first Outlier flag (modify if needed)
        'Inv Date': 'first' # Keeps the first Outlier flag (modify if needed)
    }  # Computes avg COGS and keeps first Outlier flag
    ).reset_index().sort_values("Inv Date")

    # Groupby to get the AVG value
    outlier_df = outlier_df.groupby(['Inv Date (MMM-YYYY)', 'Material Code']).agg(
    {
        'Total COGS EA': 'mean',
        'Total COGS CTN': 'mean',
        'Total COGS Value': 'mean',
        'Outlier': 'first', # Keeps the first Outlier flag (modify if needed)
        'Inv Date': 'first' # Keeps the first Outlier flag (modify if needed)
    }  # Computes avg COGS and keeps first Outlier flag
    ).reset_index().sort_values("Inv Date")

    # df_predicted["Storage Location Code"] = df_predicted["Storage Location Code"].astype
    df_predicted["Year"] = df_predicted["Date"].dt.year  # Extract Year for Filtering
    df_predicted_filtered = df_predicted[
        (df_predicted["Material Code"] == mat_option) & 
        (df_predicted["COGS Type"] == COGS_opt) &
        (df_predicted["Storage Location Code"] == str(sloc_option)) 
    ].reset_index().sort_values("Date")

    if year_option != "Whole":
        df_predicted_filtered = df_predicted_filtered[
            (df_predicted_filtered["Year"] == year_option)
        ].reset_index().sort_values("Date")

    # Make another column
    df_predicted_filtered["Inv Date (MMM-YYYY)"] = df_predicted_filtered["Date"].dt.strftime("%b - %Y")

    # Make baseline and outlier df
    df_predicted_filtered_baseline = df_predicted_filtered[df_predicted_filtered["Outlier"] != True]
    df_predicted_filtered_outlier = df_predicted_filtered[df_predicted_filtered["Outlier"] == True]

    # Plot
    fig = plot(baseline_df, outlier_df, cogs_column, year_option, df_predicted_filtered_baseline, df_predicted_filtered_outlier)

    # Show in Streamlit
    st.plotly_chart(fig)
    
    