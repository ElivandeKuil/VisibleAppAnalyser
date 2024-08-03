import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests, ccf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.utils.dataframe import dataframe_to_rows
import io
import warnings
import streamlit as st

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Function to convert data to DataFrame
def convert_to_dataframe(data, column_names):
    df = pd.DataFrame(data, columns=column_names)
    df['date'] = pd.to_datetime(df['date'])
    for col in df.columns:
        if col != 'date':
            df[col] = pd.to_numeric(df[col].str.strip(), errors='coerce')
    df.set_index('date', inplace=True)
    return df



# Function for aligned correlation
def aligned_correlation(x, y):
    aligned = pd.concat([x, y], axis=1).dropna()
    if len(aligned) < 2 or aligned.iloc[:, 0].nunique() < 2 or aligned.iloc[:, 1].nunique() < 2:
        return np.nan
    return aligned.iloc[:, 0].corr(aligned.iloc[:, 1])

# Function for lagged correlation
def lagged_correlation(x, y, max_lag):
    results = []
    for lag in range(max_lag + 1):
        if lag == 0:
            corr = aligned_correlation(x, y)
        else:
            corr = aligned_correlation(x.shift(lag), y)
        results.append((lag, corr))
    return pd.DataFrame(results, columns=['lag', 'correlation']).set_index('lag')

# Function for modified Granger causality
def modified_granger_causality(x, y, max_lag):
    aligned = pd.concat([x, y], axis=1).dropna()
    if len(aligned) <= max_lag + 1:
        return np.nan, np.nan
    try:
        result = grangercausalitytests(aligned, maxlag=max_lag, verbose=False)
        p_values = [result[i+1][0]['ssr_ftest'][1] for i in range(max_lag)]
        min_p_value = min(p_values)
        best_lag = p_values.index(min_p_value) + 1
        return min_p_value, best_lag
    except:
        return np.nan, np.nan

# New function for Time Series Decomposition
def perform_time_series_decomposition(series, period=7):
    try:
        result = seasonal_decompose(series.dropna(), model='additive', period=period)
        return result
    except:
        return None

# New function for Cross-Correlation Function
def perform_ccf(x, y, max_lag):
    x = x.dropna()
    y = y.dropna()
    if len(x) != len(y):
        min_len = min(len(x), len(y))
        x = x[-min_len:]
        y = y[-min_len:]
    ccf_result = ccf(x, y, adjusted=False)
    return pd.Series(ccf_result[:max_lag], index=range(max_lag))

def symptom_correlation_analysis(data, symptom_columns):
    # Get the intersection of symptom_columns and available columns in the data
    available_symptom_columns = list(set(symptom_columns) & set(data.columns))
    
    if not available_symptom_columns:
        print("No symptom columns available for correlation analysis.")
        return None, None

    symptom_data = data[available_symptom_columns]
    
    # Check for columns with no variation
    constant_columns = symptom_data.columns[symptom_data.nunique() <= 1]
    if len(constant_columns) > 0:
        print(f"Columns with no variation: {', '.join(constant_columns)}")
        symptom_data = symptom_data.drop(columns=constant_columns)
        available_symptom_columns = list(set(available_symptom_columns) - set(constant_columns))

    if symptom_data.empty or len(available_symptom_columns) < 2:
        print("Insufficient data for correlation analysis after removing constant columns.")
        return None, None

    corr_matrix = symptom_data.corr(method='spearman')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Symptom Correlation Heatmap')
    plt.tight_layout()
    
    return corr_matrix, available_symptom_columns

def process_file(file):
    # reading CSV file
    data = file
    
    substring = 'Funcap'
    filter = data['tracker_category'].str.contains(substring)
    filter2 = data['tracker_category'].str.contains('Note')
    filtered_df = data[~filter]
    filtered_df = filtered_df[~filter2]

    unique_trackers = filtered_df['tracker_name'].unique()

    np_data = filtered_df.to_numpy()
    np_data = np.flip(np_data, 0)

    first = True;

    processed_data = []
    date = [None] * (len(unique_trackers) + 1)
    last_date = ''

    for row in np_data:
        if first == False and last_date != row[0]:
            processed_data.append(date)
            date = [None] * (len(unique_trackers) + 1)
            last_date = row[0]
        
        if first == True:
            last_date = row[0]
            first = False

        date[0] = row[0]
        
        
        index = np.where(unique_trackers==row[1])[0][0]
        date[index + 1] = row[3]
    processed_data.append(date)

    unique_trackers = np.insert(unique_trackers, 0, 'date', axis=0)

    # Assuming you have your data and column_names defined
    data_df = convert_to_dataframe(processed_data, unique_trackers)

    # Identify symptom and activity columns
    symptom_columns = []  # Adjust these based on your actual symptom names
    activity_columns = []  # Adjust these based on your actual activity names

    activity_categories = ["Cognitive", "Emotional", "Experience", "Medication", "Social", "Physical", "Menstrual", "Measurement"]
    boolean_categories = ["Experience", "Medication"]

    boolean_columns = []

    for name in unique_trackers:
        if name != 'date':
            filtered = filtered_df[filtered_df['tracker_name'] == name]
            unique_value = filtered["tracker_category"].unique()[0]
            if unique_value in activity_categories:
                activity_columns.append(name)
            else:
                symptom_columns.append(name)
            
            if unique_value in boolean_categories:
                boolean_columns.append(name)


    non_boolean_columns = [col for col in data_df.columns if col not in boolean_columns and col != 'date']

    # Filter out columns with insufficient data
    min_unique_values = 2
    min_non_null_count = 10
    valid_columns = data_df.columns[
        (data_df.nunique() >= min_unique_values) & 
        (data_df.count() >= min_non_null_count)
    ].tolist()
    data_df_filtered = data_df[valid_columns]
    # Perform correlation analysis
    correlation_results = {}
    for symptom in symptom_columns:
        for activity in activity_columns:
            if symptom in valid_columns and activity in valid_columns:
                corr = aligned_correlation(data_df_filtered[symptom], data_df_filtered[activity])
                if not np.isnan(corr):
                    correlation_results[f"{symptom} vs {activity}"] = corr


    # Perform lagged correlation analysis
    max_lag = 3
    lagged_correlation_results = {}
    for symptom in symptom_columns:
        for activity in activity_columns:
            if symptom in valid_columns and activity in valid_columns:
                lagged_corr = lagged_correlation(data_df_filtered[symptom], data_df_filtered[activity], max_lag)
                if not lagged_corr['correlation'].isna().all():
                    max_corr = lagged_corr['correlation'].abs().idxmax()
                    lagged_correlation_results[f"{symptom} vs {activity}"] = (max_corr, lagged_corr.loc[max_corr, 'correlation'])


    # Perform Granger causality analysis
    granger_results = {}
    for symptom in symptom_columns:
        for activity in activity_columns:
            if symptom in valid_columns and activity in valid_columns:
                p_value, best_lag = modified_granger_causality(data_df_filtered[activity], data_df_filtered[symptom], max_lag)
                if not np.isnan(p_value):
                    granger_results[f"{activity} -> {symptom}"] = (p_value, best_lag)



    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    corr_matrix = pd.DataFrame(index=symptom_columns, columns=activity_columns)
    for symptom in symptom_columns:
        for activity in activity_columns:
            key = f"{symptom} vs {activity}"
            corr_matrix.loc[symptom, activity] = correlation_results.get(key, np.nan)

    # Convert to float and replace any remaining non-numeric values with NaN
    corr_matrix = corr_matrix.astype(float)

    # Create mask for NaN values
    mask = np.isnan(corr_matrix)

    # Create Excel workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Results and Interpretation"

    # Add data description
    ws.append(["Data Description"])
    ws.append(["Total number of days:", len(data_df)])
    ws.append(["Date range:", f"{data_df.index.min()} to {data_df.index.max()}"])
    ws.append(["Number of variables:", len(data_df.columns)])
    ws.append([""])

    # Add data completeness information
    ws.append(["Data Completeness"])
    ws.append(["Interpretation: Higher percentages indicate more complete data for that variable. Variables with low completeness may have less reliable results."])
    completeness = data_df.notna().sum() / len(data_df) * 100
    for col, comp in completeness.sort_values(ascending=False).items():
        ws.append([col, f"{comp:.2f}%"])
    ws.append([""])

    # Add correlation results
    ws.append(["Correlation Analysis"])
    ws.append(["Interpretation: Values range from -1 to 1. Closer to 1: Strong positive correlation. Closer to -1: Strong negative correlation. Close to 0: Weak or no correlation."])
    ws.append(["In this context, a negative correlations means a beneficial effect (higher activity means lower symptoms)"])
    ws.append(["Symptom", "Activity", "Correlation"])
    for pair, corr in sorted(correlation_results.items(), key=lambda x: abs(x[1]), reverse=True):
        symptom, activity = pair.split(" vs ")
        ws.append([symptom, activity, corr])
    ws.append([""])

    # Add lagged correlation results
    ws.append(["Lagged Correlation Analysis"])
    ws.append(["Interpretation: 'Best Lag' indicates the number of days offset with the strongest correlation."])
    ws.append(["Positive lag: Activity precedes symptom. Negative lag: Symptom precedes activity. "])
    ws.append(["In this context, a negative correlations means a beneficial effect (higher activity means lower symptoms)"])
    ws.append(["Symptom", "Activity", "Best Lag", "Correlation"])
    for pair, (lag, corr) in sorted(lagged_correlation_results.items(), key=lambda x: abs(x[1][1]), reverse=True):
        symptom, activity = pair.split(" vs ")
        ws.append([symptom, activity, lag, corr])
    ws.append([""])

    # Add Granger causality results
    ws.append(["Granger Causality Analysis"])
    ws.append(["Interpretation: P-value < 0.05 suggests the activity may help predict the symptom."])
    ws.append(["'Best Lag' indicates the most significant time offset. So when p is smaller than 0.05, the occurrence of the activity makes it significantly more likely for the symptom to worsen"])
    ws.append(["Activity", "Symptom", "P-value", "Best Lag"])
    for pair, (p_value, lag) in sorted(granger_results.items(), key=lambda x: x[1][0]):
        activity, symptom = pair.split(" -> ")
        ws.append([activity, symptom, p_value, lag])
    ws.append([""])

    # Perform Symptom Correlation Analysis
    ws.append(["Symptom Correlation Analysis"])
    ws.append(["Interpretation: Shows how different symptoms are correlated with each other. Values range from -1 (strong negative correlation) to 1 (strong positive correlation). 0 indicates no correlation."])

    symptom_corr_matrix, available_symptoms = symptom_correlation_analysis(data_df_filtered, symptom_columns)

    if symptom_corr_matrix is not None:
        # Save the heatmap
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img = Image(img_buffer)
        ws.add_image(img, f'A{ws.max_row + 5}')
        plt.close()

        ws.append(["Note: Strong positive correlations might indicate symptoms that tend to occur together or increase/decrease together."])
        ws.append(["Strong negative correlations might indicate symptoms that tend to have opposite patterns."])
        ws.append(["This analysis can help identify symptom clusters or potential underlying factors affecting multiple symptoms."])
    else:
        ws.append(["Insufficient data for symptom correlation analysis."])
        ws.append(["Please check the console output for more information about data issues."])



    # Save the Excel file
    wb.save('C:/Users/eliva/OneDrive/Documents/GitHub/LC/symptom_activity_analysis.xlsx')

    print("Analysis complete. Results saved in 'symptom_activity_analysis.xlsx'.")

    return wb

st.title("Excel File Processor")
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    result = process_file(df)
    st.download_button(
        label="Download processed file",
        data=result.to_excel(index=False),
        file_name="processed_file.xlsx"
    )