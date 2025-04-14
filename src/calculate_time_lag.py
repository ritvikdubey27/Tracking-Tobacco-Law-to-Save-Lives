import pandas as pd

def calculate_time_lag():
    input_file = "../output/deduplicated_file.csv"
    output_file = "../output/time_lag_cleaned_file.csv"
    log_file = "../log/time_lag_errors.csv"
    
    df = pd.read_csv(input_file)

    df['Enacted_Date'] = pd.to_datetime(
        df['Enacted_Date'], 
        errors='coerce', 
        infer_datetime_format=True
    )
    df['Effective_Date'] = pd.to_datetime(
        df['Effective_Date'], 
        errors='coerce', 
        infer_datetime_format=True
    )

    date_errors = df[
        df['Enacted_Date'].isna() | 
        df['Effective_Date'].isna()
    ].copy()

    valid_df = df.dropna(subset=['Enacted_Date', 'Effective_Date'])
    valid_df['Time_Lag'] = (
        valid_df['Effective_Date'] - valid_df['Enacted_Date']
    ).dt.days

    valid_df['Negative_Lag'] = valid_df['Time_Lag'] < 0

    valid_df.to_csv(output_file, index=False)
    date_errors.to_csv(log_file, index=False)

    print(f"""
    Successfully processed {len(valid_df)} rows.
    Found {len(date_errors)} rows with invalid/missing dates.
    Time Lag saved to: {output_file}
    Date errors logged to: {log_file}
    """)

if __name__ == "__main__":
    calculate_time_lag()
