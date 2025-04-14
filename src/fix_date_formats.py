import pandas as pd
from datetime import datetime

def fix_date_format(date_str):
    if pd.isna(date_str):
        return date_str
        
    formats_to_try = [
        ('%d/%m/%Y', 'dayfirst'),  # DD/MM/YYYY
        ('%m/%d/%Y', 'monthfirst'), # MM/DD/YYYY
        ('%Y-%m-%d', 'iso'),        # ISO format
        ('%d-%b-%y', 'dayabbr'),    # 01-Jan-23
        ('%b-%d-%Y', 'monthabbr'),  # Jan-01-2023
    ]
    
    for fmt, fmt_type in formats_to_try:
        try:
            dt = datetime.strptime(date_str, fmt)
            
            if fmt_type == 'monthfirst' and dt.month > 12:
                continue  # Skip invalid month
            if fmt_type == 'dayfirst' and dt.day > 31:
                continue  # Skip invalid day
                
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            continue

    try:
        return pd.to_datetime(date_str, errors='raise').strftime('%Y-%m-%d')
    except:
        return pd.NaT

def clean_dates(input_file, output_file):

    df = pd.read_csv(input_file)
    original_dates = {
        'Enacted_Date': df['Enacted_Date'].copy(),
        'Effective_Date': df['Effective_Date'].copy()
    }

    date_columns = ['Enacted_Date', 'Effective_Date']
    for col in date_columns:
        df[col] = df[col].apply(fix_date_format)

    change_log = []
    for idx in df.index:
        for col in date_columns:
            original = original_dates[col][idx]
            new = df.at[idx, col]
            if original != new and pd.notna(new):
                change_log.append({
                    'Row_Index': idx,
                    'Column': col,
                    'Original_Value': original,
                    'New_Value': new
                })

    df.to_csv(output_file, index=False)

    log_df = pd.DataFrame(change_log)
    log_df.to_csv('../log/date_format_changes_log.csv', index=False)
    
    print(f"Date fixed data saved to {output_file}")
    print(f"Format changes logged to date_format_changes_log.csv")

if __name__ == "__main__":
    input_path = "../data/cleaned_file.csv"
    output_path = "../data/date_fixed_cleaned_file.csv"
    clean_dates(input_path, output_path)
