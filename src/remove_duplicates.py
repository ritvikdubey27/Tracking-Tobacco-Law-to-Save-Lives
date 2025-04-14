import pandas as pd

def remove_duplicates(input_file, output_file):

    df = pd.read_csv(input_file, parse_dates=['Enacted_Date'])

    key_columns = [
        'LocationAbbr', 
        'MeasureDesc', 
        'ProvisionDesc', 
        'ProvisionValue', 
        'Enacted_Date'
    ]

    df = df.sort_values('Year')

    df = df.drop_duplicates(subset=key_columns, keep='first')

    df.to_csv(output_file, index=False)
    print(f"Removed duplicates. Cleaned data saved to {output_file}")

if __name__ == "__main__":
    input_file = "../data/date_fixed_cleaned_file.csv"
    output_file = "../output/deduplicated_file.csv"
    remove_duplicates(input_file, output_file)
