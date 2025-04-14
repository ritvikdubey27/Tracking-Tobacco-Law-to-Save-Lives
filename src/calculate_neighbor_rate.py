import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

NON_CONTIGUOUS_STATES = ['PR', 'HI', 'GU', 'VI', 'MP', 'AS', 'MH', 'PW', 'MH', 'AK']
COPYCAT_THRESHOLDS = {
    'min_adoption_rate': 0.3,
    'max_adoption_lag': 3650
}

def load_adjacency_data(file_path):
    df = pd.read_csv(file_path)
    return df.groupby('State_Abbr')['Neighbor_Abbr'].apply(list).to_dict()

def create_state_mapping(df):
    return df[['LocationAbbr', 'LocationDesc']].drop_duplicates().set_index('LocationAbbr')['LocationDesc'].to_dict()

def precompute_earliest_dates(df):
    earliest_dates = defaultdict(dict)
    for _, row in df.iterrows():
        key = (row['MeasureDesc'], row['ProvisionDesc'], row['ProvisionValue'])
        state = row['LocationAbbr']
        current_date = earliest_dates[state].get(key, pd.NaT)
        if pd.isna(current_date) or row['Enacted_Date'] < current_date:
            earliest_dates[state][key] = row['Enacted_Date']
    return earliest_dates

def calculate_neighbor_metrics(row, neighbor_dict, earliest_dates, state_names):
    state = row['LocationAbbr']
    key = (row['MeasureDesc'], row['ProvisionDesc'], row['ProvisionValue'])
    enact_date = row['Enacted_Date']

    if state in NON_CONTIGUOUS_STATES or state not in neighbor_dict:
        return pd.Series([
            False,   # Copied_From_Neighbor
            0,       # Total_Neighbors
            0,       # No_Neighbors_Already_Implemented
            np.nan,  # Neighbor_Adoption_Rate
            [],      # Neighbors_Already_Implemented
            "",      # Neighbor_Who_Implemented_First
            np.nan,  # Adoption_Lag
            0.0      # Copy_Confidence_Score
        ])
    
    neighbors = neighbor_dict.get(state, [])
    adopted_neighbors = []
    neighbor_dates = []

    for neighbor_abbr in neighbors:
        neighbor_date = earliest_dates.get(neighbor_abbr, {}).get(key, pd.NaT)
        if not pd.isna(neighbor_date) and neighbor_date <= enact_date:
            adopted_neighbors.append(neighbor_abbr)
            neighbor_dates.append(neighbor_date)

    total_neighbors = len(neighbors)
    adopted_count = len(adopted_neighbors)
    rate = adopted_count / total_neighbors if total_neighbors else np.nan
    first_neighbor_abbr = adopted_neighbors[0] if adopted_neighbors else ""
    first_neighbor_date = min(neighbor_dates) if neighbor_dates else pd.NaT

    adoption_lag = (enact_date - first_neighbor_date).days if not pd.isna(first_neighbor_date) else np.nan

    is_copycat = False
    confidence = 0.0
    if not pd.isna(rate) and not pd.isna(adoption_lag):
        meets_rate = rate >= COPYCAT_THRESHOLDS['min_adoption_rate']
        meets_lag = adoption_lag <= COPYCAT_THRESHOLDS['max_adoption_lag']
        is_copycat = meets_rate and meets_lag

        rate_weight = min(rate / 0.8, 1.0)
        lag_weight = max(1 - (adoption_lag / 730), 0)  # Prevent negative weights
        confidence = (rate_weight * 0.7) + (lag_weight * 0.3)

    neighbor_list_full = [state_names.get(n, "") for n in adopted_neighbors]
    first_neighbor_full = state_names.get(first_neighbor_abbr, "")
    
    return pd.Series([
        is_copycat,
        total_neighbors,
        adopted_count,
        rate,
        neighbor_list_full,
        first_neighbor_full,
        adoption_lag,
        confidence
    ])

def main():
    input_file = "../output/time_lag_cleaned_file.csv"
    adjacency_file = "../output/state_neighbors_adjacency.csv"
    output_file = "../output/final_dataset_corrected.csv"
    
    df = pd.read_csv(input_file, parse_dates=['Enacted_Date', 'Effective_Date'])
    neighbor_dict = load_adjacency_data(adjacency_file)
    state_names = create_state_mapping(df)
    earliest_dates = precompute_earliest_dates(df)

    tqdm.pandas(desc="Analyzing Neighbor Influence")
    results = df.progress_apply(
        lambda row: calculate_neighbor_metrics(row, neighbor_dict, earliest_dates, state_names), 
        axis=1,
        result_type='expand'
    )

    df[[
        'Copied_From_Neighbor',
        'Total_Neighbors',
        'No_Neighbors_Already_Implemented',
        'Neighbor_Adoption_Rate',
        'Neighbors_Already_Implemented',
        'Neighbor_Who_Implemented_First',
        'Adoption_Lag',
        'Copy_Confidence_Score'
    ]] = results

    df.loc[df['LocationAbbr'].isin(NON_CONTIGUOUS_STATES), 'Neighbor_Adoption_Rate'] = np.nan
    df.loc[df['LocationAbbr'].isin(NON_CONTIGUOUS_STATES), 'Neighbors_Already_Implemented'] = df.loc[df['LocationAbbr'].isin(NON_CONTIGUOUS_STATES), 'Neighbors_Already_Implemented'].apply(lambda x: [])

    df.to_csv(output_file, index=False)
    print(f"Corrected dataset saved to {output_file}")

if __name__ == "__main__":
    main()
