import pandas as pd
import json

cleaned_data_file = pd.read_csv('C:/Users/ritvi/Code/DMA/smoke/Tracking-Tobacco-Law-to-Save-Lives/data/cleaned_file.csv')
state_mapping = cleaned_data_file[['LocationAbbr', 'LocationDesc']].drop_duplicates()

with open('C:/Users/ritvi/Code/DMA/smoke/Tracking-Tobacco-Law-to-Save-Lives/data/adjacent_us_states.json') as f:
    neighbors_json = json.load(f)

adjacency_list = []

for state_abbr, neighbor_abbrs in neighbors_json.items():
    state_full = state_mapping[state_mapping['LocationAbbr'] == state_abbr]['LocationDesc'].values
    if len(state_full) == 0:
        continue
    state_full = state_full[0]

    for neighbor_abbr in neighbor_abbrs:
        neighbor_full = state_mapping[state_mapping['LocationAbbr'] == neighbor_abbr]['LocationDesc'].values
        if len(neighbor_full) == 0:
            continue
        neighbor_full = neighbor_full[0]
        
        adjacency_list.append({
            'State_Abbr': state_abbr,
            'State_Name': state_full,
            'Neighbor_Abbr': neighbor_abbr,
            'Neighbor_Name': neighbor_full
        })

adjacency_df = pd.DataFrame(adjacency_list)
adjacency_df.to_csv('C:/Users/ritvi/Code/DMA/smoke/Tracking-Tobacco-Law-to-Save-Lives/output/state_neighbors_adjacency.csv', index=False)

print("Adjacency CSV created!")
