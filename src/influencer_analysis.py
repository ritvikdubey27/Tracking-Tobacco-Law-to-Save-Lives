import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(file_path="../output/final_dataset_corrected.csv"):
    df = pd.read_csv(file_path, parse_dates=['Enacted_Date', 'Effective_Date'])
    return df

def identify_influencer_states(df):
    copied_policies = df[df['Copied_From_Neighbor'] == True].copy()

    first_implementer_counts = copied_policies['Neighbor_Who_Implemented_First'].value_counts()
    influencer_df = pd.DataFrame({
        'State': first_implementer_counts.index,
        'Times_First_Implemented': first_implementer_counts.values
    })

    avg_lags = {}
    for state in influencer_df['State']:
        avg_lags[state] = copied_policies[copied_policies['Neighbor_Who_Implemented_First'] == state]['Adoption_Lag'].mean()
    
    influencer_df['Avg_Adoption_Lag'] = influencer_df['State'].map(avg_lags)

    influencer_df = influencer_df.sort_values('Times_First_Implemented', ascending=False)
    
    return influencer_df

def analyze_by_policy_type(df):
    policy_categories = {
        'Restaurant_Policies': df['MeasureDesc'].str.contains('Restaurant'),
        'Worksite_Policies': df['MeasureDesc'].str.contains('Worksite'),
        'Public_Space_Policies': df['MeasureDesc'].str.contains('Indoor Air'),
        'Day_Care_Policies': df['MeasureDesc'].str.contains('Day Care')
    }

    policy_analysis = {}
    for policy_name, mask in policy_categories.items():
        subset = df[mask]
        if len(subset) > 0:
            policy_analysis[policy_name] = {
                'Total_Policies': len(subset),
                'Copied_Policies': subset['Copied_From_Neighbor'].sum(),
                'Copy_Rate': subset['Copied_From_Neighbor'].mean(),
                'Avg_Adoption_Lag': subset[subset['Copied_From_Neighbor'] == True]['Adoption_Lag'].mean(),
                'Median_Adoption_Lag': subset[subset['Copied_From_Neighbor'] == True]['Adoption_Lag'].median()
            }

    policy_df = pd.DataFrame(policy_analysis).T
    policy_df = policy_df.reset_index().rename(columns={'index': 'Policy_Type'})
    
    return policy_df

def visualize_influencers(influencer_df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Times_First_Implemented', y='State', data=influencer_df.head(15))
    plt.title('Top 15 Influencer States', fontsize=14)
    plt.xlabel('Number of Times State Was First Implementer')
    plt.tight_layout()
    plt.savefig('../output/figures/top_influencer_states.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='Avg_Adoption_Lag', 
        y='Times_First_Implemented', 
        data=influencer_df,
        size='Times_First_Implemented',
        sizes=(20, 200),
        alpha=0.7
    )

    for i, row in influencer_df.head(10).iterrows():
        plt.annotate(
            row['State'],
            xy=(row['Avg_Adoption_Lag'], row['Times_First_Implemented']),
            xytext=(5, 0),
            textcoords='offset points'
        )
    
    plt.title('Influencer States: Frequency vs. Average Adoption Lag', fontsize=14)
    plt.xlabel('Average Adoption Lag (Days)')
    plt.ylabel('Times First Implemented')
    plt.tight_layout()
    plt.savefig('../output/figures/influencer_lag_comparison.png', dpi=300, bbox_inches='tight')

def main():
    os.makedirs('../output/figures', exist_ok=True)

    print("Loading data...")
    df = load_data()

    print("Identifying influencer states...")
    influencer_df = identify_influencer_states(df)
    influencer_df.to_csv('../output/influencer_states_analysis.csv', index=False)

    print("Analyzing adoption patterns by policy type...")
    policy_df = analyze_by_policy_type(df)
    policy_df.to_csv('../output/policy_type_analysis.csv', index=False)

    print("Creating visualizations...")
    visualize_influencers(influencer_df)
    
    print("Analysis complete! Results saved to output directory.")

if __name__ == "__main__":
    main()
