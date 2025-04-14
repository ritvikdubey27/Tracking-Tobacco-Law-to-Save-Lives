import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.express as px
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available")

def load_data(file_path="../output/final_dataset_corrected.csv"):
    print(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path, parse_dates=['Enacted_Date', 'Effective_Date'])
        print(f"Successfully loaded {len(df)} rows of data")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def create_region_mapping():
    return {
        'Northeast': ['CT', 'MA', 'ME', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA'],
        'South': ['DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV', 'DC', 'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'OK', 'TX'],
        'Midwest': ['IL', 'IN', 'IA', 'KS', 'MI', 'MN', 'MO', 'NE', 'ND', 'OH', 'SD', 'WI'],
        'West': ['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'OR', 'UT', 'WA', 'WY', 'CA']
    }

def add_region_column(df):
    region_mapping = create_region_mapping()
    state_to_region = {state: region for region, states in region_mapping.items() for state in states}

    df['Region'] = df['LocationAbbr'].map(state_to_region)
    return df

def filter_contiguous_states(df):
    non_contiguous = ['PR', 'HI', 'GU', 'VI', 'MP', 'AS', 'MH', 'PW', 'MH', 'AK']
    return df[~df['LocationAbbr'].isin(non_contiguous)]

def create_policy_spread_matplotlib(df, output_dir="../output/figures"):
    print("Creating policy spread visualization using matplotlib...")

    df = filter_contiguous_states(df)

    df = add_region_column(df)

    policy_types = [
        "Smokefree Indoor Air - Restaurants",
        "Government Worksites", 
        "Private Worksites",
        "Smokefree Indoor Air - Other Sites"
    ]
    
    policy_mapping = {
        "Smokefree Indoor Air - Restaurants": "Restaurants",
        "Government Worksites": "Government Worksites",
        "Private Worksites": "Private Worksites",
        "Smokefree Indoor Air - Other Sites": "Other Sites"
    }
    
    for policy_type in policy_types:
        print(f"  Processing: {policy_type}")

        if "-" in policy_type:
            base_type, subtype = policy_type.split(" - ", 1)
            policy_data = df[df['MeasureDesc'].str.contains(base_type, na=False)]
            if subtype != "Other Sites":
                policy_data = policy_data[policy_data['MeasureDesc'].str.contains(subtype, na=False)]
        else:
            policy_data = df[df['MeasureDesc'].str.contains(policy_type, na=False)]
        
        if len(policy_data) == 0:
            print(f"  No data found for {policy_type}, skipping")
            continue

        state_adoption = policy_data.groupby('LocationAbbr')['Enacted_Date'].min().reset_index()

        state_adoption['Year'] = pd.to_datetime(state_adoption['Enacted_Date']).dt.year

        state_adoption = state_adoption.sort_values('Year')

        plt.figure(figsize=(14, 8))
        plt.bar(state_adoption['LocationAbbr'], state_adoption['Year'], color='steelblue')
        plt.title(f'Year of Adoption of {policy_mapping.get(policy_type, policy_type)} Policies by State', fontsize=14)
        plt.xlabel('State', fontsize=12)
        plt.ylabel('Year of Adoption', fontsize=12)
        plt.xticks(rotation=90)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        safe_name = policy_type.replace(" ", "_").replace("-", "_").replace("–", "_")
        output_path = os.path.join(output_dir, f'policy_adoption_years_{safe_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved visualization to: {output_path}")
        plt.close()

        years_range = range(min(state_adoption['Year']), max(state_adoption['Year'])+1)
        adoption_counts = []
        
        for year in years_range:
            count = sum(state_adoption['Year'] <= year)
            adoption_counts.append(count)
            
        plt.figure(figsize=(12, 6))
        plt.plot(list(years_range), adoption_counts, marker='o', linewidth=2, color='darkblue')
        plt.title(f'Cumulative Adoption of {policy_mapping.get(policy_type, policy_type)} Policies (1975-2002)', fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Number of States with Policy Adopted', fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'policy_diffusion_timeline_{safe_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved diffusion timeline to: {output_path}")
        plt.close()

        state_adoption.to_csv(os.path.join(output_dir, f'policy_data_{safe_name}.csv'), index=False)

def create_enhanced_adoption_lag_plot(df, output_dir="../output/figures"):
    print("Creating enhanced adoption vs lag visualization...")

    scatter_data = df[
        (df['Copied_From_Neighbor'] == True) & 
        (~df['Adoption_Lag'].isna()) & 
        (~df['Neighbor_Adoption_Rate'].isna())
    ].copy()

    scatter_data = add_region_column(scatter_data)

    scatter_data = scatter_data.dropna(subset=['Region'])
    
    print(f"  Using {len(scatter_data)} copied policy instances for visualization")

    plt.figure(figsize=(12, 8))

    if len(scatter_data) == 0:
        print("  No valid data for adoption lag plot")
        return

    size_var = 'Copy_Confidence_Score' if 'Copy_Confidence_Score' in scatter_data.columns else None

    if size_var and not scatter_data[size_var].isna().all():
        scatter = sns.scatterplot(
            data=scatter_data,
            x='Neighbor_Adoption_Rate',
            y='Adoption_Lag',
            hue='Region',
            size=size_var,
            sizes=(20, 200),
            alpha=0.7
        )
    else:
        scatter = sns.scatterplot(
            data=scatter_data,
            x='Neighbor_Adoption_Rate',
            y='Adoption_Lag',
            hue='Region',
            alpha=0.7
        )

    for region, color in zip(scatter_data['Region'].unique(), sns.color_palette(n_colors=len(scatter_data['Region'].unique()))):
        region_data = scatter_data[scatter_data['Region'] == region]
        if len(region_data) > 5:
            sns.regplot(
                data=region_data,
                x='Neighbor_Adoption_Rate',
                y='Adoption_Lag',
                scatter=False,
                color=color,
                line_kws={'linestyle': '--'}
            )

    plt.title('Relationship Between Neighbor Adoption Rate and Adoption Lag', fontsize=14)
    plt.xlabel('Neighbor Adoption Rate', fontsize=12)
    plt.ylabel('Adoption Lag (Days)', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.annotate(
        "States tend to adopt policies faster when\nmore of their neighbors have already adopted",
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        backgroundcolor='white',
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'enhanced_adoption_rate_vs_lag.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved adoption lag plot to: {output_path}")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=scatter_data, x='Region', y='Adoption_Lag', palette='viridis')
    plt.title('Adoption Lag by Region', fontsize=14)
    plt.xlabel('Region', fontsize=12)
    plt.ylabel('Adoption Lag (Days)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'adoption_lag_by_region.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved regional adoption lag comparison to: {output_path}")
    plt.close()

def create_policy_type_comparison(df, output_dir="../output/figures"):
    """Create visualization comparing different policy types"""
    print("Creating policy type comparison visualization...")

    df = filter_contiguous_states(df)

    df['PolicyCategory'] = 'Other'
    df.loc[df['MeasureDesc'].str.contains('Restaurant', na=False), 'PolicyCategory'] = 'Restaurant'
    df.loc[df['MeasureDesc'].str.contains('Worksite', na=False), 'PolicyCategory'] = 'Worksite'
    df.loc[df['MeasureDesc'].str.contains('Day Care', na=False), 'PolicyCategory'] = 'Day Care'
    df.loc[df['MeasureDesc'].str.contains('Hotel', na=False), 'PolicyCategory'] = 'Hotel/Motel'
    df.loc[df['MeasureDesc'].str.contains('Indoor Air', na=False), 'PolicyCategory'] = 'Indoor Air'

    policy_copy_rates = df.groupby('PolicyCategory')['Copied_From_Neighbor'].mean().reset_index()
    policy_copy_rates = policy_copy_rates.sort_values('Copied_From_Neighbor', ascending=False)

    copied_policies = df[df['Copied_From_Neighbor'] == True]
    if len(copied_policies) > 0:
        policy_lag = copied_policies.groupby('PolicyCategory')['Adoption_Lag'].mean().reset_index()
        policy_lag = policy_lag.sort_values('Adoption_Lag')

        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            policy_copy_rates['PolicyCategory'], 
            policy_copy_rates['Copied_From_Neighbor'],
            color='cornflowerblue'
        )

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f"{height:.1%}",
                ha='center', 
                fontsize=10
            )
            
        plt.title('Proportion of Policies Copied from Neighbors by Policy Type', fontsize=14)
        plt.xlabel('Policy Type', fontsize=12)
        plt.ylabel('Proportion Copied', fontsize=12)
        plt.ylim(0, max(policy_copy_rates['Copied_From_Neighbor']) * 1.2)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'policy_type_copy_rates.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved policy copy rates to: {output_path}")
        plt.close()

        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            policy_lag['PolicyCategory'], 
            policy_lag['Adoption_Lag'],
            color='teal'
        )

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 50,
                f"{int(height)} days",
                ha='center', 
                fontsize=10
            )
            
        plt.title('Average Adoption Lag by Policy Type', fontsize=14)
        plt.xlabel('Policy Type', fontsize=12)
        plt.ylabel('Average Days Until Adoption', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'policy_type_adoption_lag.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved policy adoption lag comparison to: {output_path}")
        plt.close()

def create_influencer_states_analysis(df, output_dir="../output/figures"):
    print("Creating influencer states analysis...")

    df = filter_contiguous_states(df)
    copied_policies = df[df['Copied_From_Neighbor'] == True].copy()
    
    if len(copied_policies) == 0:
        print("  No copied policies found for influencer analysis")
        return

    if 'Neighbor_Who_Implemented_First' not in copied_policies.columns:
        print("  'Neighbor_Who_Implemented_First' column not found")
        return

    copied_policies['First_Implementer'] = copied_policies['Neighbor_Who_Implemented_First'].astype(str)
    copied_policies['First_Implementer'] = copied_policies['First_Implementer'].str.replace('[', '').str.replace(']', '').str.replace("'", "")

    implementer_counts = copied_policies['First_Implementer'].value_counts().reset_index()
    implementer_counts.columns = ['State', 'Times_First_Implemented']

    implementer_counts = implementer_counts[implementer_counts['State'].str.len() <= 2]

    top_implementers = implementer_counts.sort_values('Times_First_Implemented', ascending=False).head(15)

    plt.figure(figsize=(12, 8))
    bars = plt.barh(
        top_implementers['State'][::-1], 
        top_implementers['Times_First_Implemented'][::-1],
        color='darkblue'
    )

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.5,
            bar.get_y() + bar.get_height()/2.,
            f"{int(width)}",
            va='center', 
            fontsize=10
        )
        
    plt.title('Top Influencer States in Policy Diffusion', fontsize=14)
    plt.xlabel('Number of Times State Was First Implementer', fontsize=12)
    plt.ylabel('State', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'top_influencer_states.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved influencer states analysis to: {output_path}")
    plt.close()

    implementer_counts.to_csv(os.path.join(output_dir, 'influencer_states_data.csv'), index=False)

def create_regional_adoption_patterns(df, output_dir="../output/figures"):
    print("Creating regional adoption patterns visualization...")

    df = filter_contiguous_states(df)
    df = add_region_column(df)
    df = df.dropna(subset=['Region'])

    region_copy_rates = df.groupby('Region')['Copied_From_Neighbor'].mean().reset_index()
    region_copy_rates = region_copy_rates.sort_values('Copied_From_Neighbor', ascending=False)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        region_copy_rates['Region'], 
        region_copy_rates['Copied_From_Neighbor'],
        color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f"{height:.1%}",
            ha='center', 
            fontsize=10
        )
        
    plt.title('Regional Differences in Policy Copying Behavior', fontsize=14)
    plt.xlabel('Region', fontsize=12)
    plt.ylabel('Proportion of Policies Copied from Neighbors', fontsize=12)
    plt.ylim(0, max(region_copy_rates['Copied_From_Neighbor']) * 1.2)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'regional_adoption_patterns.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved regional adoption patterns to: {output_path}")
    plt.close()

    if 'Year' in df.columns:
        yearly_region = df.groupby(['Year', 'Region'])['Copied_From_Neighbor'].mean().reset_index()

        plt.figure(figsize=(12, 6))
        
        for region in yearly_region['Region'].unique():
            region_data = yearly_region[yearly_region['Region'] == region]
            plt.plot(
                region_data['Year'], 
                region_data['Copied_From_Neighbor'],
                marker='o',
                label=region
            )
            
        plt.title('Evolution of Policy Copying by Region (1995-2002)', fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Proportion of Policies Copied', fontsize=12)
        plt.legend(title='Region')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'regional_adoption_over_time.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved regional adoption timeline to: {output_path}")
        plt.close()

def main():
    output_dir = "../output/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("Loading data...")
        df = load_data()

        create_policy_spread_matplotlib(df, output_dir)
        create_enhanced_adoption_lag_plot(df, output_dir)
        create_policy_type_comparison(df, output_dir)
        create_influencer_states_analysis(df, output_dir)
        create_regional_adoption_patterns(df, output_dir)
        
        print("Visualizations complete! All figures saved to output/figures directory.")
        
    except Exception as e:
        print(f"Error in visualization process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
