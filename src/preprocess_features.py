import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

def load_data(file_path="../output/final_dataset_corrected.csv"):
    df = pd.read_csv(file_path, parse_dates=['Enacted_Date', 'Effective_Date'])
    return df

def create_region_feature(df):
    region_mapping = {
        'Northeast': ['CT', 'MA', 'ME', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA'],
        'South': ['DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV', 'DC', 'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'OK', 'TX'],
        'Midwest': ['IL', 'IN', 'IA', 'KS', 'MI', 'MN', 'MO', 'NE', 'ND', 'OH', 'SD', 'WI'],
        'West': ['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'OR', 'UT', 'WA', 'WY', 'CA']
    }
    
    df['Region'] = df['LocationAbbr'].map(
        {state: region for region, states in region_mapping.items() for state in states}
    )
    return df

def create_features(df):
    first_policy_date = df.groupby(['MeasureDesc', 'ProvisionDesc'])[
        'Enacted_Date'].min().reset_index()
    
    first_policy_date.rename(columns={'Enacted_Date': 'First_Known_Date'}, inplace=True)
    df = pd.merge(df, first_policy_date, on=['MeasureDesc', 'ProvisionDesc'], how='left')
    
    df['Years_Since_First_Adoption'] = (df['Enacted_Date'] - 
                                       df['First_Known_Date']).dt.days / 365.25

    df['Time_To_Implementation'] = (df['Effective_Date'] - df['Enacted_Date']).dt.days

    df['Policy_Type'] = df['MeasureDesc'] + " - " + df['ProvisionDesc']
    df['Is_Restaurant_Policy'] = df['MeasureDesc'].str.contains('Restaurant').astype(int)
    df['Is_Worksite_Policy'] = df['MeasureDesc'].str.contains('Worksite').astype(int)
    df['Is_Ban_Policy'] = df['ProvisionValue'].str.contains('Ban').astype(int)
    
    return df

def preprocess_data(df, target_col='Copied_From_Neighbor'):
    non_contiguous = ['PR', 'HI', 'GU', 'VI', 'MP', 'AS', 'MH', 'PW', 'MH', 'AK']
    df = df[~df['LocationAbbr'].isin(non_contiguous)]

    df = df.dropna(subset=[target_col])

    df = create_region_feature(df)
    df = create_features(df)

    categorical_features = ['Region', 'MeasureDesc']
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    
    encoded_features = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(
        encoded_features, 
        columns=encoder.get_feature_names_out(),
        index=df.index
    )
    
    df = pd.concat([df, encoded_df], axis=1)
    
    return df, encoder

def prepare_classification_data(df):
    X = df[[
        'Neighbor_Adoption_Rate', 'Total_Neighbors', 'No_Neighbors_Already_Implemented',
        'Years_Since_First_Adoption', 'Year', 'Time_To_Implementation',
        'Is_Restaurant_Policy', 'Is_Worksite_Policy', 'Is_Ban_Policy'
    ]]
    
    cat_columns = [col for col in df.columns if col.startswith('Region_') or col.startswith('MeasureDesc_')]
    X = pd.concat([X, df[cat_columns]], axis=1)
    
    y = df['Copied_From_Neighbor']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

def prepare_regression_data(df):
    df_reg = df[df['Copied_From_Neighbor'] == True].copy()
    
    X = df_reg[[
        'Neighbor_Adoption_Rate', 'Total_Neighbors', 'No_Neighbors_Already_Implemented',
        'Years_Since_First_Adoption', 'Year', 'Time_To_Implementation',
        'Is_Restaurant_Policy', 'Is_Worksite_Policy', 'Is_Ban_Policy'
    ]]

    cat_columns = [col for col in df_reg.columns if col.startswith('Region_') or col.startswith('MeasureDesc_')]
    X = pd.concat([X, df_reg[cat_columns]], axis=1)
    
    y = df_reg['Adoption_Lag']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

def main():
    print("Start preprocessing")
    df = load_data()
    df, encoder = preprocess_data(df)

    os.makedirs('../output/preprocessed', exist_ok=True)
    df.to_csv('../output/preprocessed/model_ready_data.csv', index=False)

    X_train, X_test, y_train, y_test, clf_scaler, feature_names = prepare_classification_data(df)

    np.save('../output/preprocessed/X_train_clf.npy', X_train)
    np.save('../output/preprocessed/X_test_clf.npy', X_test)
    np.save('../output/preprocessed/y_train_clf.npy', y_train)
    np.save('../output/preprocessed/y_test_clf.npy', y_test)

    X_train_reg, X_test_reg, y_train_reg, y_test_reg, reg_scaler, _ = prepare_regression_data(df)

    np.save('../output/preprocessed/X_train_reg.npy', X_train_reg)
    np.save('../output/preprocessed/X_test_reg.npy', X_test_reg)
    np.save('../output/preprocessed/y_train_reg.npy', y_train_reg)
    np.save('../output/preprocessed/y_test_reg.npy', y_test_reg)
    
    print("End Preprocessing")
    print(f"Classification samples: {len(y_train) + len(y_test)}")
    print(f"Regression samples: {len(y_train_reg) + len(y_test_reg)}")
    
    pd.Series(feature_names).to_csv('../output/preprocessed/feature_names.csv', index=False)

if __name__ == "__main__":
    main()
