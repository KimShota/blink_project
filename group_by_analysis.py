import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau, zscore
import dcor
from sklearn.feature_selection import mutual_info_regression 
import os 
from statsmodels.formula.api import ols
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

def group_by_analysis(filepath): 
    df = pd.read_csv(filepath)

    Q1 = 46
    Q3 = 61

    #remove outliers using zscore
    df = df[(np.abs(zscore(df[['blink duration (mean)', 'total_blinks', 'burst counts', 'inter-blink intervals (mean)', 'Stress Score']])) < 3).all(axis=1)]

    # cols = ['blink duration (mean)', 'total_blinks', 'burst counts', 'inter-blink intervals (mean)', 'Stress Score']
    # for col in cols: 
    #     df[col] = pd.to_numeric(df[col], errors='coerce')

    # def winsorization(series, lower=0.01, upper=0.99): 
    #     return series.clip(series.quantile(lower), series.quantile(upper))
    
    # df['blink duration (mean)'] = winsorization(df['blink duration (mean)'])
    # df['total_blinks'] = winsorization(df['total_blinks'])
    # df['burst counts'] = winsorization(df['burst counts'])
    # df['inter-blink intervals (mean)'] = winsorization(df['inter-blink intervals (mean)'])

    def category(score): 
        if 0 <= score <= Q1: 
            return 'Low'
        elif Q1 < score < Q3: 
            return 'Mid'
        elif Q3 <= score <= 100: 
            return 'High'
        
    df['Stress Group'] = df['Stress Score'].apply(category)

    high_num = df[df['Stress Group'] == 'High'].shape[0]
    mid_num = df[df['Stress Group'] == 'Mid'].shape[0]
    low_num = df[df['Stress Group'] == 'Low'].shape[0]

    print(f'Number of people with high brain fatigue: {high_num}')
    print(f'Number of people with mid brain fatigue: {mid_num}')
    print(f'Number of people with low brain fatigue: {low_num}')

    #folders/files
    home = os.path.expanduser('~')
    group_folder = os.path.join(home, 'Downloads', 'blink_project', 'Group_By_Analysis_Plots')
    os.makedirs(group_folder, exist_ok=True)
    summary_folder = os.path.join(home, 'Downloads', 'blink_project', 'summary')
    os.makedirs(summary_folder, exist_ok=True)
    summary_file = os.path.join(summary_folder, 'group_by_analysis_result.csv')
    interaction_file = os.path.join(summary_folder, 'interaction_analysis.csv')
    rfr_file = os.path.join(summary_folder, 'random_forest_regression.csv')
    pca_file = os.path.join(summary_folder, 'pca_analysis.csv')
    pca_pear_file = os.path.join(summary_folder, 'pca_pearson_analysis.csv')

    # #Blink duration vs Stress Score (by group)
    # features = {
    #     'blink duration (mean)': 'Blink Duration', 
    #     'total_blinks': 'Blink Frequency', 
    #     'burst counts': 'Blink Flurries', 
    #     'inter-blink intervals (mean)': 'Inter-blink intervals', 
    # }

    for c in ['blink duration (mean)', 'total_blinks', 'burst counts', 'inter-blink intervals (mean)', 'Stress Score']: 
        df[c + '_log'] = np.log1p(df[c])
    
    features = {
        'blink duration (mean)_log': 'Blink Duration (log)', 
        'total_blinks_log': 'Blink Frequency (log)', 
        'burst counts_log': 'Blink Flurries (log)', 
        'inter-blink intervals (mean)_log': 'Inter-blink intervals (log)', 
    }

    result = []
    for col, label in features.items(): 
        for group in ['High', 'Mid', 'Low']: 
            x = df[df['Stress Group'] == group][col]
            y = df[df['Stress Group'] == group]['Stress Score']

            #scatter plot with a global regression line 
            plt.figure()
            sns.regplot(x=x, y=y, color='blue')
            plt.title(f'scatter plot with glob reg line {group}')
            plt.xlabel(f'{col} in {group} brain fatigue group')
            plt.ylabel(f'Stress score in {group}')
            plt.grid(True)
            plt.savefig(os.path.join(group_folder, f'{label}_{group}_global.png'))
            plt.close()

            #scatter plot with a local regression line
            plt.figure()
            sns.regplot(x=x, y=y, color='blue', lowess=True)
            plt.title(f'scatter plot with loc reg line {group}')
            plt.xlabel(f'{col} in {group} brain fatigue group')
            plt.ylabel(f'Stress score in {group}')
            plt.grid(True)
            plt.savefig(os.path.join(group_folder, f'{label}_{group}_local.png'))
            plt.close()

            #pearson 
            r_value, p_pear = stats.pearsonr(x, y)
            sig_pear = 'Significant' if p_pear <= 0.1 else 'Not Significant'

            #spearman 
            rho, p_spear = stats.spearmanr(x, y)
            sig_spear = 'Significant' if p_spear <= 0.1 else 'Not Significant'

            #kendalltau 
            tau, p_tau = stats.kendalltau(x, y)
            sig_tau = 'Significant' if p_tau <= 0.1 else 'Not Significant'

            #distance correlation 
            dcor_value = dcor.distance_correlation(x.values.reshape(-1, 1), y.values.reshape(-1, 1))

            #mutual information
            mi = mutual_info_regression(x.values.reshape(-1, 1), y)[0]

            result.append({
                'Feature': label, 
                'Group': group, 
                'Pearson R Value': r_value, 
                'Pearson P Value': p_pear, 
                'Significance for pearson': sig_pear,  
                'Spearman rho': rho, 
                'Spearman P Value': p_spear,
                'Significance for spearman': sig_spear, 
                'Kendall Tau': tau, 
                'Kendall P Value': p_tau, 
                'Significance for kendalltau': sig_tau, 
                'Distance Correlation': dcor_value, 
                'Mutual Information': mi
            })

    rdf = pd.DataFrame(result)

    for col in ['Pearson P Value', 'Spearman P Value', 'Kendall P Value']: 
        mask = rdf[col].notna()
        adjust = np.full(rdf.shape[0], np.nan, dtype=float)

        if mask.sum() > 0: 
            reject, corrected_P_val, _, _ = multipletests(rdf.loc[mask, col].values, method='fdr_bh')
            adjust[mask] = corrected_P_val
            rdf[f'{col} adjusted by Benjamini'] = adjust
            rdf[f'Significance of adjusted P Value for {col}'] = np.where(mask, reject, np.nan)

    rdf.to_csv(summary_file, mode='w', index=False)

    #interaction analysis 
    keys_list = list(features.keys())

    interaction_result = []
    for i in range(len(keys_list)):
        for j in range(i + 1, len(keys_list)): 
            a = keys_list[i]
            b = keys_list[j]
            label_a = features[a]
            label_b = features[b]
            interaction = f'interaction_{label_a}_{label_b}'.replace(' ', '_')

            df[interaction] = df[a] * df[b] #create a new column in df for each interaction 
            
            formula = f'Q("Stress Score") ~ Q("{a}") + Q("{b}") + Q("{interaction}")'
            model = ols(formula, data=df).fit()

            interaction_result.append({
                'Variable A': label_a,
                'Variable B': label_b, 
                'Interaction name': interaction, 
                'R Squared': model.rsquared,
                'P values for interaction': model.pvalues.get(interaction, np.nan), 
                'Significant': 'Yes' if model.pvalues.get(interaction, 1) < 0.05 else 'No'
            })

    pd.DataFrame(interaction_result).to_csv(interaction_file, mode='w', index=False)

    #random forest regression 
    X = df[['blink duration (mean)', 'total_blinks', 'burst counts', 'inter-blink intervals (mean)']]
    y = df['Stress Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=300, random_state=0) #create a model for random forest regressor 
    rf.fit(X_train, y_train) #calculate the average of all

    y_prediction = rf.predict(X_test) #test the trained random forest regression model

    importance = rf.feature_importances_

    mse = mean_squared_error(y_test, y_prediction)
    r2 = r2_score(y_test, y_prediction)

    rfr = {
        'Mean Squared Error': mse, 
        'R Squared Score': r2, 
    }

    cols = X.columns
    
    for column, impo in zip(cols, importance):
        rfr[f'Importance of {column}'] = impo

    pd.DataFrame([rfr]).to_csv(rfr_file, mode='w', index=False)

    #Principal Component Analysis 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)

    pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pc_df['Stress Score'] = df['Stress Score']

    pc_df = pc_df.dropna(subset=['PC1', 'PC2', 'Stress Score'])

    pc1_r_val, p_pear_pc1 = stats.pearsonr(pc_df['PC1'], pc_df['Stress Score'])
    pc2_r_val, p_pear_pc2 = stats.pearsonr(pc_df['PC2'], pc_df['Stress Score'])

    print(f'PC1 R value: {pc1_r_val}')
    print(f'PC2 R value: {pc2_r_val}')

    pca_pear_result = {
        'Pearson for PC1': pc1_r_val, 
        'Significance for PC1': 'Significant' if p_pear_pc1 <= 0.01 else 'Not Significant', 
        'Pearson for PC2': pc2_r_val, 
        'Significance for PC"': 'Significant' if p_pear_pc2 <= 0.01 else 'Not Significant'
    }

    pd.DataFrame([pca_pear_result]).to_csv(pca_pear_file, mode='w', index=False)

    components = ['blink duration (mean)', 'total_blinks', 'burst counts', 'inter-blink intervals (mean)']

    pca_result = pd.DataFrame(
        pca.components_, 
        columns=components, 
        index=['PC1', 'PC2']
    )

    pca_result.to_csv(pca_file, mode='w')

if __name__ == '__main__': 
    filepath = 'summary.csv'
    group_by_analysis(filepath)
