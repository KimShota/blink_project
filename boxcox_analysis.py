import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import os
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau, zscore, boxcox
import dcor 
from sklearn.feature_selection import mutual_info_regression 
from statsmodels.formula.api import ols
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

def boxcox_analysis(filepath): 
    df = pd.read_csv(filepath)

    #remove outliers with zscore 
    df = df[(np.abs(zscore(df[['blink duration (mean)', 'total_blinks', 'burst counts', 'inter-blink intervals (mean)', 'Stress Score']])) < 3).all(axis=1)]

    features = {
        'blink duration (mean)': 'Blink Duration', 
        'total_blinks': 'Blink Frequency', 
        'burst counts': 'Burst Flurries', 
        'inter-blink intervals (mean)': 'Inter-blink intervals', 
    }

    results = []
    interaction_result = []
    rf_resutl = []

    #folders/files
    home = os.path.expanduser('~')
    bc_folder = os.path.join(home, 'Downloads', 'blink_project', 'Boxcox_plots')
    os.makedirs(bc_folder, exist_ok=True)
    bc_folder = os.path.join(home, 'Downloads', 'blink_project', 'boxcox')
    os.makedirs(bc_folder, exist_ok=True)
    summary_file = os.path.join(bc_folder, 'boxcox_summary.csv')
    interaction_file = os.path.join(bc_folder, 'interaction_analysis.csv')
    rf_file = os.path.join(bc_folder, 'randomforest_analysis.csv')

    for feature, label in features.items(): 
        new_val, _ = boxcox(df[feature] + 0.01) #in case there is a value being 0
        df[f'{feature} (coxbox ver)'] = new_val

        a = df[f'{feature} (coxbox ver)']
        b = df['Stress Score']

        #scatter plot with a glob regression line 
        plt.figure()
        sns.regplot(x=a, y=b, color='blue')
        plt.title('Scatter plot with a global reg line')
        plt.xlabel(f'{label}')
        plt.ylabel('Stress Score')
        plt.grid(True)
        plt.savefig(os.path.join(bc_folder, f'Global {feature}'))
        plt.close()

        plt.figure()
        sns.regplot(x=a, y=b, color='blue', lowess=True)
        plt.title('Scatter plot with a local reg line')
        plt.xlabel(f'{label}')
        plt.ylabel('Stress Score')
        plt.grid(True)
        plt.savefig(os.path.join(bc_folder, f'Local {feature}'))
        plt.close()

        #Pearson's correlation coefficient 
        r_value, p_pear = stats.pearsonr(a, b)
        sig_pear = 'Significant' if p_pear <= 0.1 else 'Not Significant'

        #Spearman's rank correlation coefficient
        rho, p_spear = stats.spearmanr(a, b)
        sig_spear = 'Significant' if p_spear <= 0.1 else 'Not Significant'

        #Kendalltau
        tau, p_tau = stats.kendalltau(a, b)
        sig_tau = 'Significant' if p_tau <= 0.1 else 'Not Significant'

        #Distance correlation 
        dcor_val = dcor.distance_correlation(a.values.reshape(-1, 1), b.values.reshape(-1, 1))

        #Mutual information 
        mi = mutual_info_regression(a.values.reshape(-1, 1), b)

        results.append({
            'Metric': label,
            'Feature Name': feature,
            'Pearson Correlation': r_value,
            'Pearson p-value': p_pear,
            'Pearson Significance': sig_pear,
            'Spearman Correlation': rho,
            'Spearman p-value': p_spear,
            'Spearman Significance': sig_spear,
            'Kendall Tau': tau,
            'Kendall p-value': p_tau,
            'Kendall Significance': sig_tau,
            'Distance Correlation': dcor_val,
            'Mutual Information': mi[0]
        })

        #Interaction effect analysis
        for new_feature in features: 
            if new_feature == feature: 
                continue; 
            
            interaction = f'interaction_{feature}_{new_feature}'
            df[interaction] = df[feature] * df[new_feature]

            formula = f'Q("Stress Score") ~ Q("{feature}") + Q("{new_feature}") + Q("{interaction}")'
            model = ols(formula, data=df).fit()

            interaction_result.append({
                'Variable a': f'{feature}', 
                'Variable b': f'{new_feature}', 
                'Interaction name': interaction, 
                'R sqaured': model.rsquared, 
                'P Values for interaction': model.pvalues.get(interaction, np.nan), 
                'Significance': 'Significant' if model.pvalues.get(interaction, 1) <= 0.1 else 'Not Significant'
            })

    pd.DataFrame(results).to_csv(summary_file, mode='w', index=False)
    pd.DataFrame(interaction_result).to_csv(interaction_file, mode='w', index=False)

    #Random Forest Regressor 
    X = df[['blink duration (mean)', 'total_blinks', 'burst counts', 'inter-blink intervals (mean)']]
    y = df['Stress Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, random_state=0)
    model.fit(X_train, y_train)

    y_prediction = model.predict(X_test)

    importance = model.feature_importances_

    mse = mean_squared_error(y_test, y_prediction)
    r2score = r2_score(y_test, y_prediction)

    cols = X.columns; 

    rfr = {
        'Mean Squared Error': mse, 
        'R Squared Score': r2score
    }

    for col, imp in zip(cols, importance): 
        rfr[f'Importance of {col}'] = imp

    pd.DataFrame([rfr]).to_csv(rf_file, mode='w', index=False)

if __name__ == '__main__': 
    filepath = 'summary.csv'
    boxcox_analysis(filepath)
