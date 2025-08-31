import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats 
from scipy.stats import pearsonr 
from scipy.stats import spearmanr
import statsmodels.api as sm
import os
import glob

def mismatch_analyzer(filepath):
    df = pd.read_csv(filepath)

    def category(score): 
        if 0 <= score <= 30: 
            return 'Low'
        elif 30 < score < 70: 
            return 'Mid'
        elif 70 < score <= 100: 
            return 'High'
    
    df['Stress Group'] = df['Stress Score'].apply(category)

    mismatch_logcounts = np.log(df['mismatch Counts'] + 1)
    df['log mismatch'] = np.log(df['mismatch Counts'] + 1)

    #calculate spearman's rank correlation 
    rho, p_value_spearman = stats.spearmanr(df['mismatch Counts'], df['Stress Score'])
    significance_spearman = 'Significant' if p_value_spearman < 0.05 else 'Not Significant'

    #calculate pearson's correlation coefficient 
    r_value, p_value_pearson = stats.pearsonr(mismatch_logcounts, df['Stress Score'])
    significance_pearson = 'Significant' if p_value_pearson < 0.05 else 'Not Significant'

    home = os.path.expanduser('~')
    output_folder = os.path.join(home, 'Downloads', 'blink_project', 'Plots_Mismatch')
    os.makedirs(output_folder, exist_ok=True)
    output_file_robust = os.path.join(output_folder, 'scatter_plot_robust_regline.png')
    summary_folder = os.path.join(home, 'Downloads', 'blink_project', 'summary')
    summary_file = os.path.join(summary_folder, 'mismatch_counts_analysis.csv')
    # output_file_globreg = os.path.join(output_folder, 'scatterplot_globreg.png')
    # output_file_locreg = os.path.join(output_folder, 'scatterplot_locreg.png')

    #Scatter plot with a regression line and a local regression line 
    # plt.figure()
    # sns.regplot(x='log mismatch', y='Stress Score', data=df)
    # plt.title('mismatch VS stress score with global reg')
    # plt.xlabel('mismatch')
    # plt.ylabel('stress score')
    # plt.savefig(output_file_globreg)
    # plt.close()

    # plt.figure()
    # sns.regplot(x='log mismatch', y='Stress Score', data=df, lowess=True)
    # plt.title('mismatch VS stress score with local reg')
    # plt.xlabel('mismatch')
    # plt.ylabel('stress score')
    # plt.savefig(output_file_locreg)
    # plt.close()

    #Robust Linear Regression Line 
    x = df['log mismatch']
    y = df['Stress Score']
    x_with_constant = sm.add_constant(x)
    robust_linear_model = sm.RLM(y, x_with_constant)
    robust_linear_regression = robust_linear_model.fit()

    plt.scatter(x, y, color='blue')
    plt.plot(x, robust_linear_regression.predict(x_with_constant), color='red', label='Robust Regression Line')
    plt.title('Scatter Plot with Robust Regression Line')
    plt.xlabel('log version of mismatch counts')
    plt.ylabel('stress score')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file_robust)
    plt.close()

    #Welch's t-test
    high_stress_group = np.log(df[df['Stress Group'] == 'High']['mismatch Counts'] + 1)
    low_stress_group = np.log(df[df['Stress Group'] == 'Low']['mismatch Counts'] + 1)

    t_stat, p_value_ttest = stats.ttest_ind(high_stress_group, low_stress_group, equal_var=False)

    significance_welch = 'Significant' if p_value_ttest < 0.05 else 'Not significant'

    #shapiro wilk's test to check if our data is normally distributed (bell-shaped like) or not
    shapiro_result = {}
    outliers = []
    for group in ['High', 'Mid', 'Low']: 
        group_data = pd.to_numeric(df[df['Stress Group'] == group]['mismatch Counts'], errors='coerce').dropna()
        if len(group_data) >= 3: #this is because shapiro wilk test requires at least 3 samples 
            stat, p_value_shapiro = stats.shapiro(group_data)
            shapiro_result[f'{group} Shapiro Statistics'] = stat
            shapiro_result[f'{group} P Value for shapiro'] = p_value_shapiro
            significance_shapiro = 'Significant' if p_value_shapiro < 0.05 else 'Not significant'
            shapiro_result[f'{group} Significance'] = significance_shapiro
            
            #identify outliers in each group 
            Q1 = np.percentile(group_data, 25)
            Q3 = np.percentile(group_data, 75)

            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers.extend([x for x in group_data if x < lower_bound or x > upper_bound])

    result = {
        'Spearmans rank correlation coefficient': rho, 
        'P Value for spearman': p_value_spearman, 
        'Significance for spearman': significance_spearman, 
        'Pearsons correlation coefficient': r_value,  
        'P Value for pearson': p_value_pearson, 
        'Significance for pearson': significance_pearson,
        'Welchs T stats': t_stat, 
        'P Value for Welch': p_value_ttest, 
        'Significance for welch': significance_welch, 
        'Outliers': outliers
    }
    result.update(shapiro_result)

    pd.DataFrame([result]).to_csv(summary_file, mode='w', index=False)

    print('The csv file and plots have been successfully produced')

if __name__ == '__main__': 
    filepath = 'summary.csv'
    mismatch_analyzer(filepath)
