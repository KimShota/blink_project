import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import os
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau, zscore
import dcor 
from sklearn.feature_selection import mutual_info_regression  


def parametric_analysis(filepath): 
    df = pd.read_csv(filepath)

    #remove outliers 
    # df = df[(np.abs(zscore(df[['blink duration (mean)', 'total_blinks', 'burst counts', 'inter-blink intervals (mean)', 'Stress Score']])) < 3).all(axis=1)]

    cols = ['blink duration (mean)', 'total_blinks', 'burst counts', 'inter-blink intervals (mean)', 'Stress Score']
    for col in cols: 
        df[col] = pd.to_numeric(df[col], errors='coerce')

    def winsorization(s, lower=0.01, upper=0.99): 
        return s.clip(s.quantile(lower), s.quantile(upper))
    
    df['blink duration (mean)'] = winsorization(df['blink duration (mean)'])
    df['total_blinks'] = winsorization(df['total_blinks'])
    df['burst counts'] = winsorization(df['burst counts'])
    df['inter-blink intervals (mean)'] = winsorization(df['inter-blink intervals (mean)'])

    dfc = df[cols].dropna().copy()

    duration  = dfc['blink duration (mean)']
    frequency = dfc['total_blinks']
    flurries  = dfc['burst counts']
    interval  = dfc['inter-blink intervals (mean)']
    score     = dfc['Stress Score']

    #folders/files
    home = os.path.expanduser('~')
    duration_folder = os.path.join(home, 'Downloads', 'blink_project', 'Blink Duration plots')
    os.makedirs(duration_folder, exist_ok=True)
    duration_glob = os.path.join(duration_folder, 'Scatter Plot with global reg line')
    duration_loc = os.path.join(duration_folder, 'Scatter Plot with local reg line')
    
    fre_folder = os.path.join(home, 'Downloads', 'blink_project', 'Blink Frequency plots')
    os.makedirs(fre_folder, exist_ok=True)
    fre_glob = os.path.join(fre_folder, 'Scatter Plot with global reg line')
    fre_loc = os.path.join(fre_folder, 'Scatter Plot with local reg line')

    flur_folder = os.path.join(home, 'Downloads', 'blink_project', 'Blink Flurries plots')
    os.makedirs(flur_folder, exist_ok=True)
    flur_glob = os.path.join(flur_folder, 'Scatter Plot with global reg line')
    flur_loc = os.path.join(flur_folder, 'Scatter Plot with local reg line')

    int_folder = os.path.join(home, 'Downloads', 'blink_project', 'Inter-blink interval plots')
    os.makedirs(int_folder, exist_ok=True)
    int_glob = os.path.join(int_folder, 'Scatter Plot with global reg line')
    int_loc = os.path.join(int_folder, 'Scatter Plot with local reg line')

    summary_folder = os.path.join(home, 'Downloads', 'blink_project', 'summary')
    os.makedirs(summary_folder, exist_ok=True)
    summary_file = os.path.join(summary_folder, 'parametric_analysis_result.csv')

    #Blink Duration (mean) vs Stress Score
    plt.figure()
    sns.regplot(x=duration, y=score, color='blue')
    plt.title('Scatter plot with global reg line')
    plt.xlabel('Blink Duration (mean)')
    plt.ylabel('Stress Score')
    plt.grid(True)
    plt.savefig(duration_glob)
    plt.close()

    plt.figure()
    sns.regplot(x=duration, y=score, color='blue', lowess=True)
    plt.title('Scatter plot with local reg line')
    plt.xlabel('Blink Duration (mean)')
    plt.ylabel('Stress Score')
    plt.grid(True)
    plt.savefig(duration_loc)
    plt.close()

    r_value, p_val_pear = stats.pearsonr(duration, score)
    sig_pear = 'Significant' if p_val_pear <= 0.1 else 'not significant'

    rho, p_val_spear = stats.spearmanr(duration, score)
    sig_spear = 'Significant' if p_val_spear <= 0.1 else 'not significant'

    tau, p_val_tau = stats.kendalltau(duration, score)
    sig_tau = 'Significant' if p_val_tau <= 0.1 else 'not significant'

    dcor_dur = dcor.distance_correlation(duration.values.reshape(-1, 1), score.values.reshape(-1, 1))

    mi_dur = mutual_info_regression(duration.values.reshape(-1, 1), score)

    #Blink Frequency vs Stress Score
    plt.figure()
    sns.regplot(x=frequency, y=score, color='blue')
    plt.title('Scatter plot with global reg line')
    plt.xlabel('Blink Frequency')
    plt.ylabel('Stress Score')
    plt.grid(True)
    plt.savefig(fre_glob)
    plt.close()

    plt.figure()
    sns.regplot(x=frequency, y=score, color='blue', lowess=True)
    plt.title('Scatter plot with local reg line')
    plt.xlabel('Blink Frequency')
    plt.ylabel('Stress Score')
    plt.grid(True)
    plt.savefig(fre_loc)
    plt.close()

    r_val_fre, p_pear_fre = stats.pearsonr(frequency, score)
    sig_pear_fre = 'Significant' if p_pear_fre <= 0.1 else 'Not Significant'

    rho_spear_fre, p_spear_fre = stats.spearmanr(frequency, score)
    sig_spear_fre = 'Significant' if p_spear_fre <= 0.1 else 'Not Significant'

    tau_fre, p_tau_fre = stats.kendalltau(frequency, score)
    sig_tau_fre = 'Significant' if p_tau_fre <= 0.1 else 'Not Significant'

    dcor_fre = dcor.distance_correlation(frequency.values.reshape(-1, 1), score.values.reshape(-1, 1))

    mi_fre = mutual_info_regression(frequency.values.reshape(-1, 1), score)

    #Blink Flurries vs Stress Score
    plt.figure()
    sns.regplot(x=flurries, y=score, color='blue')
    plt.title('Scatter plot with global reg line')
    plt.xlabel('Blink Flurries')
    plt.ylabel('Stress Score')
    plt.grid(True)
    plt.savefig(flur_glob)
    plt.close()

    plt.figure()
    sns.regplot(x=flurries, y=score, color='blue', lowess=True)
    plt.title('Scatter plot with local reg line')
    plt.xlabel('Blink Flurries')
    plt.ylabel('Stress Score')
    plt.grid(True)
    plt.savefig(flur_loc)
    plt.close()

    r_pear_flur, p_pear_flur = stats.pearsonr(flurries, score)
    sig_pear_flur = 'Significant' if p_pear_flur <= 0.1 else 'Not significant'

    rho_spear_flur, p_spear_flur = stats.spearmanr(flurries, score)
    sig_spear_flur = 'Significant' if p_spear_flur <= 0.1 else 'Not significant'

    tau_flur, p_tau_flur = stats.kendalltau(flurries, score)
    sig_tau_flur = 'Significant' if p_tau_flur <= 0.1 else 'Not Significant'

    dcor_flur = dcor.distance_correlation(flurries.values.reshape(-1, 1), score.values.reshape(-1, 1))

    mi_flur = mutual_info_regression(flurries.values.reshape(-1, 1), score)

    #inter-blink intervals 
    plt.figure()
    sns.regplot(x=interval, y=score, color='blue')
    plt.title('Scatter plot with a glob regression line')
    plt.xlabel('inter-blink intervals (mean)')
    plt.ylabel('stress score')
    plt.grid(True)
    plt.savefig(int_glob)
    plt.close()
    
    plt.figure()
    sns.regplot(x=interval, y=score, color='blue', lowess=True)
    plt.title('Scatter plot with a local regression line')
    plt.xlabel('inter-blink intervals (mean)')
    plt.ylabel('stress score')
    plt.grid(True)
    plt.savefig(int_loc)
    plt.close()

    r_pear_int, p_pear_int = stats.pearsonr(interval, score)
    sig_pear_int = 'Significant' if p_pear_int <= 0.1 else 'Not significant'

    rho_spear_inter, p_spear_inter = stats.spearmanr(interval, score)
    sig_spear_inter = 'Significant' if p_spear_inter <= 0.1 else 'Not significant'

    tau_inter, p_tau_inter = stats.kendalltau(interval, score)
    sig_tau_inter = 'Significant' if p_tau_inter <= 0.1 else 'Not Significant'

    dcor_inter = dcor.distance_correlation(interval.values.reshape(-1, 1), score.values.reshape(-1, 1))

    mi_inter = mutual_info_regression(interval.values.reshape(-1, 1), score)

    results = [
        {
            'Metric': 'Blink Duration',
            'Pearson Correlation': r_value,
            'Pearson p-value': p_val_pear,
            'Pearson Significance': sig_pear,
            'Spearman Correlation': rho,
            'Spearman p-value': p_val_spear,
            'Spearman Significance': sig_spear,
            'Kendall Tau': tau,
            'Kendall p-value': p_val_tau,
            'Kendall Significance': sig_tau,
            'Distance Correlation': dcor_dur,
            'Mutual Information': mi_dur[0]
        },
        {
            'Metric': 'Blink Frequency',
            'Pearson Correlation': r_val_fre,
            'Pearson p-value': p_pear_fre,
            'Pearson Significance': sig_pear_fre,
            'Spearman Correlation': rho_spear_fre,
            'Spearman p-value': p_spear_fre,
            'Spearman Significance': sig_spear_fre,
            'Kendall Tau': tau_fre,
            'Kendall p-value': p_tau_fre,
            'Kendall Significance': sig_tau_fre,
            'Distance Correlation': dcor_fre,
            'Mutual Information': mi_fre[0]
        },
        {
            'Metric': 'Blink Flurries',
            'Pearson Correlation': r_pear_flur,
            'Pearson p-value': p_pear_flur,
            'Pearson Significance': sig_pear_flur,
            'Spearman Correlation': rho_spear_flur,
            'Spearman p-value': p_spear_flur,
            'Spearman Significance': sig_spear_flur,
            'Kendall Tau': tau_flur,
            'Kendall p-value': p_tau_flur,
            'Kendall Significance': sig_tau_flur,
            'Distance Correlation': dcor_flur,
            'Mutual Information': mi_flur[0]
        },
        {
            'Metric': 'Inter-blink Interval',
            'Pearson Correlation': r_pear_int,
            'Pearson p-value': p_pear_int,
            'Pearson Significance': sig_pear_int,
            'Spearman Correlation': rho_spear_inter,
            'Spearman p-value': p_spear_inter,
            'Spearman Significance': sig_spear_inter,
            'Kendall Tau': tau_inter,
            'Kendall p-value': p_tau_inter,
            'Kendall Significance': sig_tau_inter,
            'Distance Correlation': dcor_inter,
            'Mutual Information': mi_inter[0]
        }
    ]

    pd.DataFrame(results).to_csv(summary_file, mode='w', index=False)
    print('The result has been stored into the csv file')

if __name__ == '__main__': 
    filepath = 'summary.csv'
    parametric_analysis(filepath)
