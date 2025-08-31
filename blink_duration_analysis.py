import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import statsmodels.api as sm
import os 

def blink_duration_analysis(filepath): 
    df = pd.read_csv(filepath)

    def category(score):
        if 0 <= score <= 30: 
            return 'Low'
        elif 30 < score < 70: 
            return 'Mid'
        elif 70 <= score <= 100: 
            return 'High'
        
    df['Stress Group'] = df['Stress Score'].apply(category)

    #calculate the mean and std for each group 
    mean_std_result = {}
    for group in ['High', 'Mid', 'Low']: 
        #for mean
        mean_std_result[f'mean for {group} (mean)'] = df[df['Stress Group'] == group]['blink duration (mean)'].mean()
        mean_std_result[f'std for {group} (mean)'] = df[df['Stress Group'] == group]['blink duration (mean)'].std()

        #for max
        mean_std_result[f'mean for {group} (max)'] = df[df['Stress Group'] == group]['blink duration (max)'].mean()
        mean_std_result[f'std for {group} (max)'] = df[df['Stress Group'] == group]['blink duration (max)'].std()


    #Shapiro Wilk Test 
    stat_mean, p_value_shapiro_mean = stats.shapiro(df['blink duration (mean)'])
    stat_max, p_value_shapiro_max = stats.shapiro(df['blink duration (max)'])

    normal_mean = 'Normally Distributed' if p_value_shapiro_mean > 0.05 else 'Not normally distributed'
    normal_max = 'Normally Distributed' if p_value_shapiro_max > 0.05 else 'Not normally distributed'

    #Pearson's Correlation Coefficient (Linear)
    r_value_mean, p_value_pearson_mean = stats.pearsonr(df['blink duration (mean)'], df['Stress Score'])
    r_value_max, p_value_pearson_max = stats.pearsonr(df['blink duration (max)'], df['Stress Score'])

    sig_pear_mean = 'Significant' if p_value_pearson_mean < 0.05 else 'Not Significant'
    sig_pear_max = 'Significant' if p_value_pearson_max < 0.05 else 'Not Significant'

    #Spearman's Rank Correlation Coefficient (Monotonic)
    rho_mean, p_value_spearman_mean = stats.spearmanr(df['blink duration (mean)'], df['Stress Score'])
    rho_max, p_value_spearman_max = stats.spearmanr(df['blink duration (max)'], df['Stress Score'])

    sig_spear_mean = 'Significant' if p_value_spearman_mean < 0.05 else 'Not Significant'
    sig_spear_max = 'Significant' if p_value_spearman_max < 0.05 else 'Not Significant'

    #Kendall Tau Test
    tau_mean, p_value_ken_mean = stats.kendalltau(df['blink duration (mean)'], df['Stress Score'])
    tau_max, p_value_ken_max = stats.kendalltau(df['blink duration (max)'], df['Stress Score'])

    sig_tau_mean = 'Significant' if p_value_ken_mean < 0.05 else 'Not Significant'
    sig_tau_max = 'Significant' if p_value_ken_max < 0.05 else 'Not Significant'

    #Scatter Plot with Robust Linear Regression Line(robust to outliers)
    x = df['blink duration (mean)']
    y = df['Stress Score']

    x_with_constant = sm.add_constant(x)

    robust_linear_model_mean = sm.RLM(y, x_with_constant)
    robust_linear_cal_mean = robust_linear_model_mean.fit()

    home = os.path.expanduser('~')
    output_folder = os.path.join(home, 'Downloads', 'blink_project', 'Plots_Blink_Duration')
    os.makedirs(output_folder, exist_ok=True)
    output_file_mean = os.path.join(output_folder, 'scatterplot_rob_mean.png')
    summary_folder = os.path.join(home, 'Downloads', 'blink_project', 'summary')
    summary_file = os.path.join(summary_folder, 'blink_duration_summary.csv')


    plt.figure()
    plt.scatter(x, y, color='blue')
    plt.plot(x, robust_linear_cal_mean.predict(x_with_constant), color='red')
    plt.title('Scatter Plot with Robust Linear Regression Line')
    plt.xlabel('blink duration (mean)')
    plt.ylabel('stress score')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file_mean)
    plt.close()

    a = df['blink duration (max)']

    a_with_constant = sm.add_constant(a)

    robust_linear_model_max = sm.RLM(y, a_with_constant)
    robust_linear_cal_max = robust_linear_model_max.fit()

    output_file_max = os.path.join(output_folder, 'scatterplot_rob_max.png')

    plt.figure()
    plt.scatter(a, y, color='blue')
    plt.plot(a, robust_linear_cal_max.predict(a_with_constant), color='red')
    plt.title('Scatter Plot with Robust Linear Regression Line')
    plt.xlabel('blink duration (max)')
    plt.ylabel('stress score')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file_max)
    plt.close()

    #ストレスグループ別に相関を計算する
    group_wise_result = {}
    for group in ['High', 'Mid', 'Low']:
        duration_mean = df[df['Stress Group'] == group]['blink duration (mean)'] 
        duration_max = df[df['Stress Group'] == group]['blink duration (max)'] 
        stress_data = df[df['Stress Group'] == group]['Stress Score']

        #pearson 
        r_value_group_mean, p_val_pear_group_mean = stats.pearsonr(duration_mean, stress_data) 
        r_value_group_max, p_val_pear_group_max = stats.pearsonr(duration_max, stress_data)

        sig_pear_group_mean = 'Significant' if p_val_pear_group_mean < 0.05 else 'Not Significant'
        sig_pear_group_max = 'Significant' if p_val_pear_group_max < 0.05 else 'Not Significant'

        #spearman
        rho_group_mean, p_val_spear_group_mean = stats.spearmanr(duration_mean, stress_data)
        rho_group_max, p_val_spear_group_max = stats.spearmanr(duration_max, stress_data)
        sig_spear_group_mean = 'Significant' if p_val_spear_group_mean < 0.05 else 'Not Significant'
        sig_spear_group_max = 'Significant' if p_val_spear_group_max < 0.05 else 'Not Significant'

        #kendall tau
        tau_group_mean, p_val_ken_group_mean = stats.kendalltau(duration_mean, stress_data)
        tau_group_max, p_val_ken_group_max = stats.kendalltau(duration_max, stress_data)

        sig_ken_group_mean = 'Significant' if p_val_ken_group_mean < 0.05 else 'Not Significant'
        sig_ken_group_max = 'Significant' if p_val_ken_group_max < 0.05 else 'Not Significant'

        group_wise_result.update({
            f'Pearson r ({group}, mean)': r_value_group_mean,
            f'Pearson p ({group}, mean)': p_val_pear_group_mean,
            f'Pearson significance ({group}, mean)': sig_pear_group_mean,

            f'Pearson r ({group}, max)': r_value_group_max,
            f'Pearson p ({group}, max)': p_val_pear_group_max,
            f'Pearson significance ({group}, max)': sig_pear_group_max,

            f'Spearman rho ({group}, mean)': rho_group_mean,
            f'Spearman p ({group}, mean)': p_val_spear_group_mean,
            f'Spearman significance ({group}, mean)': sig_spear_group_mean,

            f'Spearman rho ({group}, max)': rho_group_max,
            f'Spearman p ({group}, max)': p_val_spear_group_max,
            f'Spearman significance ({group}, max)': sig_spear_group_max,

            f'Kendall tau ({group}, mean)': tau_group_mean,
            f'Kendall p ({group}, mean)': p_val_ken_group_mean,
            f'Kendall significance ({group}, mean)': sig_ken_group_mean,

            f'Kendall tau ({group}, max)': tau_group_max,
            f'Kendall p ({group}, max)': p_val_ken_group_max,
            f'Kendall significance ({group}, max)': sig_ken_group_max,
        })

    #dictionary
    result = {
        'Shapiro Wilk Test Stat (mean)': stat_mean, 
        'P Value for Shapiro (mean)': p_value_shapiro_mean, 
        'Is data normally distributed (mean)?': normal_mean, 
        'Shapiro Wilk Test Stat (max)': stat_max, 
        'P Value for Shapiro (max)': p_value_shapiro_max, 
        'Is data normally distributed? (max)': normal_max, 
        'Pearson Stat (mean)': r_value_mean, 
        'P value for Pearson (mean)': p_value_pearson_mean, 
        'Conclusion (pearson mean)': sig_pear_mean, 
        'Pearson Stat (max)': r_value_max, 
        'P value for Pearson(max)': p_value_pearson_max, 
        'Conclusion (pearson max)': sig_pear_max,
        'Spearman stat (mean)': rho_mean, 
        'P value for spearman (mean)': p_value_spearman_mean, 
        'Conclusion (spearman mean)': sig_spear_mean, 
        'Spearman stat (max)': rho_max, 
        'P value for spearman (max)': p_value_spearman_max, 
        'Conclusion (spearman max)': sig_spear_max,  
        'kendall stat (mean)': tau_mean, 
        'P value for kendall (mean)': p_value_ken_mean, 
        'Conclusion (kendall mean)': sig_tau_mean,
        'kendall stat (max)': tau_max, 
        'P value for kendall (max)': p_value_ken_max, 
        'Conclusion (kendall max)': sig_tau_max, 
    }
    result.update(mean_std_result)
    result.update(group_wise_result)


    pd.DataFrame([result]).to_csv(summary_file, mode='w', index=False)

    print(f'The summary file has been successfully produced: {os.path.basename(summary_file)}')

if __name__ == '__main__': 
    filepath = 'summary.csv'
    blink_duration_analysis(filepath)
