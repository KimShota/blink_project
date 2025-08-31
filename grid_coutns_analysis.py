import glob 
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy import stats 

#main function
def grid_counts_per_blink_analysis(filepath): 
    df = pd.read_csv(filepath)

    def category(score): 
        if 0 <= score <= 30: 
            return 'Low'
        elif 30 < score <= 70: 
            return 'Mid'
        elif 70 < score <= 100: 
            return 'High'
    
    #create a new column in data frame called Stress Group and label each row with an appropriate category
    df['Stress Group'] = df['Stress Score'].apply(category)

    #grid counts per blink 
    #箱ひげ図
    home = os.path.expanduser('~')
    plot_folder = os.path.join(home, 'Downloads', 'blink_project', 'plots')
    os.makedirs(plot_folder, exist_ok=True)

    #Does the number of grid cells a person passes through per blink change across different stress scores?
    plt.figure()
    sns.boxplot(x='Stress Group', y='grid counts per blink (mean)', data=df) 
    plt.title("Distribution of grid counts per blink by stress group")
    plt.xlabel("Stress Group")
    plt.ylabel("Grid counts per blink (mean)")
    plt.savefig(os.path.join(plot_folder, f'{os.path.basename(filepath)}_boxplot.png'))
    plt.close()

    high_stress_group = df[df['Stress Group'] == 'High']['grid counts per blink (mean)']
    mid_stress_group = df[df['Stress Group'] == 'Mid']['grid counts per blink (mean)']
    low_stress_group = df[df['Stress Group'] == 'Low']['grid counts per blink (mean)']

    #mean for each group
    high_stress_mean = np.mean(high_stress_group)
    mid_stress_mean = np.mean(mid_stress_group)
    low_stress_mean = np.mean(low_stress_group)

    #standard deviation for each group 
    high_stress_std = np.std(high_stress_group)
    mid_stress_std = np.std(mid_stress_group)
    low_stress_std = np.std(low_stress_group)

    #Welch's t-test
    t_stats, p_value = stats.ttest_ind(high_stress_group, low_stress_group, equal_var=False)
    print(t_stats, p_value)

    conclusion = 'Significant' if p_value < 0.05 else 'Not significant'

    #ヒストグラム (How grid counts per blink are distributed)
    plt.figure()
    sns.histplot(df['grid counts per blink (mean)'], bins=20, kde=True, color='blue')
    plt.title('ヒストグラム')
    plt.xlabel('Grid Counts')
    plt.ylabel('frequency')
    plt.savefig(os.path.join(plot_folder, f'{os.path.basename(filepath)}_histogram.png'))
    plt.close()

    #散布図 (scatter plot with a regression line)
    plt.figure(figsize=(6, 4))
    #Do changes in blinking behavior affect a stress score?
    sns.regplot(x='grid counts per blink (mean)', y='Stress Score', data=df)
    plt.title("")
    plt.savefig(os.path.join(plot_folder, f'{os.path.basename(filepath)}_regression.png'))
    plt.close()

    result = {
        'Filename': os.path.basename(filepath), 
        'Mean for High Stress Group': high_stress_mean, 
        'Mean for Mid Stress Group': mid_stress_mean, 
        'Mean for Low Stress Group': low_stress_mean, 
        'Std for High Stress Group': high_stress_std, 
        'Std for Mid Stress Group': mid_stress_std,
        'Std for Low Stress Group': low_stress_std,
        "T statistics": t_stats, 
        "P value": p_value, 
        "How significant": conclusion
    }

    home = os.path.expanduser('~')
    summary_folder = os.path.join(home, 'Downloads', 'blink_project', 'summary')
    os.makedirs(summary_folder, exist_ok=True)
    output_file = os.path.join(summary_folder, 'grid_counts_per_blink_results.csv')
    pd.DataFrame([result]).to_csv(output_file, mode='w', index=False)

if __name__ == '__main__': 
    filepath = "summary.csv"
    grid_counts_per_blink_analysis(filepath)
