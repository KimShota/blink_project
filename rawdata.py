import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 

def rawdata(filepath):
    df = pd.read_csv(filepath)

    home = os.path.expanduser('~')
    plot_folder = os.path.join(home, 'Downloads', 'blink_project', 'plots_stress_score')
    os.makedirs(plot_folder, exist_ok=True)
    histo_file = os.path.join(plot_folder, 'histo_stress_score.png')
    box_file = os.path.join(plot_folder, 'box_stress_score.png')
    summary_folder = os.path.join(home, 'Downloads', 'blink_project', 'summary')
    summary_file = os.path.join(summary_folder, 'desc.csv')


    #ヒストグラム
    plt.figure()
    sns.histplot(x= 'Stress Score', data=df, bins=20, color='blue')
    plt.title('histogram')
    plt.xlabel('stress score')
    plt.ylabel('frequency')
    plt.grid(True)
    plt.savefig(histo_file)
    plt.close()

    #箱ひげ図
    sns.boxplot(x='Stress Score', data=df, color='blue')
    plt.title('boxplot')
    plt.xlabel('stress score')
    plt.grid(True)
    plt.savefig(box_file)
    plt.close()

    #Stats regarding stress score
    data = df['Stress Score'].describe()
    pd.DataFrame([data]).to_csv(summary_file, mode='w', index=False)

if __name__ == '__main__': 
    filepath = 'summary.csv'
    rawdata(filepath)



