import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau, shapiro
import statsmodels.api as sm
import os

def total_blinks_analysis(filepath): 
    df = pd.read_csv(filepath)

    blinks = df['total_blinks']
    score = df['Stress Score']

    home = os.path.expanduser('~')
    plot_folder = os.path.join(home, 'Downloads', 'blink_project', 'Plots_total_blinks')
    os.makedirs(plot_folder, exist_ok=True)
    plot_file = os.path.join(plot_folder, 'scatterplot_robreg.png')
    summary_folder = os.path.join(home, 'Downloads', 'blink_project', 'summary')
    summary_file = os.path.join(summary_folder, 'total_blinks_summary.csv')

    #Shapiro Wilk's test 
    stat, p_val_shapiro = stats.shapiro(blinks)
    normal = 'Normally Distributed' if p_val_shapiro > 0.05 else 'not normally distributed'

    #Pearson's correlation coefficient test
    r_val, p_val_pear = stats.pearsonr(blinks, score)
    sig_pear = 'Significant' if p_val_pear < 0.05 else 'Not Significant'

    #Spearman's rank correlation coefficient test
    rho, p_val_spear = stats.spearmanr(blinks, score)
    sig_spear = 'Significant' if p_val_spear < 0.05 else 'Not Significant'

    #Kendall Tau's test
    tau, p_val_ken = stats.kendalltau(blinks, score)
    sig_ken = 'Significant' if p_val_ken < 0.05 else 'Not Significant'

    #scatter plot with a robust linear regression line 
    x = blinks 
    y = score 
    x_with_constant = sm.add_constant(x)

    regression_line_model = sm.RLM(y, x_with_constant) #this sets up a formula 
    regression_line_cal = regression_line_model.fit() #this actually calculates the best-fit slope and intercept for the model

    plt.figure()
    plt.scatter(x, y, color='blue')
    plt.plot(x, regression_line_cal.predict(x_with_constant), color='red')
    plt.title('scat plot with rob reg')
    plt.xlabel('total blinks')
    plt.ylabel('stress score')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_file)
    plt.close()

    result = {
        'Shapiro-Wilk p-value (Total Blinks)': round(p_val_shapiro, 4),
        'Normality of Blink Distribution': normal,

        'Pearson r': round(r_val, 3),
        'Pearson p-value': round(p_val_pear, 4),
        'Pearson Significance': sig_pear,

        'Spearman rho': round(rho, 3),
        'Spearman p-value': round(p_val_spear, 4),
        'Spearman Significance': sig_spear,

        'Kendall tau': round(tau, 3),
        'Kendall p-value': round(p_val_ken, 4),
        'Kendall Significance': sig_ken,

        'Regression Summary': regression_line_cal.summary().as_text()
    }

    pd.DataFrame([result]).to_csv(summary_file, mode='w', index=False)
    print(f'Successfully analyzed')

if __name__ == '__main__': 
    filepath = 'summary.csv'
    total_blinks_analysis(filepath)
