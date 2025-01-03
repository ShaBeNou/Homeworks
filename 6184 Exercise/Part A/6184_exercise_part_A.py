# Imports
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from seaborn import heatmap


def load_data(hardcoded_path=None):
    """
    A function to load the .pkl data file.
    if run as a standalone script assumes the .pkl file is in the same directory as this code!

    hardcoded_path: (str) path to hardcoded data file, in case the code is run through IDE.
    """

    # Check if run through IDE:
    if hasattr(sys.modules['__main__'], '__file__'):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'Python_data.pkl')
        if os.path.exists(file_path):
            data = pd.read_pickle(file_path)
            print('data loaded')
        else:
            raise FileNotFoundError(
                'When running this code as a standalone script data file must be named "Python_data.pkl" and be in '
                'the same directory as this script!')
    else:
        file_path = hardcoded_path
        if os.path.exists(file_path):
            data = pd.read_pickle(file_path)
        else:
            raise FileNotFoundError(
                'File not found. Please manually define a valid path to the .pkl file in `hardcoded_path`.')

    return data

def Q_1(data):

    # Create a heatmap for reward trials
    plt.figure() # new figure
    # Create a new dataframe object where rows are trials where data['reward']==1 and columns are the corresponding 'green' values
    reward_df = pd.DataFrame((data.loc[data['reward'] == 1])['green'].to_list(), index=(data.loc[data['reward'] == 1])['trial_number'])
    heatmap(reward_df, xticklabels=1000) #TODO: check if mouse poked at time 0 or 1, add titles #plot the data
    plt.show()

    # Create a heatmap for omission trials
    plt.figure()
    omission_df = pd.DataFrame((data.loc[data['reward'] == 0])['green'].to_list(), index=(data.loc[data['reward'] == 0])['trial_number'])
    heatmap(omission_df, xticklabels=1000)  # plot the heatmap from the new dataframe
    plt.show()

    # Create new dataframe for average reward trials and plot it
    plt.figure()
    avg_rwd = reward_df.mean()
    plt.plot(avg_rwd)

    # Create new dataframe for average omission trials and plot it
    avg_oms = omission_df.mean()
    plt.plot(avg_oms)
    plt.show()

    #  - - -  Find average signal and make a bar plot - - -

    # first, create a dataframe with the mean signal per trial within 1 second of the reward
    rwd_bar = reward_df.iloc[:,1000:2000].mean(axis=1)
    oms_bar = omission_df.iloc[:,1000:2000].mean(axis=1)

    # perform a t-test between the two series
    t_stat, p_value = ttest_ind(rwd_bar, oms_bar)

    # create a bar plot
    plt.figure()
    plt.bar(['Reward', 'No reward'], [rwd_bar.mean(), oms_bar.mean()])

    # add the p-value
    pval_text = f"P-value: {p_value:.3e}"
    plt.text(0.5, max(rwd_bar.mean(), oms_bar.mean()) + 0.1, pval_text, ha='center', fontsize=10)

    # add labels and title
    plt.ylabel('Average Value')
    plt.title('Comparison of Series Averages with P-value')
    plt.show()

def Q_2(data):

    plt.figure()
    rwd_low = data['green'].loc[(data['reward']==1) & (data['reward_prob']==20)].to_numpy().mean(axis=0)
    rwd_high = data['green'].loc[(data['reward'] == 1) & (data['reward_prob'] == 80)].to_numpy().mean(axis=0)
    plt.plot(rwd_low)
    plt.plot(rwd_high)
    plt.show()

    plt.figure()
    oms_low = data['green'].loc[(data['reward'] == 0) & (data['reward_prob'] == 20)].to_numpy().mean(axis=0)
    oms_high = data['green'].loc[(data['reward'] == 0) & (data['reward_prob'] == 80)].to_numpy().mean(axis=0)
    plt.plot(oms_low)
    plt.plot(oms_high)
    plt.show()

if __name__ == '__main__':
    try:
        data = load_data(hardcoded_path='C:/Users/97252/PycharmProjects/Homeworks/6184 Exercise/Part A/Python_data.pkl')
    except FileNotFoundError as e:
        print(e)
    Q_1(data)
    Q_2(data)
else:
    raise Exception('This code is not intended to be run as a module, '
                    'please run as a standalone script or within an IDE')




