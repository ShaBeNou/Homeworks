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
    plt.figure()
    # Create a new dataframe object where rows are trials where data['reward']==1 and columns are the corresponding 'green' values
    reward_df = pd.DataFrame(
        (data.loc[data['reward'] == 1])['green'].to_list(),
        index=(data.loc[data['reward'] == 1])['trial_number']
    )
    # Plot the heatmap with xticks relative to nose poke
    heatmap(reward_df, xticklabels=False, cbar_kws={'label': 'Photometry signal'})
    plt.xticks(ticks=[0,1000,2000,3000,4000,5000,6000], labels=['-1','0','1','2','3','4','5'])

    # Add titles
    plt.suptitle("Activity of dopaminergic axons relative to nose poke - Reward trials")  # Center title relative to the entire figure
    plt.xlabel("Time relative to nose poke (sec)")
    plt.ylabel("Trial Number")
    plt.tight_layout()
    plt.show()

    # Create a heatmap for omission trials
    plt.figure()
    omission_df = pd.DataFrame(
        (data.loc[data['reward'] == 0])['green'].to_list(),
        index=(data.loc[data['reward'] == 0])['trial_number']
    )
    heatmap(omission_df, xticklabels=False, cbar_kws={'label': 'Photometry signal'})
    plt.xticks(ticks=[0, 1000, 2000, 3000, 4000, 5000, 6000], labels=['-1', '0', '1', '2', '3', '4', '5'])

    plt.suptitle(
        "Activity of dopaminergic axons relative to nose poke - Omission trials")  # Center title relative to the entire figure
    plt.xlabel("Time relative to nose poke (sec)")
    plt.ylabel("Trial Number")
    plt.tight_layout()
    plt.show()

    # Create new dataframe for average reward trials and plot it
    plt.figure()
    avg_rwd = reward_df.mean()
    plt.plot(avg_rwd, label='Reward Trials')

    # Create new dataframe for average omission trials and plot it on the same graph
    avg_oms = omission_df.mean()
    plt.plot(avg_oms, label='Omission Trials')

    # Titles, legend
    plt.xticks(ticks=[0, 1000, 2000, 3000, 4000, 5000, 6000], labels=['-1', '0', '1', '2', '3', '4', '5'])
    plt.title('Mean photometric signal for reward and omission trials')
    plt.xlabel("Time relative to nose poke (sec)")
    plt.ylabel("Mean photometric signal")
    plt.legend()
    plt.tight_layout()
    plt.show()

    #  - - -  Find average signal and make a bar plot - - -

    # first, create a dataframe with the mean signal per trial within 1 second of the reward
    rwd_bar = reward_df.iloc[:,1000:2000].mean(axis=1)
    oms_bar = omission_df.iloc[:,1000:2000].mean(axis=1)

    # perform a t-test between the two series
    t_stat, p_value = ttest_ind(rwd_bar, oms_bar)

    # create a bar plot
    plt.figure()
    plt.bar(['Reward', 'No reward'], [rwd_bar.mean(), oms_bar.mean()], color=['#1f77b4', '#ff7f0e'])

    # add labels and title
    plt.ylabel('Mean photometric signal')
    plt.xlabel(f'P-value: {p_value:.3e}')
    plt.title(f'Mean photometric signal within 1 second of nose poke')
    plt.tight_layout()
    plt.show()

def Q_2(data):

    # Create arrays for the responses of reward trials - one for low probability and one for high probability
    rwd_low = data['green'].loc[(data['reward'] == 1) & (data['reward_prob'] == 20)].to_numpy().mean(axis=0)
    rwd_high = data['green'].loc[(data['reward'] == 1) & (data['reward_prob'] == 80)].to_numpy().mean(axis=0)

    # plot the results
    plt.figure()
    plt.plot(rwd_low, label='Low probability')
    plt.plot(rwd_high, label='High probability')
    plt.xticks(ticks=[0, 1000, 2000, 3000, 4000, 5000, 6000], labels=['-1', '0', '1', '2', '3', '4', '5'])
    plt.title('Mean photometric signal for reward trials with high and low probabilities')
    plt.xlabel("Time relative to nose poke (sec)")
    plt.ylabel("Mean photometric signal")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Do the same for omission trials
    oms_low = data['green'].loc[(data['reward'] == 0) & (data['reward_prob'] == 20)].to_numpy().mean(axis=0)
    oms_high = data['green'].loc[(data['reward'] == 0) & (data['reward_prob'] == 80)].to_numpy().mean(axis=0)
    plt.figure()
    plt.plot(oms_low, label='Low probability')
    plt.plot(oms_high, label='High probability')
    plt.xticks(ticks=[0, 1000, 2000, 3000, 4000, 5000, 6000], labels=['-1', '0', '1', '2', '3', '4', '5'])
    plt.title('Mean photometric signal for omission trials with high and low probabilities')
    plt.xlabel("Time relative to nose poke (sec)")
    plt.ylabel("Mean photometric signal")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # in order to show these reactions are experience based, I will show the dopaminergic axons response
    # for reward and omission trials for the first and last 15 trials
    rwd_first_15 = data['green'].loc[(data['reward'] == 1) &
                                                (data['reward_prob'] == 20)].to_numpy()[:15].mean(axis=0)
    rwd_last_15 = data['green'].loc[(data['reward'] == 1) &
                                    (data['reward_prob'] == 20)].to_numpy()[-15:].mean(axis=0)
    plt.figure()
    plt.plot(rwd_first_15, label='first 15 trials')
    plt.plot(rwd_last_15, label='last 15 trials')
    plt.xticks(ticks=[0, 1000, 2000, 3000, 4000, 5000, 6000], labels=['-1', '0', '1', '2', '3', '4', '5'])
    plt.title('Mean photometric signal for the first and last reward trials \n20% chance of reward')
    plt.xlabel("Time relative to nose poke (sec)")
    plt.ylabel("Mean photometric signal")
    plt.legend()
    plt.tight_layout()
    plt.show()

    oms_first_15 = data['green'].loc[(data['reward'] == 0) &
                                     (data['reward_prob'] == 20)].to_numpy()[:15].mean(axis=0)
    oms_last_15 =  data['green'].loc[(data['reward'] == 0) &
                                                (data['reward_prob'] == 20)].to_numpy()[-15:].mean(axis=0)
    plt.figure()
    plt.plot(oms_first_15, label='first 15 trials')
    plt.plot(oms_last_15, label='last 15 trials')
    plt.xticks(ticks=[0, 1000, 2000, 3000, 4000, 5000, 6000], labels=['-1', '0', '1', '2', '3', '4', '5'])
    plt.title('Mean photometric signal for the first and last omission trials \n20% chance of reward')
    plt.xlabel("Time relative to nose poke (sec)")
    plt.ylabel("Mean photometric signal")
    plt.legend()
    plt.tight_layout()
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




