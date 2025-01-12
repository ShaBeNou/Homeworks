# Imports
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter1d

def load_data(hardcoded_path=None):
    """
    A function to load the .pkl data file.
    if run as a standalone script assumes the .pkl file is in the same directory as this code!

    hardcoded_path: (str) path to hardcoded data file, in case the code is run through IDE.
    """

    # Check if run through IDE:
    if hasattr(sys.modules['__main__'], '__file__'):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'data/Python_C4886.pkl')
        if os.path.exists(file_path):
            data = pd.read_pickle(file_path)
            print('data loaded')
        else:
            raise FileNotFoundError(
                'When running this code as a standalone script data file must be named "Python_data.pkl" and be in '
                'the same directory as this script or provided in hardcoded_path!')
    else:
        file_path = hardcoded_path
        if os.path.exists(file_path):
            data = pd.read_pickle(file_path)
        else:
            raise FileNotFoundError(
                'File not found. Please manually define a valid path to the .pkl file in `hardcoded_path`.')

    return data

def P2Q_1(data):
    # calculate whether up has higher probability for reward over right
    dominantOpt = len(data[data['U_prob']>data['R_prob']])

    # calculate the number of times the monkey picked the better probability option
    betOpt = len(data[
                     ((data['U_prob'] > data['R_prob']) & (data['choice'] == 0)) |
                     ((data['U_prob'] < data['R_prob']) & (data['choice'] == 1))
    ])
    print(f'The number of trials where the up choice was better than right is {dominantOpt} trials,'
          f' which are {np.round(dominantOpt*100/len(data), 2)} percent of all trials.')
    print(f'The number of trials where the monkey picked the better probability choice is {betOpt} trials,'
          f' which are {np.round(betOpt*100/len(data), 2)} percent of all trials.')

def P2Q_2(data):
    # Create a new dataframe representing neural activity for trials where the monkey picked the "up" choice
    up_raster = pd.DataFrame(
        (data.loc[data['choice'] == 0])['spikes'].to_list()
    )

    # Plot this data as a heatmap
    plt.figure()
    sns.heatmap(up_raster, xticklabels=100, cbar=False, cmap=['white', 'black'])
    plt.suptitle(
        "Spikes per trials of substantia nigra pars reticulata neurons - Up selected trials")
    plt.xlabel("Time (msec)")
    plt.ylabel("Trial")
    plt.tight_layout()
    plt.show()

    # Exactly same analysis, but for trials where the monkey picked the "right" choice
    right_raster = pd.DataFrame(
        (data.loc[data['choice'] == 1])['spikes'].to_list()
    )
    plt.figure()
    sns.heatmap(right_raster, xticklabels=100, cbar=False, cmap=['white', 'black'])
    plt.suptitle(
        "Spikes per trials of substantia nigra pars reticulata neurons - Right selected trials")
    plt.xlabel("Time (msec)")
    plt.ylabel("Trial")
    plt.tight_layout()
    plt.show()

def P2Q_3(data, right=25, up=75, fig_size=(15,15), h_space=0.75):
    """
    A function that creates a figure with PSTHs subplots for each condition, and additional PSTH with smoothing.
    Unlike the other function, this function gets optional arguments.
    These optional arguments allow choosing a specific PSTH to be plotted with smoothing, and parameters to adjust
    subplots size and vertical spacing.
    Default values were picked based on what works best on my personal machine.
    """
    # Create a list of tuples for all the conditions
    conditions = [(r_prob, u_prob)for r_prob in np.unique(data['R_prob']) for u_prob in np.unique(data['U_prob'])
        if r_prob != u_prob]

    # Define number of rows and columns for the subplots
    n_conditions = len(conditions)
    n_rows = int(np.ceil(np.sqrt(n_conditions)))
    n_cols = int(np.ceil(n_conditions / n_rows))

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=n_rows,ncols=n_cols,figsize=fig_size,sharey=True)
    axes = axes.flatten()

    # Iterate over the conditions and create subplots
    for idx, (r_prob, u_prob) in enumerate(conditions):
        # Filter data for the current condition
        psth = pd.DataFrame(
            (data.loc[(data['R_prob'] == r_prob) & (data['U_prob'] == u_prob)])['spikes'].to_list()
        ).mean(axis=0) * 1000

        # Plot the PSTH in the fitting subplot
        ax = axes[idx]
        ax.plot(psth)
        ax.set_title(f'Right probability ={r_prob}, Up probability={u_prob}')
        ax.set_xlabel("Time (msec)")
        if idx % n_cols == 0:  # Only label the y-axis for the first column
            ax.set_ylabel("Firing Rate (spikes/sec)")
        ax.tick_params(axis='both', which='major')

    # Show the plots
    plt.tight_layout()
    plt.subplots_adjust(hspace=h_space)  # Increase this value for more spacing
    plt.show()

    # Plot one of the PSTHs again, this time with smoothing
    plt.figure()
    new_psth = pd.DataFrame(
                    (data.loc[(data['R_prob'] == right) & (data['U_prob'] == up)])['spikes'].to_list()
                ).mean(axis=0)*1000
    smoothed_psth = uniform_filter1d(new_psth, 100)
    plt.plot(new_psth, label='unsmoothed')
    plt.plot(smoothed_psth, label='smoothed', lw=3.0, color='r')
    plt.legend()
    plt.title(f'Average firing rate - Right probability={right}, Up probability={up}')
    plt.xlabel("Time (msec)")
    plt.ylabel("Firing Rate (spikes/sec)")
    plt.tight_layout()
    plt.show()

def P2Q_4(data, colormap="tab20"):

    # Same list of tuples as in P2Q_3
    conditions = [(r_prob, u_prob) for r_prob in np.unique(data['R_prob']) for u_prob in np.unique(data['U_prob'])
                  if r_prob != u_prob]

    plt.figure()
    plt.tight_layout()

    # Use a colormap for different colors
    cmap = plt.get_cmap(colormap)  # A set of distinct colors

    for idx, (r_prob, u_prob) in enumerate(conditions):
        # Filter data for the current condition
        psth = pd.DataFrame(
            (data.loc[(data['R_prob'] == r_prob) & (data['U_prob'] == u_prob)])['spikes'].to_list()
        ).mean(axis=0) * 1000
        smoothed_psth = uniform_filter1d(psth, 100)
        plt.plot(smoothed_psth, label=f'R={r_prob}, U={u_prob}', color=cmap(idx % 20))
        plt.legend()
        plt.title(f'Average smoothed firing rates per condition')
        plt.xlabel("Time (msec)")
        plt.ylabel("Firing Rate (spikes/sec)")
        plt.show()

    # Create two PSTHs - one for all conditions where R_prob>U_prob and vice versa
    r_psth = pd.DataFrame(
            (data.loc[data['R_prob']>data['U_prob']])['spikes'].to_list()
        ).mean(axis=0) * 1000
    smoothed_r_psth = uniform_filter1d(r_psth, 100)

    u_psth = pd.DataFrame(
            (data.loc[data['U_prob']>data['R_prob']])['spikes'].to_list()
        ).mean(axis=0) * 1000
    smoothed_u_psth = uniform_filter1d(u_psth, 100)

    # plot the smoothed PSTHs
    plt.figure()
    plt.plot(smoothed_r_psth, label='R>U')
    plt.plot(smoothed_u_psth, label='U>R')
    plt.legend()
    plt.title(f'Average smoothed firing rates per condition')
    plt.xlabel("Time (msec)")
    plt.ylabel("Firing Rate (spikes/sec)")
    plt.show()


if __name__ == '__main__':
    try:
        data = load_data(hardcoded_path='C:/Users/97252/PycharmProjects/Homeworks/6184 Exercise/Part B/data/Python_C4886.pkl')
    except FileNotFoundError as e:
        print(e)
    P2Q_1(data)
    P2Q_2(data)
    P2Q_3(data)
    P2Q_4(data)
else:
    pass