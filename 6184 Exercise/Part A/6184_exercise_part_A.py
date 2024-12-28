# Imports
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    reward_df = data.loc[data['reward'] == 1].drop(columns=['reward', 'side', 'reward_prob'])
    reward_heatmap = pd.DataFrame(reward_df['green'].to_list(), index=reward_df['trial_number'])
    heatmap(reward_heatmap, xticklabels=1000) #TODO: check if mouse poked at time 0 or 1, add titles
    plt.show()

if __name__ == '__main__':
    try:
        data = load_data(hardcoded_path='C:/Users/97252/PycharmProjects/Homeworks/6184 Exercise/Part A/Python_data.pkl')
    except FileNotFoundError as e:
        print(e)
    Q_1(data)
else:
    raise Exception('This code is not intended to be run as a module, '
                    'please run as a standalone script or within an IDE')




