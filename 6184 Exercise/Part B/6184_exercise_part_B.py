# Imports
import sys
import os
import pandas as pd
import numpy as np

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
                'the same directory as this script!')
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

if __name__ == '__main__':
    try:
        data = load_data(hardcoded_path='C:/Users/97252/PycharmProjects/Homeworks/6184 Exercise/Part B/data/Python_C4886.pkl')
    except FileNotFoundError as e:
        print(e)
else:
    raise Exception('This code is not intended to be run as a module, '
                    'please run as a standalone script or within an IDE')