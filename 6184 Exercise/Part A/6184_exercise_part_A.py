# Imports
import sys
import os
import numpy as np
import pandas as pd


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

if __name__ == '__main__':
    try:
        data = load_data(hardcoded_path='/6184 Exercise/Python_data.pkl')
    except FileNotFoundError as e:
        print(e)

else:
    raise Exception('This code is not intended to be run as a module, '
                    'please run as a standalone script or within an IDE')




