# Imports
import sys
import os
import numpy as np
import pandas as pd

# A lot of code just to make sure the file can be run as a standalone script
if hasattr(sys.modules['__main__'], '__file__'): # This 'if/else' statement allows the file to be run as a script
                                                 # without manually editing the pkl file directory as long as both
                                                 # the .pkl and .py file are in the same directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'Python_data.pkl')
    if os.path.exists(file_path):
        data = pd.read_pickle(file_path)
        print('data loaded')
    else:
        raise Exception('If running this code as a standalone script data file should be in the same directory as this script.')
else:                                            # if the code is being run from within an IDE console, uses hardcoded
                                                 # set directory
    file_path = 'C:/Users/97252/PycharmProjects/Homeworks/6184 Exercise/Part A/Python_data.pkl'
    if os.path.exists(file_path):
        data = pd.read_pickle(file_path)
    else:
        raise Exception('File not found, please manually define the data file directory')

