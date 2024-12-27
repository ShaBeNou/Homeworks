# Imports
import sys
import os
import numpy as np
import pandas as pd

if hasattr(sys.modules['__main__'], '__file__'): # This 'if/else' statement allows the file to be run as a script
                                                 # without manually editing the pkl file directory as long as both
                                                 # the .pkl and .py file are in the same directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'Python_data.pkl')
    data = pd.read_pickle(file_path)
else:
    data = pd.read_pickle('C:/Users/97252/PycharmProjects/Homeworks/6184 Exercise/Part A/Python_data.pkl')
