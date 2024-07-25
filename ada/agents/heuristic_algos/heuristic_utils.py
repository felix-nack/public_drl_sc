# heuristic_utils: Contains various helper functions to extract data from 
# heuristic functions and set default configurations.
# Author: Christian Hubbs
# Contact: christiandhubbs@gmail.com
# Date: 20.02.2019

import os 
from datetime import datetime

def check_settings(settings=None):
    '''
    Input
    settings: dict of values required to parameterize the mip. 
        Missing values or None is permissible as these will
        be populated with defaults.

    Output
    settings: dict of values required to completely specify the mip.
    '''
    # Default settings for the heuristic algorithm
    defaults = {
        'HEURISTIC_ALGO': 'RANDOM'
    }
    
    # If no settings are provided, use the default settings
    if settings is None:
        settings = defaults
    else:
        # Populate missing settings with default values
        for key in defaults.keys():
            if key not in settings.keys():
                settings[key] = defaults[key]
            elif defaults[key] is not None:
                # Ensure the type of the setting matches the default type
                settings[key] = type(defaults[key])(settings[key])
                
    # If DATA_PATH is not specified, create a default path
    if 'DATA_PATH' not in settings.keys():
        default_data_path = os.getcwd() + "/RESULTS/" + settings['HEURISTIC_ALGO'].upper() + '/'\
            + datetime.now().strftime("%Y_%m_%d_%H_%M")
        settings['DATA_PATH'] = default_data_path

    # Create the directory if it does not exist
    os.makedirs(settings['DATA_PATH'], exist_ok=True)

    return settings
