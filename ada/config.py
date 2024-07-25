#!/usr/bin/env python3

import sys 
import os
import string
import re
import numpy as np
import torch
import pandas as pd
import warnings
from copy import copy
from argparse import ArgumentParser
from datetime import datetime, timedelta

# Import necessary modules
from ada.environments import env_utils
from ada.agents.rl_algos import rl_utils
from ada.agents.mip_algos import mip_utils
from ada.agents.heuristic_algos import heuristic_utils
from ada.Loggermixin import *

# Initialize a logger using Loggermixin
# use self.logger from Loggermixin within classes
# use h_logger outside classes
h_logger=Loggermixin.get_default_logger()

# Define configuration dictionary with default values
config = {
    'AGENT_CLASS': 'RL',          # Type of agent
    'ENVIRONMENT': 'TARTAN',      # Environment name
    'N_PRODUCTS': 6,              # Number of products
    'START_TIME': '2018-01-01',   # Start time of the simulation
    'END_TIME': '2018-12-31',     # End time of the simulation
    'REWARD_FUNCTION': 'OTD1'     # Reward function to be used
    # Additional configuration parameters can be added here
}

# Function to parse command line arguments
def parse_cl_args(argv):
    config_args = ['default', '<path>']  # List of acceptable arguments for config
    config_help = 'Define the configuration file. Acceptable arguments are:'
    
    # Create an argument parser object with a description
    parser = ArgumentParser(description="Import configuration file or define" + 
        " the relevant parameters.")
    
    # Add a command line argument for the configuration file path
    parser.add_argument('--config', metavar='CONFIGPATH', type=str, default='default',
        help='Define the configuration file. Acceptable arguments are: ' + 
        ', '.join(config_args))

    # Parse and return the command line arguments
    return parser.parse_args()

def parse_config_file(config_data):
    # Function to ensure config files contain all relevant information
    # Raise warning if the following configuration requirements are not met
    # then populate with defaults.
    
    # Capitalize all configuration values
    config_data = capitalize_config_values(config_data)
    
    # Check if 'AGENT_CLASS' is defined in the configuration
    if 'AGENT_CLASS' not in config_data.keys():
        # Warn and set default value if not defined
        warn_string = '\nNo agent_class defined in configuration.' + \
            ' Loading A2C by default.'
        warnings.warn(warn_string)
        config_data['AGENT_CLASS'] = 'RL'
    
    # Check if 'ENVIRONMENT' is defined in the configuration
    if 'ENVIRONMENT' not in config_data.keys():
        # Warn and set default value if not defined
        warn_string = '\nNo environment defined in configuration.' + \
            ' Loading TARTAN by default.'
        warnings.warn(warn_string)
        config_data['ENVIRONMENT'] = 'TARTAN'
    
    # Check remaining configuration settings using environment utilities
    config_data = env_utils.check_env_settings(config_data)
    
    # Further check settings based on the type of agent class
    if config_data['AGENT_CLASS'] == 'RL':
        config_data = rl_utils.check_settings(config_data)
    elif config_data['AGENT_CLASS'] == 'MIP':
        config_data = mip_utils.check_settings(config_data)
    elif config_data['AGENT_CLASS'] == 'HEURISTIC':
        config_data = heuristic_utils.check_settings(config_data)
    else:
        # Raise an error if the agent class is not recognized
        raise ValueError('AGENT_CLASS {} not recognized.'.format(
            config_data['AGENT_CLASS']))

    # Return the validated and possibly modified configuration data
    return config_data

def save_config_file(config_data):
    # Function to save configuration data to a file

    # Get the file path from the configuration data
    filepath = config_data['DATA_PATH']  # Data Path set by agent settings
    
    # Open the file in write mode
    file = open(os.path.join(filepath, 'config_settings.txt'), 'w')
    
    # Write the header for the configuration settings
    file.writelines('parameter,value')
    
    # Define the primary settings to be saved first
    primary_settings = ['AGENT_CLASS', 'MIP_ALGO', 'RL_ALGO', 'ENVIRONMENT',
                        'SYS_START_TIME', 'DATA_PATH']

    # Save primary configuration settings
    [file.writelines("\n{},{}".format(key, config_data[key])) 
        for key in primary_settings if key in config_data.keys()]
    
    # Save remaining configuration settings
    [file.writelines("\n{},{}".format(key, config_data[key])) 
        for key in config_data.keys() if key not in primary_settings]
    
    # Close the file after writing all settings
    file.close()
    
    # Log primary configuration settings for debugging
    [h_logger.debug("{}: {}".format(key, config_data[key]))
        for key in primary_settings if key in config_data.keys()]
    
    # Log remaining configuration settings for debugging
    [h_logger.debug("{}: {}".format(key, config_data[key]))
        for key in config_data.keys() if key not in primary_settings]

def capitalize_config_values(config_data):
    # Ensure the input is in dictionary format
    assert type(config_data) == dict, "Configuration data not in dictionary format."
    
    # Iterate over each key in the configuration data
    for key in config_data.keys():
        # Skip keys that contain 'PATH' as they may be case sensitive
        if 'PATH' in key:
            continue
        
        # If the value is a string, convert it to uppercase
        if type(config_data[key]) == str:
            config_data[key] = config_data[key].upper()
    
    # Return the modified configuration data
    return config_data

def load_config_file(filepath):
    # Function to load configuration data from a file

    config_data = None  # Initialize config_data to None
    supported_extensions = ['.csv', '.txt', '.xls', '.xlsx']  # Supported file extensions

    # Check config file extension
    user_input = filepath  # Get the file path from the user input
    filename, file_ext = os.path.splitext(user_input)  # Split the file name and extension
    h_logger.debug(user_input)  # Log the user input for debugging
    count = 5  # Initialize a counter for retry attempts

    # Loop until config_data is successfully loaded or user quits
    while config_data is None:
        try:
            # Load config file based on its extension
            if file_ext == '.csv' or file_ext == '.txt':
                # Load CSV or TXT file into a DataFrame
                data = pd.read_csv(user_input, header=0)
                data = data.set_index(data[data.columns[0]])
                config_data = data.xs(data.columns[1], axis=1, drop_level=True).to_dict()
            elif file_ext == '.xlsx' or file_ext == '.xls':
                # Load Excel file into a DataFrame
                data = pd.read_excel(user_input, header=0)
                data = data.set_index(data[data.columns[0]])
                config_data = data.xs(data.columns[1], axis=1, drop_level=True).to_dict()
        except FileNotFoundError:
            # Handle file not found error
            h_logger.debug("No valid configuration file found.")
            user_input = input("Enter path to file, or 1 to load default." +
                               " 2 to quit. Be sure to escape any backslashes " +
                               "on Windows.\n>>>> ")
            filename, file_ext = os.path.splitext(user_input)
            if user_input == str(1):
                # Load default configuration if user inputs '1'
                config_data = config
                break
            if user_input == str(2):
                # Exit if user inputs '2'
                sys.exit('No valid file entered. Exiting.')

        # Check if the file extension is supported
        if file_ext not in supported_extensions:
            h_logger.debug("Invalid file format from {}. Valid extensions are:".format(user_input))
            [h_logger.debug(ext) for ext in supported_extensions]
            count += 1  # Increment the counter
            if count >= 5:
                # Exit if the maximum number of attempts is reached
                sys.exit('No valid file entered. Exiting.')

    # Return the configuration data as a dictionary
    if type(config_data) == pd.core.frame.DataFrame:
        return config_data.to_dict()
    elif type(config_data) == dict:
        return config_data
    
# Useful for deleting unnecessary results folders during testing
def get_results_path(path, target):
    # Initialize path_name to None
    path_name = None
    
    # Loop until the target directory name is found
    while path_name != target:
        # Split the path into the directory and the last component
        path_split = os.path.split(path)
        path_name = path_split[1]  # Get the last component of the path
        path = path_split[0]  # Update the path to the directory part

    # Return the full path to the target directory
    return os.path.join(path, path_name)

def set_up_sim(args, default_path=None, config_dict=None):
    # Use explicitly supplied configuration if provided
    if config_dict is not None:
        if type(config_dict) == dict:
            config_data = config_dict
        else:
            # Raise an error if the provided config_dict is not a dictionary
            raise ValueError("config_dict not dict type. {} passed.".format(type(config_dict)))
    # Process the args, if supplied
    elif args is not None and args.config is not None:
        if args.config.lower() == 'default':
            if default_path is not None:
                # Load configuration from the default path
                config_data = load_config_file(default_path)
            else:
                # Raise an error if default path is not supplied
                raise ValueError("--config=default but no default path supplied")
        else:
            # Load configuration from the specified path
            config_data = load_config_file(args.config)
    # Otherwise, fall back to the default path
    elif default_path is not None:
        # Load configuration from the default path
        config_data = load_config_file(default_path)
    else:
        # Raise an error if no configuration is supplied
        raise ValueError("No config supplied")

    # Parse and validate the configuration data
    config_data = parse_config_file(config_data)
    
    # Set random seeds for reproducibility
    np.random.seed(config_data['RANDOM_SEED'])
    torch.manual_seed(config_data['RANDOM_SEED'])
    try:
        if config_data['DEVICE'] == 'GPU':
            # Set seed for CUDA if using GPU
            torch.cuda.manual_seed_all(config_data['RANDOM_SEED'])
    except KeyError:
        # Default to CPU if DEVICE is not specified
        config_data['DEVICE'] = 'CPU'
    
    # Initialize environment based on the configuration
    if config_data['ENVIRONMENT'] == 'TARTAN':
        from ada.environments.tartan import productionFacility
        env = productionFacility(config_data)
    # GOPHER environment not in repository
    # elif config_data['ENVIRONMENT'] == 'GOPHER':
       # from ada.environments.gopher import productionFacility
        # env = productionFacility(config_data)
    else:
        # Raise an error if the environment name is not recognized
        raise ValueError('Environment name {} not recognized.'.format(
            config_data['ENVIRONMENT']))

    # Initialize agent based on the configuration
    if config_data['AGENT_CLASS'] == 'RL':
        from ada.agents.rl_agent import create_agent
        agent = create_agent(env)
    elif config_data['AGENT_CLASS'] == 'MIP':
        from ada.agents.opt_agent import create_agent
        agent = create_agent(env)
    elif config_data['AGENT_CLASS'] == 'HEURISTIC':
        from ada.agents.heuristic_agent import create_agent
        agent = create_agent(env)
    else:
        # Raise an error if the agent class is not recognized
        raise ValueError('AGENT_CLASS {} not recognized.'.format(
            config_data['AGENT_CLASS']))

    # Save configuration data for reference
    save_config_file(config_data)

    # Return the initialized agent
    return agent

# TODO: This has grown to be an ad hoc mess and ought to be moved
# to ppm.py where these helper functions and methods can be called
# when ORDER_BOOK_PATH, PRODUCT_DATA_PATH, or other values are populated.
def set_up_production_environment(args, default=None):
    # Check if the configuration is set to 'default' and a default path is provided
    if args.config.lower() == 'default' and default is not None:
        # Load configuration from the default path
        config_data = load_config_file(default)
    elif args.config.lower() != 'default':
        # Load configuration from the specified path
        config_data = load_config_file(args.config)
    else:
        # Raise an error if no configuration file is provided
        raise ValueError('No configuration file provided.')

    # Parse and validate the configuration data
    config_data = parse_config_file(config_data)
    
    # Set the start and end times for the production environment
    today = datetime.now()
    config_data['START_TIME'] = str(today.date())
    config_data['END_TIME'] = str(today.date() + 
        timedelta(days=config_data['LOOKAHEAD_PLANNING_HORIZON']))
    
    # Set random seeds for reproducibility
    np.random.seed(config_data['RANDOM_SEED'])
    torch.manual_seed(config_data['RANDOM_SEED'])
    if config_data['DEVICE'] == 'GPU':
        # Set seed for CUDA if using GPU
        torch.cuda.manual_seed_all(config_data['RANDOM_SEED'])
    
    # Initialize the environment based on the configuration
    if config_data['ENVIRONMENT'] == 'TARTAN':
        from ada.environments.tartan import productionFacility
        env = productionFacility(config_data)
    # GOPHER environment not in repository
    # elif config_data['ENVIRONMENT'] == 'GOPHER':
       # from ada.environments.gopher import productionFacility
        # env = productionFacility(config_data)
    else:
        # Raise an error if the environment name is not recognized
        raise ValueError('Environment name {} not recognized.'.format(
            config_data['ENVIRONMENT']))

    # Load and process the current state data
    inventory, order_book, forecast = \
        env_utils.load_current_state_data(config_data)
    env.order_book = env_utils.process_order_data(order_book, env)
    inventory = env_utils.process_inventory_data(inventory, env)
    env.inventory = copy(inventory.flatten().astype(float))

    # Forecast requires separate preprocessing
    env.monthly_forecast = env_utils.process_forecast_data(forecast, env)

    # Load the current schedule
    env.schedule = env_utils.load_current_schedule(env)
    
    # Initialize the agent based on the configuration
    if config_data['AGENT_CLASS'] == 'RL':
        from ada.agents.rl_agent import create_agent
        agent = create_agent(env)
    elif config_data['AGENT_CLASS'] == 'MIP':
        from ada.agents.opt_agent import create_agent
        agent = create_agent(env)
    elif config_data['AGENT_CLASS'] == 'HEURISTIC':
        from ada.agents.heuristic_agent import create_agent
        agent = create_agent(env)
    else:
        # Raise an error if the agent class is not recognized
        raise ValueError('AGENT_CLASS {} not recognized.'.format(
            config_data['AGENT_CLASS']))

    # Save configuration data for reference (commented out)
    # save_config_file(config_data)

    # Return the initialized agent
    return agent