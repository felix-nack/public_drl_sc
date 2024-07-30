# utils.py
# Christian Hubbs
# christiandhubbs@gmail.com
# 13.03.2018

# This file contains numerous utilities for implementing the PPM simulation.

import numpy as np
import pandas as pd
import sys
import os
import matplotlib as mpl
if sys.platform == 'linux':
    if os.environ.get('DISPLAY', '') == '':
        mpl.use('Agg')
    else:
        mpl.use('TkAgg')
import matplotlib.pyplot as plt 
import string
from copy import copy
from datetime import datetime, date, timedelta
from dateutil import parser
import calendar
import time
import pickle
import warnings
from str2bool import str2bool
import yaml # Nachträglich eingretragen, um Fehler zu beheben
import openpyxl as pyxl
import re
from simpledbf import Dbf5
from .demand_models.generate_orders import check_demand_settings
pd.options.mode.chained_assignment = None

from ada.Loggermixin import *

# use self.logger from Loggermixin within classes
# use h_logger outside classes
h_logger = Loggermixin.get_default_logger()

# Check environment settings to ensure everything is enumerated and revert
# to defaults where values are missing.
def check_env_settings(settings=None, *args, **kwargs):
    '''
    Input
    settings: dict of values required to parameterize the simulation 
        environment. Missing values or None is permissible as these will
        be populated with defaults.

    Output
    settings: dict of values required to completely specify the simulation
        environment.
    '''
    # Define default settings for the environment
    defaults = {
         'SYS_START_TIME': datetime.strftime(
            datetime.fromtimestamp(time.time()), "%Y-%m-%d"),
         'RANDOM_SEED': int(time.time()),
         'N_PRODUCTS': 6,
         'TRANSITION_MATRIX_SETTING': 'RANDOM',
         'BASE_TIME_INTERVAL': 1,
         'BASE_TIME_UNIT': 'DAY',
         'FIXED_PLANNING_HORIZON': 7,
         'LOOKAHEAD_PLANNING_HORIZON': 14,
         'MAINTENANCE_MODEL': 'UNIFORM_STOP',
         'DEMAND_MODEL': 'SEASONAL_DEMAND',
         'PRODUCT_DATA_PATH': None, # Enter file path to import data if available
         'SHUTDOWN_PROB': 0.0,
         'START_TIME': '2018-01-01',
         'END_TIME': '2018-04-01',
         'WEEKEND_SHIPMENTS': True,
         'REWARD_FUNCTION': 'OTD1',
         'STATE_SETTING': 'INV_BALANCE_PRODUCTION',
         'ORDER_BOOK': None,
         'WORKING_CAPITAL_PERCENTAGE': 0.1 / 365,
         'LATE_PENALTY': 25,
         'TRAIN': True,
         'IMPOSE_MIN_CAMPAIGN': True
    }

    # If no settings are provided, use the default settings
    if settings is None:
        settings = defaults
    else:
        # Ensure all default keys are present in the provided settings
        for key in defaults.keys():
            if key not in settings.keys():
                settings[key] = defaults[key]
            elif key == 'START_TIME' or key == 'END_TIME':
                # Ensure standard formatting of YYYY-MM-DD
                settings[key] = str(parser.parse(settings[key]).date())
            elif defaults[key] is not None:
                if type(defaults[key]) == bool:
                    # Convert string to boolean if the default value is a boolean
                    settings[key] = str2bool(str(settings[key]))
                else:
                    # Convert the provided setting to the type of the default value
                    settings[key] = type(defaults[key])(settings[key])
            elif 'PATH' in key and settings[key] is not None:
                # Ensure paths are strings
                settings[key] = str(settings[key])

    # Validate environment-specific settings
    if settings['ENVIRONMENT'] == 'TARTAN':
        assert settings['BASE_TIME_UNIT'] == 'DAY', 'BASE_TIME_UNIT = {}. \
        Tartan model only supports Days as the base time interval'.format(
            settings['BASE_TIME_UNIT'])
    elif settings['ENVIRONMENT'] == 'GOPHER':
        assert settings['BASE_TIME_UNIT'] == 'HOUR', 'BASE_TIME_UNIT = {}. \
        Gopher model only supports hours as the base time interval'.format(
            settings['BASE_TIME_UNIT'])

    # Check and update demand settings
    settings = check_demand_settings(settings)

    return settings

def load_scenario_data(path, env=None):
    """
    Load scenario data from a specified file path. Supports .pkl, .xlsx, and .yaml file extensions.

    Args:
        path (str): The file path to load the scenario data from.
        env (optional): The environment object, required for loading .yaml files.

    Returns:
        tuple: Contains prod_data, trans_mat, zfin, zfin_to_gmid, zfin_data.
    """
    print(path)
    supported_extensions = ['.pkl', '.xlsx', '.yaml']
    
    # Check for both relative and absolute paths
    if os.path.exists(path):
        pass
    elif os.path.exists(os.path.join(os.getcwd(), path)):
        path = os.path.join(os.getcwd(), path)
    else:
        # Raise an error if the file is not found
        raise FileNotFoundError('File not found: {}'.format(path))
    
    filename, file_ext = os.path.splitext(path)
    
    if file_ext.lower() in supported_extensions:
        zfin_to_gmid = None
        zfin_data = None
        
        # Load data from Pickle files
        if file_ext.lower() == '.pkl':
            print(path)
            prod_data, trans_mat = pickle.load(open(path, 'rb'))
            zfin = np.arange(prod_data.shape[0])
            zfin_to_gmid = {i: i for i in zfin}
            zfin_data = {i for i in zfin}
        
        # Load data from Excel files
        elif file_ext.lower() == '.xlsx':
            prod_data, trans_mat, zfin, zfin_to_gmid, zfin_data = load_product_data_from_excel(path)
        
        # Load data from YAML files
        elif file_ext.lower() == '.yaml':
            prod_data, trans_mat = load_yaml_data(env)
            zfin = np.arange(prod_data.shape[0])
            zfin_to_gmid = {i: i for i in zfin}
            zfin_data = {i for i in zfin}
    else:
        # Raise an error if the file extension is not supported
        raise ValueError('File extension {} not currently supported'.format(file_ext))
    
    return prod_data, trans_mat, zfin, zfin_to_gmid, zfin_data

def aggregate_orders(env):
    """
    Aggregate orders from the environment's order book based on the current simulation time.

    Args:
        env: The environment object containing the order book and simulation time.

    Returns:
        tuple: Contains order_pred_qty and unique_order.
    """
    # TODO: There may be situations where low demand is encountered
    # and there are no orders. This will raise an error down the line
    
    # Filter orders that have been created but not yet shipped and have a valid document number
    pred_orders = env.order_book[np.where(
        (env.order_book[:, env.ob_indices['doc_create_time']] <= env.sim_time) & 
        (env.order_book[:, env.ob_indices['shipped']] == 0) &
        (env.order_book[:, env.ob_indices['doc_num']] > 0)
    )].astype(float)

    # Get unique orders and their quantities
    unique_order, unique_order_id = np.unique(pred_orders[:, env.ob_indices['gmid']], return_inverse=True)
    order_pred_qty = np.bincount(unique_order_id, pred_orders[:, env.ob_indices['order_qty']])

    return order_pred_qty, unique_order

def get_net_forecast(env):
    """
    Calculate the net forecast by aggregating orders by month and GMID (Global Material Identifier).

    Args:
        env: The environment object containing order book, simulation time, and forecast data.

    Returns:
        np.ndarray: A 2D array representing the net forecast for each month and product.
    """
    # Get the current month from the simulation time
    current_month = env.sim_day_to_date[env.sim_time][0]
    
    # Initialize a 12xN array to store the net forecast, where N is the number of products
    net_forecast = np.zeros((12, env.n_products))
    
    # Iterate over each month from 1 to 12
    for m, month in enumerate(range(1, 13)):
        # Skip months that have already passed
        if month < current_month:
            continue
        
        # Iterate over each product (GMID)
        for j, g in enumerate(env.gmids):
            # Filter orders that match the criteria: created before or on the current simulation time,
            # not yet shipped, have a valid document number, planned to be shipped in the current month,
            # and match the current GMID
            pred_orders = env.order_book[np.where(
                (env.order_book[:, env.ob_indices['doc_create_time']] <= env.sim_time) & 
                (env.order_book[:, env.ob_indices['shipped']] == 0) &
                (env.order_book[:, env.ob_indices['doc_num']] > 0) &
                (env.order_book[:, env.ob_indices['planned_gi_month']] == month) &
                (env.order_book[:, env.ob_indices['gmid']] == g)
            )]

            # Get unique orders and their quantities
            unique_order, unique_order_id = np.unique(pred_orders[:, env.ob_indices['gmid']], return_inverse=True)
            order_pred_qty = np.bincount(unique_order_id, pred_orders[:, env.ob_indices['order_qty']])
            
            try:
                # Calculate the net forecast by subtracting the predicted order quantity from the monthly forecast
                net_forecast[m, j] = max(env.monthly_forecast[m, j] - order_pred_qty.sum(), 0)
            except IndexError:
                # If there is no forecast for the month, leave the net forecast at 0
                continue
    
    return net_forecast

# TODO: evaluate breaking into individual functions
def get_current_state(env, schedule=None, day=None):
    """
    Get the current state of the environment, which includes inventory levels and optionally scheduled production.

    Args:
        env: The environment object containing inventory and schedule information.
        schedule (optional): The production schedule to consider for future inventory.
        day (optional): The specific day to consider for the state. Defaults to the current simulation time.

    Returns:
        state: The current state of the environment based on the inventory and state setting.
    """
    # Get a copy of the current inventory
    inv = env.inventory.copy()

    # Use the current simulation time if no specific day is provided
    if day is None:
        day = env.sim_time

    # If a schedule is provided, calculate the expected inventory from unbooked production
    if schedule is not None:
        if type(schedule) == tuple:
            h_logger.debug("Schedule is tuple:", day)
        
        # Get expected, unbooked production from the schedule
        pred_production = schedule[np.where(
            (schedule[:, env.sched_indices['cure_end_time']] <= day) &
            (schedule[:, env.sched_indices['booked_inventory']] == 0)
        )]
        
        # Sum scheduled, unbooked production
        un_prod, un_prod_id = np.unique(pred_production[:, env.sched_indices['gmid']], return_inverse=True)
        pred_prod_qty = np.bincount(un_prod_id, pred_production[:, env.sched_indices['prod_qty']])
        
        # Add predicted production to inventory
        if len(pred_prod_qty) > 0:
            prod_idx = np.array([env.gmid_index_map[p] for p in un_prod.astype(int)])
            inv[prod_idx] += pred_prod_qty

        # Get the current product being produced
        current_prod = schedule[
            schedule[:, env.sched_indices['prod_start_time']] == day,
            env.sched_indices['gmid']
        ].astype(int)
        
        # Check if there is a scheduled product for the current day, otherwise indicate shutdown
        if current_prod.size == 0:
            current_prod = 0
    else:
        current_prod = 0

    # Ensure current product is a single integer value
    if type(current_prod) is np.ndarray:
        current_prod = current_prod.take(0)

    # Update the environment's current product if the day is the current simulation time
    if day == env.sim_time:
        env.current_prod = copy(current_prod)

    # Determine the state based on the environment's state setting
    if env.state_setting == 'INVENTORY':
        # State is defined by the inventory levels
        state = inv
    elif env.state_setting == 'IO_RATIO':
        # State is defined as the ratio of inventory to open orders
        order_pred_qty, gmids_to_update = aggregate_orders(env)
        
        # Calculate inventory ratios for the state
        inv_ratios = np.array([inv[env.gmid_index_map[i]] / order_pred_qty[k] for k, i in enumerate(gmids_to_update)])
        state_inv = np.zeros(env.n_products)
        indices_to_update = np.array([env.gmid_index_map[int(i)] for i in gmids_to_update])
        state_inv[indices_to_update] += inv_ratios

        state = state_inv

    # inv_balance_production setting defines the state as the inventory
    # balance concatenated with the current production state.
    elif env.state_setting == 'INV_BALANCE_PRODUCTION':
        # Aggregate open orders
        order_pred_qty, gmids_to_update = aggregate_orders(env)
        state_inv = inv.copy()
        indices_to_update = np.array([env.gmid_index_map[int(i)] for i in gmids_to_update])
        
        # Ensure array is not empty before subtracting order quantities
        if gmids_to_update.size > 0:
            state_inv[indices_to_update] -= order_pred_qty

        # Create a one-hot vector for the current production state
        one_hot = np.zeros(env.n_products)

        # Set the one-hot vector to 0 if the plant is shut down
        if current_prod != 0:
            one_hot[env.gmid_index_map[current_prod]] = 1
        
        # Concatenate the one-hot production state with the inventory state
        state = np.hstack([one_hot, state_inv])

    # The current state is defined as the ratio of inventory
    # to open orders available for that day and the current
    # production
    elif env.state_setting == 'IO_PRODUCT':            
        # Aggregate open orders on the books
        # Filtering by sim_time ensures all orders are already entered
        # Filtering shipped ensures that only open orders are considered
        order_pred_qty, gmids_to_update = aggregate_orders(env)
        
        # Calculate inventory ratios for the state prediction
        prod_index = np.array([env.gmid_index_map[i] for i in gmids_to_update])
        inv_ratios = np.array([inv[i] / order_pred_qty[k] for k, i in enumerate(prod_index)])
        state_inv = np.zeros(env.n_products)
        
        # Update the state inventory with the calculated ratios
        state_inv[prod_index] += inv_ratios

        # Include product to be produced in state prediction as a one-hot vector
        # Set the one-hot vector to 0 if the plant is shut down
        one_hot = np.zeros(env.n_products)     
        if current_prod != 0:
            one_hot[env.gmid_index_map[current_prod]] = 1
        
        # Concatenate the one-hot production state with the inventory ratios
        state = np.hstack([one_hot, state_inv])

    # Add net forecast to state
    elif env.state_setting == 'CONCAT_FORECAST':
        # Aggregate open orders
        order_pred_qty, gmids_to_update = aggregate_orders(env)
        state_inv = inv
        indices_to_update = np.array([env.gmid_index_map[int(i)] for i in gmids_to_update])
        
        # Ensure array is not empty before subtracting order quantities
        if gmids_to_update.size > 0:
            state_inv[indices_to_update] -= order_pred_qty
        
        # Create a one-hot vector for the current production state
        one_hot = np.zeros(env.n_products)
        if current_prod != 0:
            one_hot[env.gmid_index_map[current_prod]] = 1
        
        # Get the net forecast and flatten it
        net_forecast = get_net_forecast(env).flatten()
        
        # Concatenate the one-hot production state, inventory state, and net forecast
        state = np.hstack([one_hot, state_inv, net_forecast])
        
    return state

# Get state definition
# TODO: Possibly easier to simply get the state and return the dimensions
def observation_space(env):
    """
    Define the observation space based on the environment's state setting.

    Args:
        env: The environment object containing product data and state settings.

    Returns:
        np.ndarray: An array representing the observation space.
    """
    observation_space = []

    # Return inventory only for state definition
    if env.state_setting == 'INVENTORY':
        # Append GMID (Global Material Identifier) for each product to the observation space
        [observation_space.append(int(x)) for x in env.product_data[:, env.prod_data_indices['gmid']]]

    # Get ratio of inventory/open orders
    elif env.state_setting == 'IO_RATIO' or env.state_setting == 'INV_BALANCE':
        # Append GMID for each product to the observation space
        [observation_space.append(int(x)) for x in env.product_data[:, env.prod_data_indices['gmid']]]

    # Get inventory/order ratio and current product as one-hot vector
    elif env.state_setting == 'IO_PRODUCT' or env.state_setting == 'INV_BALANCE_PRODUCTION':
        # Append GMID for each product to the observation space
        [observation_space.append(int(x)) for x in env.product_data[:, env.prod_data_indices['gmid']]]
        # Double the observation space to include the one-hot vector for the current product
        observation_space = 2 * observation_space

    # Include net forecast in the observation space
    elif env.state_setting == 'CONCAT_FORECAST':
        # Append GMID for each product to the observation space
        [observation_space.append(int(x)) for x in env.product_data[:, env.prod_data_indices['gmid']]]
        # Double the observation space to include the one-hot vector for the current product
        observation_space = 2 * observation_space
        # Concatenate the observation space with zeros for the net forecast (12 months * number of products)
        observation_space = np.hstack([np.array(observation_space), np.zeros(12 * env.n_products)])

    return np.array(observation_space)

# Calculate the customer service level
def get_cs_level(env):
    """
    Calculate the customer service level based on the order book.

    Args:
        env: The environment object containing the order book and simulation time.

    Returns:
        np.ndarray: An array representing the customer service level as [on_time, late, not_shipped] ratios.
    """
    # Get orders that are due by the current simulation time
    orders_due = env.order_book[np.where(
        env.order_book[:, env.ob_indices['planned_gi_time']] <= env.sim_time)]
    
    # Calculate the number of orders that are on-time, late, or haven't shipped
    on_time = orders_due[np.where((orders_due[:, env.ob_indices['shipped']] == 1) &
                                  (orders_due[:, env.ob_indices['on_time']] == 1))].shape[0]
    late = orders_due[np.where((orders_due[:, env.ob_indices['shipped']] == 1) &
                               (orders_due[:, env.ob_indices['on_time']] == -1))].shape[0]
    not_shipped = orders_due[np.where((orders_due[:, env.ob_indices['shipped']] == 0))].shape[0]
    
    # If no orders are due, return zeroes for all service levels
    if orders_due.shape[0] == 0:
        cs_level = np.array([0, 0, 0])
    else:
        # Calculate the ratios of on-time, late, and not shipped orders
        cs_level = np.array([on_time, late, not_shipped]) / orders_due.shape[0]

    return cs_level

# Calculate the cost of holding inventory
def calculate_inventory_cost(env):
    """
    Calculate the cost of holding inventory based on the order book and inventory levels.

    Args:
        env: The environment object containing the order book, inventory, and financial parameters.

    Returns:
        float: The total cost of holding inventory.
    """
    env.order_book = env.order_book.astype(float)
    
    # Aggregate orders based on material code (GMID)
    unique_gmid, gmid_locs, gmid_counts = np.unique(
        env.order_book[:, env.ob_indices['gmid']],
        return_inverse=True,
        return_counts=True)
    
    # Calculate the average variable standard margin (beta_i) for each unique GMID
    beta_i = np.bincount(gmid_locs, env.order_book[:, env.ob_indices['var_std_margin']]) / gmid_counts

    # Add 0 as a placeholder for beta_i if lengths are unequal
    if len(beta_i) < len(env.gmid_index_map):
        for i in env.gmids:
            if i not in unique_gmid:
                beta_i = np.insert(beta_i, env.gmid_index_map[i], 0)
    
    # Determine if OG (Original Goods) is to be included in the calculation
    _og_flag = len(env.inventory) - len(env.gmid_index_map)
    assert _og_flag >= 0, "Discrepancy between GMID's and inventory mapping: {}".format(_og_flag)
    assert _og_flag <= 1, "Discrepancy between GMID's and inventory mapping: {}".format(_og_flag)
    
    # Calculate the total cost of holding inventory
    total_cost = sum([env.inventory[env.gmid_index_map[i] + _og_flag] * beta_i[env.gmid_index_map[i]] 
                      for i in unique_gmid]) * env.working_capital_per * -1
    
    return total_cost

def plot_gantt(env, save_location=None):
    """
    Plot a Gantt chart for the production schedule.

    Args:
        env: The environment object containing product data and schedule actions.
        save_location: Optional; the file path to save the plot.

    Returns:
        None
    """
    # Get available product names and unique product IDs (GMIDs) from the environment
    labels = env.product_data[:, env.prod_data_indices['product_name']]
    unique_products = env.product_data[:, env.prod_data_indices['gmid']].astype(int)

    # Find products that have not been scheduled to ensure proper labeling
    unscheduled_products = [p for p in unique_products if p not in env.containers.actions]

    # Combine actual schedule with unscheduled products
    extended_schedule = np.hstack([env.containers.actions, unscheduled_products])

    # Organize products in a matrix where rows index the product and columns index the day
    gantt_matrix = np.zeros((env.n_products, extended_schedule.shape[0]))

    # Populate the matrix with scheduled product values
    for i, j in enumerate(extended_schedule):
        for k in range(env.n_products):
            if j == k + 1:
                gantt_matrix[k, i] = j

    # Set color scheme for the Gantt chart
    cmap = mpl.cm.get_cmap('Paired')
    c = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Create a figure for the Gantt chart
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    # Cycle through products and plot their schedules
    for i in range(gantt_matrix.shape[0]):
        # Cycle through time slots for each product
        for j, k in enumerate(gantt_matrix[i]):
            if k != 0:
                start = j
                # Get the production duration for the product
                prod_duration = env.product_data[int(k - 1), env.prod_data_indices['min_run_time']].astype(int)
                # Plot the product's schedule as a horizontal bar
                ax.barh(i, prod_duration, left=start, color=c[i])

    # Format the plot
    ax.invert_yaxis()  # Invert the y-axis to have the first product at the top
    ax.grid(color='k', linestyle=':')  # Add grid lines
    pos = np.arange(gantt_matrix.shape[0]) + 0.5
    locsy, labelsy = plt.yticks(pos, unique_products)  # Set y-axis labels to product IDs
    plt.title('Gantt Chart')  # Set the plot title
    plt.xlabel('Day')  # Label the x-axis
    plt.xlim([0, len(env.containers.actions)])  # Set x-axis limits
    plt.ylabel('Product')  # Label the y-axis

    # Save the plot if a save location is provided
    if save_location is not None:
        plt.savefig(save_location)

    # Display the plot
    plt.show()
    
def get_state_labels(env, predicted=False):
    """
    Generate state labels based on the environment's state setting and observation space.

    Args:
        env: The environment object containing observation space, product data, and state settings.
        predicted: Optional; if True, prefix labels with 'predicted_'.

    Returns:
        list: A list of state labels.
    """
    # Initialize state labels list
    state_labels = []
    # Get the size of the observation space
    obs_space = env.observation_space.shape[0]
    # Get product names from the environment
    prod_names = env.product_data[:, env.product_data_cols.index('product_name')].astype(str)
    # Create state labels for each product
    state_labels = ['state_' + i.lower() for i in prod_names]

    # Check if the observation space matches the number of products
    if obs_space == env.n_products:
        pass  # No additional labels needed

    # Check if the observation space includes an additional state for 'state_og'
    elif obs_space == env.n_products + 1:
        state_labels.insert(0, 'state_og')

    # Check if the observation space includes production states
    elif obs_space == 2 * env.n_products + 1:
        # Create production state labels for each product
        prod_state_labels = ['state_production_' + i.lower() for i in prod_names]
        # Add a label for production shut down
        prod_state_labels.insert(0, 'state_production_shut_down')
        # Combine production state labels with the existing state labels
        state_labels = prod_state_labels + state_labels

    # Check if the state setting includes production states
    elif env.state_setting == 'IO_PRODUCT' or env.state_setting == 'INV_BALANCE_PRODUCTION':
        # Create production state labels for each product
        prod_state_labels = ['state_production_' + i.lower() for i in prod_names]
        # Combine production state labels with the existing state labels
        state_labels = prod_state_labels + state_labels

    # Check if the state setting includes forecast data
    elif env.state_setting == 'CONCAT_FORECAST':
        # Create production state labels for each product
        prod_state_labels = ['state_production_' + i.lower() for i in prod_names]
        # Create forecast labels for each month and product
        forecast_labels = ['net_forecast_' + str(j) + '_' + i 
            for i in calendar.month_abbr[1:] for j in env.gmids]
        # Combine production state labels, existing state labels, and forecast labels
        state_labels = prod_state_labels + state_labels + forecast_labels

    # Raise an error if no labeling rule matches the state setting and observation space
    else:
        raise ValueError("No labeling rule for {} state data of length {} and {} products.".format(
            env.settings['STATE_SETTING'], obs_space, env.n_products))

    # Prefix labels with 'predicted_' if the predicted flag is set
    if predicted:
        state_labels = ['predicted_' + i for i in state_labels]

    return state_labels

# Get planning_data headers
def get_planning_data_headers(env):
    """
    Generate headers for the planning data based on the environment's containers and action list.

    Args:
        env: The environment object containing containers and action list.

    Returns:
        tuple: A tuple containing a list of column names and a dictionary of column indices.
    """
    # Get the names of the containers from the environment
    names = env.containers.get_names()
    # Initialize column names with default headers
    col_names = ['planning_day', 'heuristic_flag']
    
    # Append action probabilities to the column names
    for a in env.action_list:
        col_names.append('action_prob_' + str(a))

    # Iterate over each container name
    for i in names:
        # Check if the container has data
        if np.array(getattr(env.containers, i)).size > 0:
            # Stack the container data vertically
            x = np.vstack(getattr(env.containers, i))
            # Get the number of dimensions of the container data
            dims = x.shape[1]
            
            # Generate column names based on the container type
            if i == 'state':
                # Get state labels for the current state
                state_labels = get_state_labels(env, predicted=False)
                col_names = col_names + state_labels
            elif i == 'predicted_state':
                # Get state labels for the predicted state
                state_labels = get_state_labels(env, predicted=True)
                col_names = col_names + state_labels
            elif dims > 1 and dims <= env.n_products:
                # Generate column names for multi-dimensional data
                for j in range(dims):
                    col_names.append(str(i) + '_' + string.ascii_lowercase[j])
            elif dims > env.n_products:
                # Handle off-grade data and differentiate between state settings
                for j in range(dims):
                    if j == 0:
                        col_names.append(str(i) + '_og')
                    else:
                        col_names.append(str(i) + '_' + string.ascii_lowercase[j - 1])
            else:
                # Append the container name as a column name
                col_names.append(str(i))

    # Create a dictionary of column names and their indices for easy reference
    planning_data_indices = {k: i for i, k in enumerate(col_names)}

    return col_names, planning_data_indices

def get_mpc_data_headers(env):
    """
    Generate headers for the MPC (Model Predictive Control) data based on the environment's containers.

    Args:
        env: The environment object containing containers.

    Returns:
        tuple: A tuple containing a list of column names and a dictionary of column indices.
    """
    # Get the names of the containers from the environment
    names = env.containers.get_names()
    # Initialize an empty list for column names
    col_names = []

    # Iterate over each container name
    for i in names:
        # Check if the container has data
        if len(getattr(env.containers, i)) > 0:
            # Stack the container data vertically
            x = np.vstack(getattr(env.containers, i))
            # Get the number of dimensions of the container data
            dims = x.shape[1]

            # Generate column names based on the container type
            if dims > 1 and dims <= env.n_products:
                # Generate column names for multi-dimensional data
                for j in range(dims):
                    col_names.append(str(i) + '_' + string.ascii_lowercase[j])
            elif dims > env.n_products:
                # Handle off-grade data and differentiate between state settings
                for j in range(dims):
                    if j == 0:
                        col_names.append(str(i) + '_og')
                    else:
                        col_names.append(str(i) + '_' + string.ascii_lowercase[j - 1])
            else:
                # Append the container name as a column name
                col_names.append(str(i))

    # Create a dictionary of column names and their indices for easy reference
    planning_data_indices = {k: i for i, k in enumerate(col_names)}

    return col_names, planning_data_indices

def load_current_state_data(settings, path=None):
    """
    Load the current state data from the specified path.

    Args:
        settings: The settings object containing configuration data.
        path: Optional; the file path to load the data from. Defaults to 'production_models/bahia_blanca/'.

    Returns:
        tuple: A tuple containing inventory, order book, and forecast data.
    """
    # Set default path if none is provided
    if path is None:
        path = 'production_models/bahia_blanca/'
    
    # List all files in the specified path
    files = os.listdir(path)
    # Initialize paths for different data types
    inv_path, prod_path, f_path, order_path = None, None, None, None

    # Iterate over each file in the directory
    for f in files:
        # Identify and set paths for inventory, product, forecast, and orders data
        if 'inventory' in f:
            inv_path = os.path.join(path, f)
        elif 'product' in f:
            prod_path = os.path.join(path, f)
        elif 'forecast' in f:
            f_path = os.path.join(path, f)
        elif 'orders' in f:
            order_path = os.path.join(path, f)

    # Load the current order data
    order_book = load_current_order_data(order_path)
    # Load the current forecast data
    forecast = load_current_forecast_data(f_path)
    # Load the current inventory data
    inventory = load_current_inventory_data(inv_path)

    return inventory, order_book, forecast

def load_current_production_data(settings, path):
    """
    Load the current production data and ensure it matches the training data.

    Args:
        settings: The settings object containing configuration data.
        path: The file path to load the production data from.

    Returns:
        tuple: A tuple containing production data and transition matrix.
    """
    # Ensure testing and training values are identical
    try:
        # Load training production data and transition matrix
        train_prods, train_tm = load_scenario_data(settings['PRODUCT_DATA_PATH'])
        training_loaded = True
    except KeyError:
        # Warn if no training environment is found
        warnings.warn('No training environment found, cannot guarantee environments match.')
        answer = input('Continue? y/n')
        if answer == False:
            sys.exit('Program exited.')

    # Load testing production data and transition matrix
    test_prods, test_tm = load_scenario_data(path)

    # Verify that training and testing data match
    if training_loaded:
        assert np.array_equal(train_prods, test_prods), 'Product data for test and train environments do not match.'
        assert np.array_equal(train_tm, test_tm), 'Transition matrices for test and train environments do not match.'

    return test_prods, test_tm

def load_current_schedule(env):
    """
    Load the current production schedule from a DBF file and build a schedule array.

    Args:
        env: The environment object containing settings and fixed planning horizon.

    Returns:
        np.ndarray: A numpy array representing the production schedule.
    """
    # Get the path to the schedule DBF file
    sched_dbf_path = os.path.split(env.settings['PRODUCT_DATA_PATH'])[0]
    sched_dbf_path = os.path.join(sched_dbf_path, "EXPORT.DBF")
    print("Loading Current Schedule from {0}".format(sched_dbf_path))
    
    # Load the DBF file into a DataFrame
    dbf = Dbf5(sched_dbf_path)
    df = dbf.to_dataframe()
    
    # Initialize variables for building the schedule
    sched = []
    b_id = 0
    booked_inv = 0.0
    off_grade = 0.0
    actual_prod = 0.0
    
    # Parse the start date and time of the schedule
    sched_start_row = df.iloc[0, :]
    start_split = sched_start_row["START_DATE"].split("-")
    if len(sched_start_row["START_TIME"]) > 3:
        start_hour = int(sched_start_row["START_TIME"][:2])
    else:
        start_hour = int(sched_start_row["START_TIME"][0])
    start_min = int(sched_start_row["START_TIME"][-2:])
    sched_start = datetime(int(start_split[0]), int(start_split[1]), int(start_split[2]), start_hour, start_min)
    
    # Initialize the end datetime of the schedule
    sched_end_dt = sched_start
    idx = 0
    
    # Cut the current schedule to only include elements within the fixed planning horizon
    while sched_end_dt < sched_start + timedelta(hours=24.0 * env.fixed_planning_horizon):
        row = df.iloc[idx, :]
        gmid = int(row["GMID"])
        prod_rate = row["PROD_RATE"]
        prod_qty = row["QUANTITY"]
        prod_time = prod_qty / prod_rate
        
        # Parse the start date and time of the current production
        start_split = row["START_DATE"].split("-")
        if len(row["START_TIME"]) > 3:
            start_hour = int(row["START_TIME"][:2])
        else:
            start_hour = int(row["START_TIME"][0])
        start_min = int(row["START_TIME"][-2:])
        datetime_start = datetime(int(start_split[0]), int(start_split[1]), int(start_split[2]), start_hour, start_min)
        
        # Calculate production start and end times
        prod_start = datetime_start - sched_start
        prod_start = prod_start.total_seconds() / (60 * 60)
        prod_end = int(prod_start + prod_time)
        
        # Define cure time and calculate cure end time
        cure_time = 24
        cure_end = prod_end + cure_time
        
        # Get the inventory index for the current GMID
        inv_index = env.gmids.index(gmid) + 1
        
        # Build the schedule row
        sched_row = [b_id, gmid, prod_rate, prod_qty, prod_time, prod_start, prod_end, 
                     cure_time, cure_end, booked_inv, inv_index, off_grade, actual_prod]
        b_id += 1
        sched.append(sched_row)
        idx += 1
        
        # Update the end datetime of the schedule
        sched_end_dt = datetime_start
    
    # Convert the schedule list to a numpy array
    schedule = np.stack(sched)
    return schedule

def load_current_order_data(path=None):
    """
    Load the current order data from the specified path.

    Args:
        path: Optional; the file path to load the order data from.

    Returns:
        DataFrame: A DataFrame containing the current order data.
    """
    try:
        # Attempt to load the current order data using a specific method
        orders = _load_current_order_data()
    except NotImplementedError:
        # If the method is not implemented, load the order data from a file
        orders = _load_state_data_by_file(path, 'Orders', True)

    return orders

def load_current_forecast_data(path=None):
    """
    Load the current forecast data from the specified path or system.

    Args:
        path: Optional; the file path to load the forecast data from.

    Returns:
        DataFrame or ndarray: The forecast data.
    """
    try:
        # Attempt to load the current forecast data using a specific method
        forecast = _load_current_forecast_data()
    except NotImplementedError:
        # If the method is not implemented, load the forecast data from a file
        forecast = _load_state_data_by_file(path, 'Forecast', True)

    return forecast

def load_current_inventory_data(path=None):
    """
    Load the current inventory data from the specified path or system.

    Args:
        path: Optional; the file path to load the inventory data from.

    Returns:
        DataFrame or ndarray: The inventory data.
    """
    try:
        # Attempt to load the current inventory data using a specific method
        inventory = _load_current_inventory_data()
    except NotImplementedError:
        # If the method is not implemented, load the inventory data from a file
        inventory = _load_state_data_by_file(path, 'Inventory', True)

    return inventory

# TODO: Complete the following function
def _load_current_inventory_data():
    """
    Load the current inventory data from SAP HANA or a relevant internal system.

    Raises:
        NotImplementedError: If the inventory data system is not defined.
    """
    raise NotImplementedError('Inventory data system not defined.')

# TODO: Complete the following function
def _load_current_forecast_data():
    """
    Load the current forecast data from SAP HANA or a relevant internal system.

    Raises:
        NotImplementedError: If the forecast data system is not defined.
    """
    raise NotImplementedError('Forecast data system not defined.')

# TODO: Complete the following function
def _load_current_order_data():
    """
    Load the current order data from SAP HANA or a relevant internal system.

    Raises:
        NotImplementedError: If the order data system is not defined.
    """
    raise NotImplementedError('Order data system not defined.')

def _load_state_data_by_file(path, dtype='', pandas=False):
    """
    Load state data from a file and handle file format and modification checks.

    Args:
        path: str; the file path to load the data from.
        dtype: str; the type of data being loaded (e.g., 'Inventory', 'Forecast').
        pandas: bool; if True, return a pandas DataFrame, otherwise return a numpy array.

    Returns:
        DataFrame or ndarray: The loaded data.

    Raises:
        ValueError: If the file extension is not supported.
    """
    # Check the last modified date of the file
    today = datetime.now().date()
    file_last_modified = datetime.utcfromtimestamp(os.path.getmtime(path)).date()
    if today > file_last_modified:
        # Prompt the user if the file is old
        user_input = input('{} file was last modified on {}'.format(
            dtype, file_last_modified) + ' Do you want to continue working with' + \
            ' this data? (y/n)\n>>>>>>\t')
        if str2bool(user_input) == False:
            sys.exit('Program exited.')

    # Supported file extensions
    supported_extensions = ['csv', 'xlsx', 'pkl']
    # Determine the file extension
    extension = os.path.basename(path).split('.')[-1].lower()

    # Load the data based on the file extension
    if extension == 'csv':
        data = pd.read_csv(path)
        if data.shape[1] <= 1:
            # Try another separator if the data has only one column
            data = pd.read_csv(path, sep=';')
    elif extension == 'xlsx':
        data = pd.read_excel(path, dtype=str)
    elif extension == 'pkl':
        data = pickle.load(open(path, 'rb'))
    else:
        # Raise an error if the file extension is not supported
        raise ValueError('Extension {} not supported.'.format(extension) + \
            ' Ensure file is in one of the following formats: {}'.format(
                supported_extensions))

    # Process the data if it is a DataFrame
    if isinstance(data, pd.core.frame.DataFrame):
        try:
            # Drop the 'Unnamed: 0' column if it exists
            data = data.drop('Unnamed: 0', axis=1)
        except KeyError:
            pass
        # Convert to numpy array if pandas is False
        if not pandas:
            data = data.values

    return data

def XLTableExpandToDataFrame(location, limit=1000, index=1):
    '''
    Convert an Excel table starting from a given cell location to a pandas DataFrame.

    Inputs
    =========================================================================
    location: openpyxl.cell.cell.cell to give location of the top left
        corner of the relevant table
    limit: int that limits the table size
    index: 0 or 1 where 0 indicates a numeric index the size of the frame
        and 1 indicates the first column of the table is used as the index

    Returns
    =========================================================================
    DataFrame: A pandas DataFrame containing the table data.
    '''
    # Ensure the index parameter is either 0 or 1
    assert index == 0 or index == 1, 'Index value must be either 0 or 1.'

    # Initialize variables to build the DataFrame
    frame = []
    frame_cell = location
    cols_count = 0
    rows_count = 0
    frame_cols = frame_cell
    frame_rows = frame_cols

    # Iterate over the rows until a blank cell is found or the limit is reached
    while not frame_rows.value is None and rows_count < limit:
        train_frame_row = []
        # Iterate over the columns until a blank cell is found or the limit is reached
        while not frame_cols.value is None and cols_count < limit:
            # Append the cell value to the current row
            train_frame_row.append(frame_cols.value)
            cols_count += 1
            # Move to the next cell in the current row
            frame_cols = frame_cell.offset(rows_count, cols_count)
        # Append the current row to the frame
        frame.append(train_frame_row)
        cols_count = 0
        rows_count += 1
        # Move to the next row
        frame_rows = frame_cell.offset(rows_count, cols_count)
        frame_cols = frame_rows

    # Convert the list of lists to a numpy array
    frame = np.vstack(frame)

    # Create a DataFrame using the first row as column headers
    if index == 1:
        frame = pd.DataFrame(data=frame[1:, 1:], columns=frame[0, 1:], 
                             index=frame[1:, 0])
    else:
        frame = pd.DataFrame(data=frame[1:, :], columns=frame[0, :], 
                             index=np.arange(frame.shape[0] - 1))

    # Convert numeric columns to appropriate numeric types
    frame = frame.apply(pd.to_numeric, downcast="float", errors="ignore")

    return frame

def load_product_data_from_excel(product_data_path):
    """
    Load product data from an Excel file and transform it into various DataFrames.

    Args:
        product_data_path (str): The file path to the Excel file containing product data.

    Returns:
        tuple: A tuple containing the transformed DataFrames and arrays.
    """
    # Load the Excel workbook
    wb = pyxl.load_workbook(product_data_path, data_only=True)
    
    # Load train data
    # Get the location of the 'Trains' table and convert it to a DataFrame
    trains_loc = wb['Overview'][wb.defined_names['Trains'].value.split("!")[1]].offset(1, 0)
    trains_df = XLTableExpandToDataFrame(trains_loc)
    
    # Load production data
    # Get the location of the 'Products' table and convert it to a DataFrame
    prod_loc = wb['Overview'][wb.defined_names['Products'].value.split('!')[1]].offset(1, 0)
    prod_df = XLTableExpandToDataFrame(prod_loc, index=0)
    # Insert the train number into the production DataFrame
    prod_df.insert(0, 'train', trains_df['train_number'].values[0].astype(int))
    
    # Load transition data
    # Get the location of the 'ProductsTransition' table and convert it to a DataFrame
    trans_loc = wb['Overview'][wb.defined_names['ProductsTransition'].value.split('!')[1]].offset(1, 0)
    trans_df = XLTableExpandToDataFrame(trans_loc, index=1)
    
    # Transform transition data
    # Get the maximum batch size from the production DataFrame
    max_losses = prod_df['batch_size'].max().astype(str)
    # Replace characters in the transition DataFrame and convert to float
    transition_matrix = replace_chars_vec(max_losses, trans_df.values).astype(float)
    # Add startup values to the transition matrix
    transition_matrix = np.hstack([prod_df['startup'].values.reshape(-1, 1), transition_matrix])
    transition_matrix = np.vstack([np.hstack([0, prod_df['startup']]), transition_matrix])

    # Get final products
    # Get the location of the 'ProductsFinished' table and convert it to a DataFrame
    zfin_loc = wb['Overview'][wb.defined_names['ProductsFinished'].value.split('!')[1]].offset(1, 0)
    zfin_list = XLTableExpandToDataFrame(zfin_loc)['gmid'].astype(int).values

    # Get ZFIN-ZEMI/GMID mappings

    # Locate the 'ProductsFinished' table in the Excel workbook and convert it to a DataFrame
    zfin_loc = wb['Overview'][wb.defined_names['ProductsFinished'].value.split('!')[1]].offset(1, 0)
    zfin_df = XLTableExpandToDataFrame(zfin_loc, index=0)

    # Extract the ZEMI part of the product names from the production DataFrame
    zemi = prod_df['product_name'].map(lambda x: ' '.join(x.split(' ')[:-2]))

    # Extract the ZFIN part of the product names from the ZFIN DataFrame
    zfin = zfin_df['product_name'].map(lambda x: ' '.join(x.split(' ')[:-2]) 
                            if x.split(' ')[-1] == 'KG' else
                            ' '.join(x.split(' ')[:-1]))

    # Create a copy of the production DataFrame and add the ZEMI column
    prod_df2 = prod_df.copy()
    prod_df2['zemi'] = zemi

    # Add the ZEMI column to the ZFIN DataFrame
    zfin_df['zemi'] = zfin

    # Merge the ZFIN DataFrame with the production DataFrame on the ZEMI column
    merged = zfin_df.merge(prod_df2, on='zemi', how='left')

    # Parse the packaging information from the product names in the merged DataFrame
    merged['packaging'] = merged['product_name_x'].map(lambda x: parse_packaging(x))

    # Add an inventory index column to the merged DataFrame
    merged['inventory_index'] = np.arange(len(merged))

    # Create a dictionary mapping ZFIN GMIDs to production GMIDs
    zfin_to_gmid = {i[0]: i[1] for i in merged[['gmid_x', 'gmid_y']].values.astype(int)}

    # Create a dictionary containing ZFIN data
    zfin_data = {int(i[0]): [int(i[1]), i[2], i[3], i[4], i[5], i[6]]
        for i in merged[['gmid_x', 'gmid_y', 
            'product_name_x', 'product_name_y', 
            'packaging', 'batch_size_x',
            'inventory_index']].values}

    # Return the relevant data
    return prod_df.values, transition_matrix, zfin_list, zfin_to_gmid, zfin_data
    
def replace_chars(replacement_value, val):
    """
    Replace all alphabetic characters in a string with a specified replacement value.

    Args:
        replacement_value (str): The value to replace alphabetic characters with.
        val (str or any): The input value which will be converted to a string if not already.

    Returns:
        str: The modified string with alphabetic characters replaced.
    """
    # Convert the input value to a string if it is not already a string
    if type(val) != str:
        val = str(val)
    # Use regular expression to replace all alphabetic characters with the replacement value
    return re.sub("[a-zA-Z]+", replacement_value, val)

# Vectorize the replace_chars function to apply it element-wise to arrays
replace_chars_vec = np.vectorize(replace_chars)

def parse_packaging(desc):
    """
    Parse the packaging type from a product description.

    Args:
        desc (str): The product description.

    Returns:
        str: The packaging type ('bag', 'ss', 'bulk', or '').
    """
    # Check for specific substrings in the description to determine the packaging type
    if 'BG6025' in desc:
        return 'bag'
    elif 'BB1200' in desc:
        return 'ss'
    elif 'BLK' in desc:
        return 'bulk'
    else:
        return ''
    
def process_forecast_data(forecast_data, env):
    """
    Process forecast data to match the environment's GMIDs and reshape it.

    Args:
        forecast_data (pd.DataFrame): The forecast data to be processed.
        env (object): The environment object containing GMIDs and ZFIN data.

    Returns:
        np.ndarray: The processed forecast data.
    """
    # Check if forecast data has already been processed
    if forecast_data.shape[1] == len(env.gmids):
        # If forecast data is a DataFrame, convert it to a NumPy array
        if type(forecast_data) == pd.core.frame.DataFrame:
            forecast_data = forecast_data.values
    else:
        # Filter forecast data to include only rows with ZFIN GMIDs present in the environment
        df = forecast_data.loc[forecast_data['Field-03'].isin(env.zfin.astype(str))]
        assert len(df) > 0, "No matching ZFIN GMID's found in forecast."
        
        # Define categories to melt and columns to use as identifiers
        melt_cats = ['ACTD', 'RSLF', 'HFSF', 'UAH7']
        id_vars = ['Field-03']
        join_cols = ['Field-03', 'Month', 'Year']
        df_reshape = None
        
        # Process each category separately
        for cat in melt_cats:
            # Identify columns related to the current category
            melt_cols = [col for col in df.columns if cat in col]
            [melt_cols.append(i) for i in id_vars]
            
            # Subset the DataFrame to include only relevant columns
            _df_sub = df.loc[:, melt_cols]
            df_sub = pd.DataFrame()
            
            # Ensure numerical columns are formatted as floats
            for col in _df_sub.columns:
                if col in id_vars:
                    df_sub = pd.concat([df_sub, _df_sub[col]], axis=1)
                else:
                    df_sub = pd.concat([df_sub, _df_sub[col].astype(float)], axis=1)
            
            # Aggregate data by identifier variables
            df_agg = df_sub.groupby(id_vars).sum()
            df_agg.reset_index(inplace=True)
            
            # Melt the aggregated DataFrame
            df_melt = pd.melt(df_agg, id_vars=id_vars)
            
            # Extract month and year from the variable column
            df_melt['Month'] = df_melt['variable'].map(lambda x: x.split(' ')[-1])
            df_melt['Year'] = df_melt['Month'].map(lambda x: x.split('/')[-1])
            df_melt['Month'] = df_melt['Month'].map(lambda x: x.split('/')[0])
            df_melt.drop('variable', axis=1, inplace=True)
            
            # Rename the value column to the current category
            col_list = df_melt.columns.tolist()
            col_list[col_list.index('value')] = cat
            df_melt.columns = col_list
            
            # Merge the reshaped data for the current category with the overall reshaped DataFrame
            if df_reshape is None:
                df_reshape = df_melt.copy()
            else:
                df_reshape = df_reshape.merge(df_melt, on=join_cols, how='outer')
                
        # Fill any NaN values in the reshaped DataFrame with 0
        df_reshape.fillna(0, inplace=True)

        # Rename the 'Field-03' column to 'ZFIN'
        col_list = df_reshape.columns.tolist()
        col_list[col_list.index('Field-03')] = 'ZFIN'
        # col_list[col_list.index('Field-04')] = 'ZFIN Name'
        df_reshape.columns = col_list

        # Reorder the columns in the DataFrame
        new_order = ['ZFIN', 'Year', 'Month', 'ACTD', 'RSLF', 'HFSF', 'UAH7']
        df_reshape = df_reshape.loc[:, new_order].copy()

        # Aggregate values by ZFIN and date (Year and Month)
        agg = df_reshape.groupby(['ZFIN', 'Year', 'Month'])[['ACTD', 'RSLF', 'HFSF', 'UAH7']].sum()

        # Filter out rows where the sum of the specified columns is zero
        agg = agg.loc[agg.sum(axis=1).values != 0].reset_index()

        # Map ZFIN values to GMID using the environment's mapping
        agg['GMID'] = agg['ZFIN'].map(lambda x: env.zfin_to_gmid_map[int(x)])

        # Aggregate the forecast data by GMID, Year, and Month, summing the 'RSLF' values
        fcast_agg = agg.groupby(['GMID', 'Year', 'Month'])['RSLF'].sum().reset_index()

        # Create a 'year_mon' column by combining Year and Month into a datetime object
        fcast_agg['year_mon'] = fcast_agg.apply(lambda x: datetime.strptime(str(x.Year) + '-' + str(x.Month), '%y-%m'), axis=1)

        # Get the first day of the current month
        now = pd.Timestamp(date.today().replace(day=1))

        # Calculate the timestamp for the same month next year
        next_year = pd.Timestamp(now.replace(year=now.year + 1, month=now.month - 1))

        # Filter the forecast data to include only the next 12 months
        fcast = fcast_agg.loc[(fcast_agg['year_mon'] >= now) & (fcast_agg['year_mon'] <= next_year)]

        # Convert the GMID column to string type
        fcast['GMID'] = fcast['GMID'].astype(str)

        # Initialize an empty forecast data array with 12 rows (months) and columns equal to the number of products
        forecast_data = np.zeros((12, env.n_products))

        # Populate the forecast data array with the RSLF values for each GMID and month
        for g in env.gmid_index_map.keys():
            for i, m in enumerate(range(1, 13)):
                forecast_data[i, env.gmid_index_map[g]] = fcast.loc[
                    (fcast['Month'] == str(m)) & (fcast['GMID'] == str(int(g)))
                ]['RSLF']

        # Return the processed forecast data
        return forecast_data

def keep_base_name(s):
    """
    Extract and return the base name from a given string.

    Args:
        s (str): The input string from which the base name is to be extracted.

    Returns:
        str: The base name extracted from the input string.
    """
    # Split the input string into a list of words, taking only the first four words
    split = s.split(' ')[:4]
    
    # Check if the last word in the split list is 'HF'
    if split[-1] == 'HF':
        # If the last word is 'HF', join all four words to form the base name
        return ' '.join(split)
    else:
        # If the last word is not 'HF', join only the first three words to form the base name
        return ' '.join(split[:3])

def process_order_data(order_data, env):
    """
    Process order data to match the environment's requirements and reshape it.

    Args:
        order_data (pd.DataFrame): The order data to be processed.
        env (object): The environment object containing order columns and ZFIN data.

    Returns:
        pd.DataFrame: The processed order data.
    """
    # Check if order data is already in the proper format
    if order_data.shape[1] == len(env.order_cols):
        return order_data
    
    # Raise an error if order_data is not a DataFrame
    if type(order_data) != pd.core.frame.DataFrame:
        raise ValueError("order_data loaded as {}; type not supported".format(type(order_data)))
    
    # Rename columns, appending 'Desc' to unnamed columns
    order_data.columns = [j if 'Unnamed:' not in j else order_data.columns[i-1] + ' Desc' for i, j in enumerate(order_data.columns)]
    
    # Filter orders to include only those with ZFIN materials present in the environment
    orders_sub = order_data.loc[order_data['Material'].isin(env.zfin.astype(str))]
    
    # Filter orders to include only specific document types
    doc_types = ['ZOR', 'ZSO', 'ZBRJ', 'ZFD', 'ZRI', 'ZBRI', 'ZVER', 'ZLOR']
    orders_sub = orders_sub.loc[orders_sub['Sales Doc. Type'].isin(doc_types)]
    
    # Convert order quantities from KG to MT and round to three decimal places
    orders_sub['order_qty'] = np.round(orders_sub['Weight - Net (w/o UoM)'].astype(float) / 1000, 3)
    
    # Define a mapping of original time columns to new order book columns
    time_cols = {
        'Dt - (OI) Customer Requested Del (Confirmed)': 'cust_req_date',
        'Dt - (DH) Goods Issue Actual': 'actl_gi_time',
        'Dt - (DH) Goods Issue Plan': 'planned_gi_time',
        'Dt - (OH) Created On': 'doc_create_time'
    }
    
    # Convert time strings to datetime objects, handling missing values by setting them to a future date
    for key in time_cols.keys():
        orders_sub[time_cols[key]] = orders_sub[key].map(
            lambda x: datetime.strptime(str(x), '%m/%d/%Y') if x != '#' else datetime.strptime('01/01/2100', '%m/%d/%Y')
        )
        
        # Extract the month from the planned goods issue date
        if key == 'Dt - (DH) Goods Issue Plan':
            orders_sub['planned_gi_month'] = orders_sub[key].map(lambda x: datetime.strptime(str(x), '%m/%d/%Y').month)
        
        # Calculate the time difference in days from the environment's start time
        times = (orders_sub[time_cols[key]] - env.start_time).map(lambda x: x.days)
        
        # Convert the time difference to hours if the base time unit is set to 'HOUR'
        if env.settings['BASE_TIME_UNIT'] == 'HOUR':
            times *= 24
        
        # Update the time columns with the calculated time differences
        orders_sub[time_cols[key]] = times
    
    # Define a mapping of original column names to new order book column names
    col_name_map = {
        'Sales Document': 'doc_num',
        'Material': 'gmid'
    }

    for k in col_name_map.keys():
    # Copy the values from the original columns to the new columns based on the mapping
        orders_sub[col_name_map[k]] = orders_sub[k].copy()

        # Initialize the 'shipped' column to 1 (indicating the order has been shipped)
        orders_sub['shipped'] = 1

        # Set 'shipped' to 0 for orders with an actual goods issue time greater than 2 years (730 days)
        orders_sub.loc[orders_sub['actl_gi_time'] > 365 * 2, 'shipped'] = 0

        # Initialize additional columns with default values
        orders_sub['on_time'] = 0
        orders_sub['late_time'] = 0
        orders_sub['cust_segment'] = 1
        orders_sub['var_std_margin'] = 0

        # Convert the 'doc_num' column to integer type
        orders_sub['doc_num'] = orders_sub['doc_num'].astype(int)

        # Calculate the 'late_time' as the difference between actual goods issue time and customer requested date
        orders_sub['late_time'] = orders_sub['actl_gi_time'] - orders_sub['cust_req_date']

        # Map the 'gmid' column using the environment's ZFIN to GMID mapping
        orders_sub['gmid'] = orders_sub['gmid'].map(lambda x: env.zfin_to_gmid_map[int(x)])

        # Copy the 'Material' column to the 'zfin' column
        orders_sub['zfin'] = orders_sub['Material'].copy()

        # Initialize the 'shipped' column to 1 again (this seems redundant)
        orders_sub['shipped'] = 1

        # Select the columns specified in the environment's order columns and convert to a NumPy array
        orders = orders_sub[env.order_cols].values

        # Set 'shipped' to 0 for future orders (actual goods issue time >= 2 years)
        orders[np.where(orders[:, env.ob_indices['actl_gi_time']] >= 365 * 2)[0], env.ob_indices['shipped']] = 0

        # Set 'late_time' to 0 for orders that have not been shipped
        orders[np.where(orders[:, env.ob_indices['shipped']] == 0)[0], env.ob_indices['late_time']] = 0

    # Return the processed orders
    return orders

def determine_date_format(date_series):
    """
    Determine the date format of a series of date strings.

    Args:
        date_series (pd.Series): A series of date strings.

    Returns:
        str: The date format string.
    """
    # Define the labels for month, day, and year
    labels = ['%m', '%d', '%Y']
    
    # Split the date strings into components and convert to integers
    dates = np.vstack(date_series.map(lambda x: re.split('\W+', x))).astype(int)
    
    # Create a dictionary mapping the position of the date components to their labels
    d = {j: labels[i] for i, j in enumerate(np.argsort(np.max(dates, axis=0)))}
    
    # Join the labels to form the date format string
    date_format_string = '-'.join([d[k] for k in range(3)])
    
    return date_format_string

def convert_date_series(date, date_format_string):
    """
    Convert a date string to a datetime object using the given date format string.

    Args:
        date (str): The date string to be converted.
        date_format_string (str): The date format string.

    Returns:
        datetime: The converted datetime object.
    """
    # Replace non-word characters with hyphens in the date string
    date_re = re.sub('\W+', '-', date)
    
    # Convert the date string to a datetime object
    return datetime.strptime(date_re, date_format_string)

def process_inventory_data(inventory_data, env):
    """
    Process inventory data to match the environment's requirements and reshape it.

    Args:
        inventory_data (pd.DataFrame): The inventory data to be processed.
        env (object): The environment object containing ZFIN and GMID mappings.

    Returns:
        np.ndarray: The processed and sorted inventory data.
    """
    # Check if inventory data is already in the proper format
    if inventory_data.shape[0] == 1:
        return inventory_data
    else:
        # Determine the date format of the 'Calendar Day' column
        date_format_string = determine_date_format(inventory_data['Calendar Day'])
        
        # Convert the 'Calendar Day' column to datetime objects
        inventory_data['datetime'] = inventory_data['Calendar Day'].map(
            lambda x: convert_date_series(x, date_format_string))
        
        # Filter the inventory data to include only the most recent date
        data_recent = inventory_data.loc[
            inventory_data['datetime'] == inventory_data['datetime'].max()]
        
        # Further filter the data to include only materials present in the environment's ZFIN list
        data_recent = data_recent.loc[data_recent['Material'].isin(env.zfin)]
        
        # Define the characters to be removed from the quantity strings
        remove = string.punctuation
        remove = remove.replace("-", "")  # don't remove hyphens
        remove = remove.replace(".", "")  # don't remove periods
        
        # Create the regex pattern for removing unwanted characters
        pattern = r"[{}]".format(remove)
        
        # Convert the 'Inventory Balance (KG)' column to float values
        data_recent['quantity'] = data_recent['Inventory Balance (KG)'].map(
            lambda x: float(re.sub(pattern, "", x)))
        
        # Map the 'Material' column to GMIDs using the environment's ZFIN to GMID mapping
        data_recent['gmid'] = data_recent['Material'].map(
            lambda x: env.zfin_to_gmid_map[int(x)])
        
        # Group the data by GMID and sum the quantities, converting from KG to MT
        inventory = data_recent.groupby(['gmid'])['quantity'].sum() / 1000
        
        # Initialize an array to hold the sorted inventory data
        inventory_sorted = np.zeros(len(inventory))
        
        # Populate the sorted inventory array using the environment's GMID index mapping
        for i in env.gmid_index_map.keys():
            inventory_sorted[env.gmid_index_map[i]] += inventory[i]
        
        return inventory_sorted

def load_yaml_data(env):
    """
    Load and process YAML data for the environment.

    Args:
        env (object): The environment object containing settings and data.
    """
    # Check if the OFFGRADE_GMID setting is a float and convert it to an integer if true
    if float(env.settings["OFFGRADE_GMID"]) == type(1.0):
        env.OffGradeGMID = int(env.settings["OFFGRADE_GMID"])
    else:
        env.OffGradeGMID = 9999  # Default value if OFFGRADE_GMID is not a float

    # Construct the file name for the parameter YAML file
    file_name = "{0}.yaml".format(env.settings["PARAMETER_FILE"])
    par_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameter_files", file_name)
    
    # Open and load the YAML file
    stream = open(par_file, 'r')
    env.product_data_yaml = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()
    
    # Extract the list of products from the YAML data
    env.products = env.product_data_yaml['products']
    env.n_products = len(env.products)  # Number of products

    # Initialize dictionaries for standard batch times, sizes, and cure times
    StandardBatchTimes = dict()
    StandardBatchSizes = dict()
    StandardCureTimes = dict()

    # Set default values for the offgrade GMID
    StandardBatchSizes[env.OffGradeGMID] = 0
    StandardCureTimes[env.OffGradeGMID] = 0
    StandardBatchTimes[env.OffGradeGMID] = 1  # 1 hour of offgrade to schedule delay

    # Initialize GMID mappings and product index dictionary
    env.gmids = {}
    env.gmidProdIndexDict = dict()
    env.gmids["offgrade"] = env.OffGradeGMID
    env.gmidProdIndexDict[env.OffGradeGMID] = 0  # Index for offgrade

    # Populate the GMID mappings and product index dictionary for each product
    p_ind = 1
    for p in env.products:
        gmid = int(env.product_data_yaml[str(p) + "_GMID"])
        env.gmids[p] = gmid
        env.gmidProdIndexDict[gmid] = p_ind
        StandardBatchSizes[gmid] = env.product_data_yaml[str(p) + "_Batch Size"]
        StandardCureTimes[gmid] = env.product_data_yaml[str(p) + "_CureTime"]
        StandardBatchTimes[gmid] = env.product_data_yaml[str(p) + "_Batch Size"] / env.product_data_yaml[str(p) + "_Rate"]
        p_ind += 1

    # Initialize the startup matrix with zeros
    startupMat = np.zeros([len(env.products)])
    
    # Populate the startup matrix with startup values from the YAML data
    ind_p = 0
    for p in env.products:
        val = env.product_data_yaml[str(p) + "_Startup"]
        startupMat[ind_p] = val
        ind_p += 1

        # Initialize the transition matrix with dimensions (number of products + 1) x (number of products + 1)
    env.transition_matrix = np.empty([len(env.products) + 1, len(env.products) + 1])

    # Set the initial transition cost to 0
    env.transition_matrix[0, 0] = 0

    # Set the startup costs for transitions from and to the initial state
    env.transition_matrix[1:, 0] = startupMat
    env.transition_matrix[0, 1:] = startupMat

    # Populate the transition matrix with values from the YAML data
    ind_p = 1
    for p in env.products:
        ind_pp = 1
        for pp in env.products:
            val = env.product_data_yaml[str(p) + "_" + str(pp)]
            if not val == 'x' and not val == 'X':
                env.transition_matrix[ind_p, ind_pp] = val
            else:
                env.transition_matrix[ind_p, ind_pp] = 100000  # Use a large number to represent an invalid transition
            ind_pp += 1
        ind_p += 1

    # Initialize the rates matrix with default values
    RatesMat = {}
    RatesMat[env.OffGradeGMID] = 1  # No rate for off-grade product

    # Populate the rates matrix with values from the YAML data
    for p in env.products:
        val = env.product_data_yaml[str(p) + "_Rate"]
        g = env.gmids[p]
        if val is not None:
            RatesMat[g] = val
        else:
            RatesMat[g] = np.inf  # Use infinity to represent an undefined rate

    # Create a list of data for each product, including asset number, product name, GMID, batch times, rates, and sizes
    data = [[asset_number, "{0}".format(p).lower(), int(env.gmids[p]),
            StandardBatchTimes[int(env.gmids[p])], prod_time_uom, RatesMat[int(env.gmids[p])], run_rate_uom,
            StandardBatchSizes[int(env.gmids[p])], size_uom]
            for p in env.products]

    # Return the data array and a copy of the transition matrix
    return np.array(data), env.transition_matrix.copy()