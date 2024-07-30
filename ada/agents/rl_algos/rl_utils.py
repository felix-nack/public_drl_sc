# rl_utils: Reinforcement Learning Utilities
# Christian Hubbs
# 10.03.2018

# Update 27.03.2018: added check_settings to manage RL settings dictionaries

# This file contains a collection of utilities to assist with developing RL
# agents.

# discountReturns: takes a vector of rewards and a discount factor to return
# the discounted returns

# generate_search_dict: develops a dictionary to run a grid search of
# network hyperparameters.

# log_data:

# check_settings:

import numpy as np
import os
from os.path import exists
from datetime import datetime
import time
from pathlib import Path
import warnings
import collections
import pickle
from ...environments.env_utils import get_planning_data_headers

# discount_returns: takes a vector of rewards and a discount factor to return
# the discounted returns
def discount_returns(returns, gamma=0.99):
    # Initialize an array to hold the discounted returns
    discounted_returns = np.zeros_like(returns)
    # Initialize cumulative returns to zero
    cumulative_returns = 0
    # Iterate over the returns in reverse order
    for i in reversed(range(0, len(returns))):
        # Update cumulative returns with the current return and discount factor
        cumulative_returns = cumulative_returns * gamma + returns[i]
        # Store the cumulative return in the discounted returns array
        discounted_returns[i] = cumulative_returns
    # Return the array of discounted returns
    return discounted_returns

# generate_search_dict: develops a dictionary to run a grid search of
# network hyperparameters.
def generate_search_dict(layer_min, node_min, layer_max=None, node_max=None,
                         learning_rate=0.001, value_estimator=True,
                         algo='a2c', num_episodes=2000, gamma=0.99, 
                         convergence_tol=0.02, save_path='default'):
    # Get the current local time
    t = time.localtime()
    # Format the current date as a string
    run_date = ''.join([str(t.tm_year), str(t.tm_mon), str(t.tm_mday)])
    # Initialize an empty dictionary to hold the search parameters
    search_dict = {}
    # Set default values for layer_max and node_max if not provided
    if layer_max is None:
        layer_max = layer_min + 1
    else: 
        layer_max += 1
    if node_max is None:
        node_max = node_min + 1
    else: 
        node_max += 1
    # Define the range of layers and nodes to search over
    layer_range = [layer_min, layer_max]
    node_range = [node_min, node_max]
    # Get the current working directory
    cwd = os.getcwd()
    
    # Initialize a counter for the search dictionary keys
    k = 0
    # Iterate over the range of layers
    for layer in range(min(layer_range), max(layer_range)):
        # Iterate over the range of nodes
        for node in range(min(node_range), max(node_range)):
            # Set the default save path if not provided
            if save_path == 'default':
                data_path = str(cwd + "/" + run_date + "/" + algo + 
                                "_" + str(layer) + "_" + str(node))
                #ckpt_path = str(cwd + "/" + run_date + "/" + algo + 
                #                "_" + str(layer) + "_" + str(node))
                # Append '_baseline' to the path if using a value estimator
                if value_estimator:
                    data_path = data_path + '_baseline'
            else:
                _save_path = save_path
            # Add the current set of hyperparameters to the search dictionary
            search_dict[k] = {'n_hidden_layers': layer,
                              'n_hidden_nodes': node,
                              'learning_rate': learning_rate,
                              'value_estimator': value_estimator,
                              'num_episodes': num_episodes,
                              'gamma': gamma,
                              'convergence_tol': convergence_tol,
                              'checkpoint_path': data_path,
                              'DATA_PATH': data_path}
            # Increment the counter
            k += 1
    
    # Return the search dictionary
    return search_dict

def log_data(data, settings, env, val_names=None):
    # Construct the model name using environment, number of products, and RL algorithm
    model_name = (settings['ENVIRONMENT'] + '_' + str(settings['N_PRODUCTS']) \
        + '_' + settings['RL_ALGO'])
    
    # Convert data to numpy array if it is a list
    if type(data) == list:
        data = np.array(data)
    
    # Construct the file name for logging data
    file_name = settings['DATA_PATH'] + '/' + model_name + '.txt'
    
    # Check if the file does not exist
    if not exists(file_name):
        # Create the directory path if it does not exist
        path = Path(settings['DATA_PATH'])
        path.mkdir(parents=True, exist_ok=True)
        
        # Open the file in write mode
        with open(file_name, 'w') as file:
            # Write initial training information
            file.write("Training began at: {:s}\n".format(settings['START_TIME']))
            file.write("Training completed at: \n")
            file.write(settings['RL_ALGO'] + " algorithm\n")
            file.write("Number of Products {}\n".format(env.n_products))
            file.write("State Setting: {}\n".format(env.state_setting))
            file.write("Reward function: {}\n".format(env.reward_function))
            file.write("Planning Time Horizon = {}\n".format(env.fixed_planning_horizon))
            file.write("Random seed: \n")
            file.write("="*78 + "\n")
            file.write("Episode Results: \n")
            
            # Write data headers if provided
            if val_names is not None:
                [file.write("{:s}\t".format(name)) for name in val_names]
                file.write("\n")
            
            # Write the data
            if data.ndim < 2:
                [file.write("{:f}\r".format(entry)) for entry in data]
            elif data.ndim == 2:
                for row in data:
                    [file.write("{:f}\t".format(col)) for col in row]
                    file.write("\r")
            else:
                # Raise an error if data has more than 2 dimensions
                raise ValueError('More than 2 dimensions in data to be written.')
        # Close the file
        file.close()
    else:
        # Open the file in append mode if it already exists
        with open(file_name, 'a') as file:
            # Append the data
            if data.ndim < 2:
                [file.write("{:f}\r".format(entry)) for entry in data]
            elif data.ndim == 2:
                for row in data:
                    [file.write("{:f}\t".format(col)) for col in row]
                    file.write("\r")
            else:
                # Raise an error if data has more than 2 dimensions
                raise ValueError('More than 2 dimensions in data to be written.')
        # Close the file
        file.close()

    # Ensure the existence of the checkpoint path
    ckpt_path = settings['DATA_PATH']
    if not exists(ckpt_path):
        ckpt_path = Path(ckpt_path)
        ckpt_path.mkdir(parents=True, exist_ok=True)

    # Construct the file name for order statistics
    order_stats_file = settings['DATA_PATH'] + '/order_statistics.pkl'
    
    # Save order statistics if the file does not exist
    if not exists(order_stats_file):
        order_stats_file = open(order_stats_file, 'wb')
        pickle.dump(env.order_statistics, order_stats_file)

def check_for_convergence(data, episode, settings):
    # The model is said to have converged when the average of the last X% of
    # episodes is within the tolerance of the preceding X% of episodes.
    # For example, if we have 100 episodes and set the percentage_check
    # value to 10% with a tolerance of 1%, the model declares convergence
    # after 50 episodes if the average results of episodes 40 to 50 are within
    # the range +/- (1 + epsilon) * average(ep_30 to ep_40).

    # Set the percentage of episodes to check for convergence
    percentage_check = 0.1

    # Ensure the percentage_check is not greater than 50%
    if percentage_check > 0.5:
        return False

    # Check if 'convergence_tol' is in the settings dictionary
    if 'convergence_tol' not in settings.keys():
        return False

    # Ensure at least 2 * percentage_check of episodes are complete
    elif episode >= settings['num_episodes'] * (2 * percentage_check):
        # Calculate the indices for the last 10% and the trailing 10% of episodes
        last_10_per = len(data) - int(settings['num_episodes'] * percentage_check)
        trailing_10_per = len(data) - int(settings['num_episodes'] * 2 * percentage_check)

        # Calculate the mean of the last 10% of episodes
        mean_last_10 = np.mean(data[-int(settings['num_episodes'] * percentage_check):])

        # Calculate the mean of the trailing 10% of episodes
        mean_trailing_10 = np.mean(data[-int(settings['num_episodes'] * 2 * percentage_check):
                                        -int(settings['num_episodes'] * percentage_check)])

        # Check if the mean of the last 10% is within the tolerance of the trailing 10%
        if mean_last_10 <= (1 + settings['convergence_tol']) * mean_trailing_10 and (
            1 - settings['convergence_tol']) * mean_trailing_10:
            # Print convergence information
            print("Policy converged after: {:d}".format(episode))
            print("Mean last 10%: {:.5f}".format(mean_last_10))
            print("Mean trailing 10%: {:.5f}".format(mean_trailing_10))
            return True
    else:
        return False

# Check RL settings to ensure all values are properly entered and available
def check_settings(settings=None):
    '''
    Input
    settings: dict of values required to parameterize the simulation 
        environment. Missing values or None is permissible as these will
        be populated with defaults.

    Output
    settings: dict of values required to completely specify the simulation
        environment.
    '''
    # Define default settings for the RL environment
    defaults = {
        'RL_ALGO': 'A2C',
        'N_EPISODES': 1000,
        'BATCH_SIZE': 10,
        'GAMMA': 0.99,
        'N_HIDDEN_NODES': 32,
        'N_HIDDEN_LAYERS': 8,
        'ACTIVATION_FUNCTION': 'ELU',
        'LEARNING_RATE': 1E-2,
        'ACTOR_LR': 0.0,
        'CRITIC_LR': 0.0,
        'BIAS': True,
        'BETA': 1E-3,
        'GRADIENT_CLIPPING': False,
        'DEVICE': 'CPU',
        'FRAMEWORK': 'PYTORCH',
        'PERIODIC_TESTING': True # Enables testing to occur at 20% intervals
    }

    # If no settings are provided, use the default settings
    if settings is None:
        settings = defaults
    else:
        # Ensure all default keys are present in the provided settings
        for key in defaults.keys():
            if key not in settings.keys():
                settings[key] = defaults[key]
            elif defaults[key] is not None:
                # Convert the provided setting to the type of the default value
                settings[key] = type(defaults[key])(settings[key])
                
    # Generate the data path if it is not provided in the settings
    if 'DATA_PATH' not in settings.keys():
        default_data_path = os.getcwd() + "/RESULTS/" + settings['RL_ALGO'].upper() + '/' \
            + datetime.now().strftime("%Y_%m_%d_%H_%M")
        settings['DATA_PATH'] = default_data_path

    # Create the directory for the data path if it does not exist
    os.makedirs(settings['DATA_PATH'], exist_ok=True)

    return settings

def z_norm(x, axis=1):
    # Ensure x is at least 2D if axis is 1
    if np.ndim(x) <= 1 and axis == 1:
        x = x.reshape(-1, 1)
    try:
        # Calculate z-normalization (standard score)
        norms = (x - x.mean(axis=axis)) / x.std(axis=axis)
    except Warning:
        # Handle potential division by zero by adding a small constant
        norms = (x - x.mean(axis=axis)) / (x.std(axis=axis) + 1E-6)
    return norms

class ExperienceBuffer():
    def __init__(self, capacity):
        # Initialize a deque with a fixed capacity to store experiences
        self.buffer = collections.deque(maxlen=capacity)
        # Define the structure of an experience tuple
        self.experience = collections.namedtuple('Experience',
            field_names=['state', 'action', 'reward', 
                'step_number', 'next_state'])

    def get_length(self):
        # Return the current number of experiences in the buffer
        return (len(self.buffer))

    def append(self, state, action, reward, step_number, next_state):
        # Create a new experience tuple and append it to the buffer
        exp = self.experience(state, action, reward, step_number, next_state)
        self.buffer.append(exp)

    def sample(self, batch_size):
        # Randomly sample a batch of experiences from the buffer
        indices = np.random.choice(self.get_length(), 
            batch_size, replace=False)
        # Unzip the sampled experiences into separate arrays
        states, actions, rewards, step_numbers, next_states = zip(* 
            (self.buffer[i] for i in indices))
        # Convert lists to numpy arrays for further processing
        states = np.array([np.array(states[i]) for i in range(batch_size)])
        next_states = np.array([np.array(next_states[i]) 
            for i in range(batch_size)])
        actions = np.array(actions, dtype=np.int)
        rewards = np.array(rewards, dtype=np.float)
        steps = np.array(step_numbers)

        return states, actions, rewards, steps, next_states

def _log_planning_data(model, data, algo):
    # Stack the data vertically and limit to the number of days in the environment
    data = np.vstack(data)[:model.env.n_days]
    if algo == 'dqn':
        # For DQN, include additional data such as container values and Q-values
        data = np.hstack([data,
            model.env.containers.stack_values()[:model.env.n_days],
            model._qvals
            ])
        if model.planning_data_headers is None:
            # Generate headers for the planning data if not already set
            headers, _ = get_planning_data_headers(model.env)
            for a in model.env.action_list:
                headers.append('qval_' + str(a))

            planning_data_indices = {k: i for i, k in enumerate(headers)}
            model.planning_data_headers = headers
            model.planning_data_indices = planning_data_indices

    if model.planning_data is None:
        # Initialize planning data if it does not exist
        model.planning_data = np.dstack([data])
    else:
        # Append new data to the existing planning data
        model.planning_data = np.dstack([model.planning_data,
            data])

def torchToNumpy(tensor, device='cpu'):
    # Convert a PyTorch tensor to a NumPy array, handling device placement
    if device == 'cuda':
        return tensor.detach().cpu().numpy()
    else:
        return tensor.detach().numpy()

def check_device_settings(settings):
    # Check the device setting and return the appropriate device string
    if settings['DEVICE'].upper() == 'CPU':
        return 'cpu'
    elif settings['DEVICE'].upper() == 'CUDA' or settings['DEVICE'].upper() == 'GPU':
        return 'cuda'
    else:
        # Raise an error if the device setting is not recognized
        raise ValueError('Device {} not recognized. Define either CPU or GPU.'.format(settings['DEVICE']))