# a2c: runs actor critic algorithm
# Author: Christian Hubbs
# christiandhubbs@gmail.com
# 06.02.2019

import numpy as np 
import warnings
from .rl_utils import *
import time
import os
import pandas as pd
from datetime import datetime
import torch
from ..networks.networks import policyEstimator, valueEstimator
from ...scheduler.network_scheduler import network_scheduler, estimate_schedule_value
from ...environments.env_utils import get_planning_data_headers

# Define a2c class
class a2c():

    def __init__(self, env):
        # Initialize the environment
        self.env = env
        # Check and set A2C-specific settings
        self.settings = self.check_a2c_settings(self.env.settings)
        # Initialize policy estimator
        self.policy_est = policyEstimator(self.env, self.settings)
        # Initialize value estimator
        self.value_est = valueEstimator(self.env, self.settings)
        # Initialize planning data and headers
        self.planning_data = None
        self.planning_data_headers = None
        # Set up containers for policy, value, entropy, and total loss
        # TODO: see if namedtuple or other modules in containers yield better performance for data logging.
        self.loss, self.policy_loss, self.policy_grads = [], [], []
        self.entropy_loss, self.value_loss, self.value_grads = [], [], []
        self.kl_div = []

    def check_a2c_settings(self, settings):
        # Add A2C-specific settings here
        return settings

    def test(self, checkpoint, n_tests=10):
        # Define the path to the scenarios directory
        path = Path('scenarios')
        # Create the scenarios directory if it does not exist
        if os.path.exists(path) == False:
            path.mkdir(parents=True, exist_ok=True)

        # Get scenarios from order_book data files in the scenarios directory
        order_books = [s for s in os.listdir(path) if 'order' in s]

        # TODO: Build a scenario generating function in case none are found
        test_data = pd.DataFrame()
        count = 0
        for s in order_books:
            # Load the order book for the current scenario
            ob_path = os.path.join('scenarios', s)
            self.env.reset()
            self.env.order_book = pickle.load(open(ob_path, 'rb'))

            # Run single-episode experiment
            schedule = None
            test_planning_data = []
            for day in range(self.env.n_days):
                # Schedule the network and get planning data
                schedule, _planning_data = network_scheduler(self.env, 
                    self.policy_est, schedule, test=True)
                # Step the environment with the current schedule
                schedule = self.env.step(schedule)
                # Append the planning data for the current day
                test_planning_data.append(_planning_data)
        
            # Calculate customer service level
            cs_level = self.env.get_cs_level()

            # Calculate inventory cost, late penalties, and shipment rewards
            inv_cost = np.round(sum(self.env.containers.inventory_cost), 0)
            late_penalties = np.round(sum(self.env.containers.late_penalties), 0)
            shipment_rewards = np.round(sum(self.env.containers.shipment_rewards), 0)
            total_rewards = inv_cost + late_penalties + shipment_rewards

            # Create a dictionary with the test data for the current scenario
            test_data_dict = {'scenario': ob_path.split('_')[-1].split('.')[0],
                         'algo': self.env.settings['RL_ALGO'],
                         'product_availability': np.round(cs_level[0], 3),
                         'delayed_order': np.round(cs_level[1], 3),
                         'not_shipped': np.round(cs_level[2], 3),
                         'total_rewards': total_rewards,
                         'inv_cost': inv_cost,
                         'late_penalties': late_penalties,
                         'shipment_rewards': shipment_rewards}
            # Append the test data to the DataFrame
            test_data = pd.concat([test_data, pd.DataFrame(test_data_dict, index=[count])])
            count += 1
    
        # Save the test data to a CSV file
        test_data.to_csv(self.settings['DATA_PATH'] + '/checkpoint_test_' \
            + str(checkpoint) + '.csv')
    
        # Save the model checkpoints if the checkpoint is not 100
        if checkpoint != 100:
            torch.save(self.value_est.state_dict(), 
                self.settings['DATA_PATH'] + '/critic_' + str(checkpoint) + '.pt')
            torch.save(self.policy_est.state_dict(), 
                self.settings['DATA_PATH'] + '/actor_' + str(checkpoint) + '.pt')

    def train(self):
        # Set up data storage for logging training rewards and customer service levels
        data_log_header = ['total_reward']
        [data_log_header.append(x) for x in self.env.cs_labels]
    
        # Initialize lists to store training data
        self.training_rewards = []
        self.training_smoothed_rewards = []
        self.training_cs_level = []
        self.batch_states = []
        self.batch_actions = []
        self.batch_rewards = []
        self.batch_value_log = []
    
        # Get the action space from the environment
        action_space = self.env.action_list
    
        # Set discount factor and number of episodes from environment settings
        self.GAMMA = self.env.settings['GAMMA']
        N_EPISODES = self.env.settings['N_EPISODES']
    
        # Loop through each episode
        for ep in range(N_EPISODES):
            # Reset the environment at the start of each episode
            self.env.reset()
            self.value_log = []
    
            # Initialize the schedule and planning data
            self.schedule = None
            _planning_data = []
    
            # Initialize step counter
            self.step = 0
    
            # Loop until the maximum number of steps is reached or the simulation ends
            while True and self.step < 1e6:
                # Skip scheduling if the environment does not call for it
                if self.env.call_scheduler == False:
                    continue
    
                # Schedule the network and get planning data
                self.schedule, planning_data = network_scheduler(self.env,
                    self.policy_est, self.schedule)
    
                # Append planning data if available
                if planning_data is not None:
                    _planning_data.append(planning_data)
    
                # Estimate the value of the current schedule if value estimator is available
                if self.value_est is not None:
                    self.value_log.append(estimate_schedule_value(self.env, 
                        self.value_est, self.schedule))
    
                # Step the environment with the current schedule
                self.schedule = self.env.step(self.schedule)
                self.step += 1
    
                # Break the loop if the simulation time exceeds the number of steps
                if self.env.sim_time >= self.env.n_steps:
                    break

            self.log_episode_data(_planning_data, ep)
            
            # Terminate the episode if the agent is stuck producing only one product
            if len(np.unique(self.env.containers.actions)) == 1:
                break
            
            # UPDATE NETWORKS
            # =============================================================== 
            # Update gradients based on batch_size
            if ((ep + 1) % self.env.settings['BATCH_SIZE'] == 0 and ep > 0) or (
                ep == N_EPISODES - 1 and ep > 0):
                # Update the neural networks with the collected batch data
                self.update_networks()
            
                # Save data since the last batch
                data_since_last_batch = np.array(
                    self.training_rewards[-self.env.settings['BATCH_SIZE']:]).reshape(-1,1)
                cs_level_batch = np.vstack(self.training_cs_level)[-self.env.settings['BATCH_SIZE']:]
                data_since_last_batch = np.hstack((data_since_last_batch, 
                    cs_level_batch))
            
                # Log the data for the current batch
                log_data(data_since_last_batch, self.env.settings, self.env, data_log_header)
                self.log_policy_data()
            
                # Pickle (serialize) the planning data
                planning_data_file = self.env.settings['DATA_PATH'] + '/planning_data.pkl'
                data = open(planning_data_file, 'wb')
                pickle.dump(self.planning_data, data)
                data.close()
            
            # Test the policy at regular intervals
            if ep % (N_EPISODES/5) == 0 or ep == N_EPISODES - 1:
                # Determine the checkpoint value
                chkpt = 100 if ep == N_EPISODES - 1 else int(ep/N_EPISODES*100)
                self.test(chkpt)
            
            converged = check_for_convergence(self.training_rewards, ep, self.env.settings)
            
            # Print episode progress
            max_percentage = 100
            training_completion_percentage = int((ep + 1) / 
                self.env.settings['N_EPISODES'] * 100)
            
            if converged:
                print("Policy Converged")
                print("Episodes {:2d}\nMean Reward (last 100 episodes): {:2f}".format(
                    ep +1, self.training_smoothed_rewards[ep]))
                break
            
            # Save network parameters after training
            path = os.path.join(self.env.settings['DATA_PATH'])
            self.policy_est.saveWeights(path)
            if self.value_est is not None:
                path = os.path.join(self.env.settings['DATA_PATH'])
                self.value_est.saveWeights(path)
            
            return print("Network trained")

    def predict(self):
        # Print the end time for building the schedule
        print("Building schedule until {}".format(self.settings['END_TIME']))
        
        # Set up data storage for logging rewards and customer service levels
        data_log_header = ['total_reward']
        [data_log_header.append(x) for x in self.env.cs_labels]

        # Initialize lists to store prediction data
        self.training_rewards = []
        self.training_smoothed_rewards = []
        self.training_cs_level = []
        self.batch_states = []
        self.batch_actions = []
        self.batch_rewards = []
        self.batch_value_log = []
        
        # Get the action space from the environment
        action_space = self.env.action_list

        # Initialize the value log
        self.value_log = []       
        
        # Load the current schedule from the environment
        self.schedule = self.env.schedule
        _planning_data = []

        # Initialize step counter
        self.step = 0
        
        # Loop until the maximum number of steps is reached or the simulation ends
        while True and self.step < 1e6:

            # Skip scheduling if the environment does not call for it
            if self.env.call_scheduler == False:
                continue
                
            # Schedule the network and get planning data
            self.schedule, planning_data = network_scheduler(self.env,
                self.policy_est, self.schedule)
            
            # Append planning data if available
            if planning_data is not None:
                _planning_data.append(planning_data)

            # Estimate the value of the current schedule if value estimator is available
            if self.value_est is not None:
                self.value_log.append(estimate_schedule_value(self.env, 
                    self.value_est, self.schedule))

            # Step the environment with the current schedule
            self.schedule = self.env.step(self.schedule)
            self.step += 1
            
            # Break the loop if the simulation time exceeds the number of steps
            if self.env.sim_time >= self.env.n_steps:
                break

    def log_episode_data(self, data, episode):
        # Convert value log to a numpy array
        self.value_log = np.array(self.value_log)
        
        # Log batch and episode data at the end of the episode
        # Calculate discounted rewards
        discounted_rewards = discount_returns(
            np.array(self.env.containers.total_reward),
            gamma=self.GAMMA)
        
        # Extend batch rewards with discounted rewards
        self.batch_rewards.extend(discounted_rewards)
        
        # Extend batch actions with actions taken during the episode
        self.batch_actions.extend(self.env.containers.actions)
        
        # Append predicted state to batch states
        self.batch_states.append(self.env.containers.predicted_state)
        
        # Extend batch value log with the value log of the episode
        self.batch_value_log.extend(self.value_log)
        
        # Ensure planning data has correct dimensions
        data = np.vstack(data)[:self.step]
        data = np.hstack([data, 
            self.env.containers.stack_values()[:self.step],
            self.value_log.reshape(-1, 1)[:self.step]][:self.step])
        
        # Initialize planning data dictionary if it does not exist
        if self.planning_data is None:
            self.planning_data = {}
            self.planning_data[episode] = np.dstack([data])
        else:
            self.planning_data[episode] = np.dstack([data])
        
        # Log planning headers if they haven't been logged already
        self.planning_data_headers, self.planning_data_indices = get_planning_data_headers(self.env)
        if self.value_est:
            self.planning_data_headers.append('value_log')
            self.planning_data_indices['value_log'] = max(self.planning_data_indices.values()) + 1
    
        # Get customer service level for the episode
        cs_level_ep = self.env.get_cs_level()
        
        # Append customer service level to training data
        self.training_cs_level.append(cs_level_ep)
        
        # Log training rewards based on the reward function
        if self.env.reward_function == 'OTD1':
            # Get the average daily on-time delivery score
            self.training_rewards.append(
                sum(self.env.containers.total_reward) / self.env.n_days)
        else:
            # Sum total rewards for the episode
            self.training_rewards.append(sum(self.env.containers.total_reward))
        
        # Calculate and log smoothed rewards
        self.training_smoothed_rewards.append(np.mean(
            self.training_rewards[-self.env.settings['BATCH_SIZE']:]))

    def update_networks(self):
        # Stack the batch states vertically
        self.batch_states = np.vstack(self.batch_states)
        # Stack the batch rewards vertically and flatten the array
        self.batch_rewards = np.vstack(self.batch_rewards).ravel()
        # Stack the batch actions vertically and flatten the array
        self.batch_actions = np.vstack(self.batch_actions).ravel()
        # Stack the batch value log vertically and flatten the array
        self.batch_value_log = np.vstack(self.batch_value_log).ravel()
        
        # Normalize rewards if the reward function includes 'VALUE'
        if 'VALUE' in self.env.settings['REWARD_FUNCTION']:
            try:
                self.batch_rewards = z_norm(self.batch_rewards, axis=0)
            except IndexError:
                print(self.batch_rewards.shape, type(self.batch_rewards))
                raise ValueError
        
        # Update value estimator if it exists
        if self.value_est is not None:
            value_loss, value_grads = self.value_est.update(
                states=self.batch_states,
                returns=self.batch_rewards)

            # Adjust rewards by subtracting the value log
            self.batch_rewards = self.batch_rewards - self.batch_value_log

        # Predict current probabilities using the policy estimator
        current_probs = self.policy_est.predict(self.batch_states)
        
        # Update the policy estimator
        loss, policy_loss, entropy_loss, policy_grads = self.policy_est.update(
                states=self.batch_states,
                actions=self.batch_actions,
                returns=self.batch_rewards)

        # Calculate KL Divergence
        new_probs = self.policy_est.predict(self.batch_states)
        new_probs = torchToNumpy(new_probs, device=self.policy_est.device)
        current_probs = torchToNumpy(current_probs, device=self.policy_est.device)
        kl = -np.sum(current_probs * np.log(new_probs / (current_probs + 1e-5)))
        if np.isnan(kl):
            kl = 0

        # Append loss values to respective lists
        self.loss.append(np.mean(loss))
        self.policy_loss.append(np.mean(policy_loss))
        self.entropy_loss.append(np.mean(entropy_loss))
        self.value_loss.append(np.mean(value_loss))
        self.policy_grads.append(policy_grads)
        self.value_grads.append(value_grads)
        self.kl_div.append(kl)

        # Clear batch holders for the next update
        self.batch_states = []
        self.batch_actions = []
        self.batch_rewards = []
        self.batch_value_log = []
        
    def log_policy_data(self):
        # Get the batch size from settings
        batch_size = self.settings['BATCH_SIZE']
        # Construct the model name using environment and RL algorithm settings
        model_name = (self.settings['ENVIRONMENT'] + '_' + \
            self.settings['RL_ALGO'] + '_' + 'LOSS_DATA')
        # Construct the file path for saving the data
        file_name = self.settings['DATA_PATH'] + '/' + model_name + '.txt'
        # Define the names of the values to be logged
        val_names = ['total_loss', 'policy_loss', 'value_loss', 
            'entropy_loss', 'kl_divergence']
        # Collect the recent batch of losses and reshape the array
        policy_data = np.array(
            [self.loss[-batch_size:],
            self.policy_loss[-batch_size:],
            self.value_loss[-batch_size:],
            self.entropy_loss[-batch_size:],
            self.kl_div[-batch_size:]]).reshape(-1, len(val_names))

        # Check if the file does not exist
        if not exists(file_name):
            # Create the directory if it does not exist
            path = Path(self.settings['DATA_PATH'])
            path.mkdir(parents=True, exist_ok=True)
            # Open the file in write mode
            with open(file_name, 'w') as file:
                # Write data headers
                if val_names is not None:
                    [file.write("{:s}\t".format(name)) for name in val_names]
                    file.write("\n")
                # Append data
                if policy_data.ndim < 2:
                    [file.write("{:f}\r".format(entry)) for entry in policy_data]
                if policy_data.ndim == 2:
                    for row in policy_data:
                        [file.write("{:f}\t".format(col)) for col in row]
                        file.write("\r")
                if policy_data.ndim > 2:
                    raise ValueError('More than 2 dimensions in data to be written.')
            file.close()
        else:
            # Open the file in append mode
            with open(file_name, 'a') as file:
                # Append data
                if policy_data.ndim < 2:
                    [file.write("{:f}\r".format(entry)) for entry in policy_data]
                if policy_data.ndim == 2:
                    for row in policy_data:
                        [file.write("{:f}\t".format(col)) for col in row]
                        file.write("\r")
                if policy_data.ndim > 2:
                    raise ValueError('More than 2 dimensions in data to be written.')
            file.close()