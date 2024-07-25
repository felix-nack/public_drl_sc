# Random Agent
# Author: Christian Hubbs
# Contact: christiandhubbs@gmail.com
# Date: 20.02.2019

import numpy as np 
from ada.environments.demand_models.demand_utils import *

class random_agent():
	# Initialize the random agent with the environment
	def __init__(self, env):
		# Store the action space from the environment
		self.action_space = env.action_list
		# Store the environment
		self.env = env

	# Method to get a random action from the action space
	def get_action(self):
		return np.random.choice(self.action_space)

	# Method to train the agent
	def train(self):
		schedule = None
		# Loop through each step in the environment
		for step in range(self.env.n_steps):
			# Get planning horizon limit and current simulation time
			planning_limit = self.env.sim_time + self.env.fixed_planning_horizon
			# Determine the current planning time
			planning_time = np.max(schedule[:, 
				self.env.sched_indices['prod_start_time']]) if schedule is not None else self.env.sim_time
			# Continue planning until the planning time exceeds the planning limit
			while planning_time < planning_limit:
				# Get a random action
				action = self.get_action()
				# Append the action to the schedule
				schedule = self.env.append_schedule(schedule, action)
				# Update the planning time
				planning_time = np.max(schedule[:, 
					self.env.sched_indices['prod_start_time']])
			# Step the environment with the current schedule
			schedule = self.env.step(schedule)
			# Check the forecast consistency
			check_forecast_consistency(self.env)

