# RL Agent: Builds RL agent according to specified algorithms
# Author: Christian Hubbs
# Contact: christiandhubbs@gmail.com
# Date: 04.02.2019

from copy import deepcopy

# Import utility functions and specific RL algorithms
from .rl_algos import rl_utils
from .rl_algos import a2c

def create_agent(env):
	# Check for default RL values and update environment settings
	env.settings = rl_utils.check_settings(env.settings)

	# Get algorithm-specific settings and hyperparameters
	if env.settings['RL_ALGO'] == 'A2C':
		# Create an A2C agent if the RL algorithm is 'A2C'
		agent = a2c.a2c(env)
		# settings = a2c.check_settings(env, settings)  # Uncomment if additional settings check is needed
	elif env.settings['RL_ALGO'] == 'DQN':
		# Raise an error if the RL algorithm is 'DQN' (not yet implemented)
		raise ValueError('RL_ALGO {} not yet implemented'.format(
			env.settings['RL_ALGO']))
	elif env.settings['RL_ALGO'] == 'PPO':
		# Raise an error if the RL algorithm is 'PPO' (not yet implemented)
		raise ValueError('RL_ALGO {} not yet implemented'.format(
			env.settings['RL_ALGO']))
	elif env.settings['RL_ALGO'] == 'TRPO':
		# Raise an error if the RL algorithm is 'TRPO' (not yet implemented)
		raise ValueError('RL_ALGO {} not yet implemented'.format(
			env.settings['RL_ALGO']))
	else:
		# Raise an error if the RL algorithm is not recognized
		raise ValueError('RL_ALGO {} not recognized'.format(
			env.settings['RL_ALGO']))

	return agent