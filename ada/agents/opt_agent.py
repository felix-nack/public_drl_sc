# Opt Agent: Builds optimization agent according to specified algorithms
# Author: Christian Hubbs
# christiandhubbs@gmail.com
# 04.02.2019

from .mip_algos import mip_utils
from ..scheduler.mip_scheduler import *

def create_agent(env, schedule=None):
    # Check for default MIP (Mixed Integer Programming) values in the environment settings
    env.settings = mip_utils.check_settings(env.settings)
    
    # Get algorithm-specific settings and hyperparameters
    if env.settings['MIP_ALGO'] == 'MPC':
        # Import and build a deterministic MIP agent for Model Predictive Control (MPC)
        from .mip_algos.deterministic_mip import buildDeterministicMIP
        agent = schedulingMPC(env, buildDeterministicMIP, schedule=schedule)
    elif 'GOD' in env.settings['MIP_ALGO']:
        # Import and build a "God" MIP agent, which has a fixed planning horizon
        from .mip_algos.god_mip import buildGodMIP
        # TODO: Fix planning horizon properly
        env.fixed_planning_horizon = env.n_days
        agent = schedulingMPC(env, buildGodMIP, schedule=schedule)
    elif env.settings['MIP_ALGO'] == 'SMPC':
        # Import and build a stochastic MIP agent for Stochastic Model Predictive Control (SMPC)
        from .mip_algos.stochastic_mip import buildStochasticMIP
        agent = schedulingMPC(env, buildStochasticMIP, schedule=schedule)
    else:
        # Raise an error if the specified MIP algorithm is not recognized
        raise ValueError('MIP_ALGO {} not recognized'.format(env.settings['MIP_ALGO']))

    return agent