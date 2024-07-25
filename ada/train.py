#!/usr/bin/env python3
# Shebang line to specify the interpreter

import sys 
# Provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter.

import os
# Provides a way of using operating system dependent functionality like reading or writing to the file system.

from argparse import ArgumentParser, ArgumentTypeError
# ArgumentParser: Facilitates parsing command-line arguments.
# ArgumentTypeError: Exception raised when an argument is of the wrong type.

import numpy as np
# NumPy: A fundamental package for scientific computing with Python, providing support for arrays and matrices, along with a collection of mathematical functions to operate on these data structures.

from config import *
# Imports all definitions from the config module, which might include configurations or settings required for the script.

def main(argv):
    # Main function to parse command-line arguments and set up the agent
    args = parse_cl_args(argv)
    agent = set_up_sim(args)
    # TODO: Train agent and log results

    agent.train()

if __name__ == "__main__":
    # Entry point of the script
    main(sys.argv)