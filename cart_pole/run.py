import argparse
import gym
import numpy as np
import os
from time import sleep

from cart_pole import action_space, run
from const import MODEL_PATH
from policy import BasePolicy, DQN


pi = BasePolicy(action_space)
#pi = DQN(action_space, model_path=MODEL_PATH)
run(pi=pi)
