import numpy as np
import matplotlib.pyplot as plt
import PIL
import os
import logging
import matplotlib as mpl
import tensorflow as tf
import matplotlib.animation as animation
from tensorflow import keras
from tf_agents.environments import suite_gym
from tf_agents.environments.wrappers import ActionRepeat
from functools import partial
from gym.wrappers import TimeLimit
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.networks.q_network import QNetwork
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.eval.metric_utils import log_metrics
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.utils.common import function
from tf_agents.policies.policy_saver import PolicySaver

policy_path = 'eval_policy'
eval_policy = tf.compat.v2.saved_model.load(policy_path)
# not sure what to do with policy from this point on