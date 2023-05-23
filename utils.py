
# Replay memory for dqn training
import random
from collections import namedtuple, deque

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

# Highway environment
import gymnasium as gym
import numpy as np

CONFIG = {
       "observation": {
           "type": "GrayscaleObservation",
           "observation_shape": (128, 64),
           "stack_size": 4,
           "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
           "scaling": 1.75,
       },
       "policy_frequency": 2
   }

def get_obs_shape():
    obs_config = CONFIG['observation']
    return (obs_config['stack_size'], obs_config['observation_shape'][0], obs_config['observation_shape'][1])

def create_highway_env(env_name):
    env = gym.make(env_name, render_mode="rgb_array")
    env.configure(CONFIG)
    env = HighwayObsWrapper(env)
    return env

_COLOR = np.array([210,164,74]).mean()
class HighwayObsWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(HighwayObsWrapper, self).__init__(env)

    def observation(self, observation):
        observation = observation.astype(np.float32)
        # Normalize to -1 to 1
        observation = (observation - 128) / 128
        return observation
    
# Action selector with epsilon greedy method
import math
import torch

class ActionSelector:
    def __init__(self,
                 device,
                 policy_net,
                 env,
                 eps_start,
                 eps_end,
                 eps_decay
                 ):
        self.net = policy_net
        self.device = device
        self.env = env
        self.step = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        
    def __call__(self, state):
        action = 0
        rn = random.random()
        threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.step / self.eps_decay)
        if rn > threshold:
            with torch.no_grad():
                action = self.net(state).max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
        self.step += 1
        
        return action
    
# Visualizer
import matplotlib
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self):
        self.buffer = []
        plt.ion()
        self.is_ipython = 'inline' in matplotlib.get_backend()
            
    def step(self, val):
        self.buffer.append(val)
    
        plt.figure(1)
        durations_t = torch.tensor(self.buffer, dtype=torch.float)
        plt.clf()
        plt.title(' ')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(durations_t.numpy())
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.pause(0.001)
            
    def close(self, dir = None):
        if dir != None:
            plt.savefig(f"{dir}/plt.png")
        plt.ioff()
        plt.show()
