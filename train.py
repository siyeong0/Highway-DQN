from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import numpy as np

import highway_env
from network import DQN
from utils import Transition, ReplayMemory, create_highway_env, ActionSelector, Visualizer, get_obs_shape

import os
#os.environ["OMP_NUM_THREADS"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['PYDEVD_DISABLE_FILE_VALIDATION']='1'

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--num-ep', type=int, default=5000,
                    help='maximum length of an episode (default: 5000)')
parser.add_argument('--batch-size', default=128,
                    help='batch size (default: 128)')
parser.add_argument('--save-freq', default=1000,
                    help='model saving frequency (default: 1000)')
parser.add_argument('--eps-start', default=0.9,
                    help='epsilon start 0~1 (default: 0.9)')
parser.add_argument('--eps-end', default=0.05,
                    help='epsilon end 0~1 (default: 0.05)')
parser.add_argument('--eps-decay', default=1000,
                    help='epsilon decay (default: 1000)')
parser.add_argument('--target-update-coef', default=0.005,
                    help='target network update coeficient (default: 0.005)')
parser.add_argument('--memory-size', default=10000,
                    help='size of the replay memory buffer (default: 10000)')
parser.add_argument('--env-name', default='highway-fast-v0',
                    help='environment to train on (default: highway-fast-v0)')

def train(args):
    # Make checkpoints directory
    DIR = f"./checkpoints/{args.env_name}"
    try:
        if not os.path.exists(DIR):
            os.makedirs(DIR)
    except OSError:
        print("Error: Failed to create the directory.")
    # Setup train environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = create_highway_env(args.env_name)
    action_space = env.action_space
    observation_shape = get_obs_shape()
    
    policy_net = DQN(observation_shape, action_space).to(device)
    target_net = DQN(observation_shape, action_space).to(device)
    
    optimizer = optim.AdamW(policy_net.parameters(), lr=args.lr, amsgrad=True)
    memory = ReplayMemory(args.memory_size)
    action_selector = ActionSelector(device, policy_net, env, 
                                     args.eps_start, args.eps_end, args.eps_decay)
    visualizer = Visualizer()
    
    # Define optimizing function
    def optimize_model():
        if len(memory) < args.batch_size:
            return
        transitions = memory.sample(args.batch_size)
        batch = Transition(*zip(*transitions))
        # Separate non final states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        # Calc loss
        q_val = policy_net(states).gather(1, actions)
        v_val = torch.zeros(args.batch_size, device=device)
        with torch.no_grad():
            v_val[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        expectation = (v_val * args.gamma) + rewards
        
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_val, expectation.unsqueeze(1))
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()
    
    for ep in range(args.num_ep):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            # Get next replay
            action = action_selector(state) # Epsilon greedy
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward)

            state = next_state
            # Optimize if replay memory is enough 
            optimize_model()
            # Update target network
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            
            coef = args.target_update_coef
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*coef + target_net_state_dict[key]*(1-coef)
            target_net.load_state_dict(target_net_state_dict)
            # Update visualizer
            if done:
                visualizer.step(t + 1)
                break
        # Save model
        if (ep + 1) % args.save_freq == 0:
            torch.save(target_net.state_dict(), f'{DIR}/net_{ep}.pth')
            print(f'# EP{ep+1} SAVED...')
            
    print("##### END #####")
    visualizer.close()
    
    torch.save(target_net.state_dict(), f'{DIR}/result.pth')
    return target_net

if __name__ == "__main__":
    args = parser.parse_args()
    net = train(args)