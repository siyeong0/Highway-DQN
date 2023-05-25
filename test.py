import torch
import time
import argparse
import highway_env

from network import DQN
from utils import create_highway_env, get_obs_shape

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['PYDEVD_DISABLE_FILE_VALIDATION']='1'
    
parser = argparse.ArgumentParser(description='TEST')
parser.add_argument('--env-name', default='highway-fast-v0',
                    help='environment to train on (default: highway-fast-v0)')
parser.add_argument('--model-path', default='./checkpoints/highway-fast-v0/highway-fast-v0.pth',
                    help='model path to test (default: ./checkpoints/highway-fast-v0/highway-fast-v0.pth)')
parser.add_argument('--fps', default=30,
                    help='display fps (default: 30)')
if __name__ == "__main__":
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device : {device}')
    env = create_highway_env(args.env_name)
    action_space = env.action_space
    observation_shape = get_obs_shape()
    net = DQN(observation_shape, action_space)
    net.load_state_dict(torch.load(args.model_path, map_location=device))

    net = net.to(device)
    net.eval()

    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    wait_msec = 1000 / args.fps
    for _ in range(300):
        start_time = int(time.time() * 1000)
        env.render()
        with torch.no_grad():
            action = net(state).max(1)[1].view(1, 1).item()
        state, reward, done, _, _ = env.step(action)

        if done:
            break

        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        interval_time = int(time.time() * 1000) - start_time
        time.sleep(max(wait_msec-interval_time, 1) / 1000)
        
    env.close()