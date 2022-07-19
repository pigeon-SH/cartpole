import gym
import torch
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch.nn.functional as F

from agent import AgentNet
import argparse
import utils

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def update_target_net(policy_net, target_net):
    target_net.load_state_dict(policy_net.state_dict())

def train(policy_net, target_net, optimizer, samples, discount):
    """ samples: (states, acitons, rewards, dones, next_states)
        ----------
        states: float32 tensors
        actions: int values
        rewards: values
        dones: bools
        next_states: float32 tensors
    """
    states = torch.stack([sample[0] for sample in samples]).to(device)
    actions = torch.Tensor(np.array([sample[1] for sample in samples])).to(device=device, dtype=torch.int64)
    rewards = torch.Tensor(np.array([sample[2] for sample in samples])).to(device)
    dones = torch.Tensor(np.array([sample[3] for sample in samples])).to(device)
    next_states = torch.stack([sample[4] for sample in samples]).to(device)

    predict = policy_net(states)
    one_hot_actions = F.one_hot(actions, 2)
    predict = torch.sum(predict * one_hot_actions, dim=1)

    with torch.no_grad():
        q_val = target_net(next_states)
    
    # bellman eqn
    target_val = rewards + (1 - dones) * discount * q_val.amax(axis=1)
    
    loss = F.mse_loss(predict, target_val)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def learn(env_name, EPISODES, obs_select, frame_num, learning_rate, load_best):
    epsilon = 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.01
    discount = 0.99

    buffer_len = 2000
    batch_size = 64
    train_start = 1000
    replay_buffer = deque(maxlen=buffer_len)

    record = []
    score_avg = 0.0
    best_score = 10
    total_steps = 0

    obs_index = [domain == 'T' for domain in obs_select]

    policy_net = AgentNet(sum(obs_index) * frame_num, 2)
    if load_best:
        policy_net = torch.load('learn/best_policy_net_' + obs_select + '.pt')
    target_net = AgentNet(sum(obs_index) * frame_num, 2)
    update_target_net(policy_net, target_net)
    
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)

    env = gym.make(env_name)

    for iter in range(EPISODES):
        # init state
        obs = env.reset()
        frame = utils.preprocess(obs, obs_index)
        frames = [frame for _ in range(frame_num)]
        state = utils.get_state(frames)

        episode_score = 0.0 # episode's reward cumulation
        steps = 0
        while True:
            # get action by e-greedy
            if random.random() <= epsilon:
                # explore
                action = random.randint(0, 1)
            else:
                # exploit
                action_score = policy_net(state)
                action = utils.score_to_action(action_score)
            
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            # action
            next_obs, reward, done, info = env.step(action)
            episode_score += reward

            # record
            if done and episode_score >= 500:
                reward = 1
            elif done:
                reward = -1
            else:
                reward = 0.1
            
            next_frame = utils.preprocess(next_obs, obs_index)
            frames.pop(0)
            frames.append(next_frame)
            next_state = utils.get_state(frames)

            experience = (state, action, reward, done, next_state)
            replay_buffer.append(experience)

            steps += 1

            # next
            state = next_state

            # train
            if total_steps > train_start:
                samples = random.sample(replay_buffer, batch_size)
                train(policy_net, target_net, optimizer, samples, discount)
            
            if done:
                break
        
        total_steps += steps
        print("iter: {:4d} / {:4d}, score_avg: {:7.4f}".format(iter, EPISODES, score_avg))

        if episode_score >= best_score:
            best_score = episode_score
            torch.save(policy_net, 'learn/best_policy_net_' + obs_select + '_' + str(frame_num) + '.pt')
        
        update_target_net(policy_net, target_net)

        score_avg = (score_avg * len(record) + episode_score) / (len(record) + 1)
        record.append(score_avg)

    torch.save(policy_net, 'learn/policy_net_' + obs_select + '_' + str(frame_num) + '.pt')
    env.close()

    plt.plot(record)
    plt.savefig('learn/' + env_name + '_graph_' + obs_select + '_' + str(frame_num) + '.png')
    #plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--obs', type=str, default="TTTT", help="You can restrict observation by giving F for that domain. Cart pole obs domain is (position, position_velocity, angle, angular_velocity) TFTF means only observe position and angle")
    parser.add_argument('--frame_num', type=int, default=4, help="how many frames you want to use for input of network")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--episode_num', type=int, default=3000)
    parser.add_argument('--load_best', type=bool, default=False)

    args = parser.parse_args()
    obs_select = args.obs
    frame_num = args.frame_num
    learning_rate = args.learning_rate
    episode_num = args.episode_num
    load_best = args.load_best
    
    learn("CartPole-v1", episode_num, obs_select, frame_num, learning_rate, load_best)
    
