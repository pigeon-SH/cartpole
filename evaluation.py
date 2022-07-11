import numpy as np
import torch
import gym
from network import AgentNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess(x):
    return torch.Tensor(x).to(dtype=torch.float32, device=device)

def score_to_action(score):
    return torch.argmax(score, dim=-1).item()
    
env = gym.make("CartPole-v1")
agent_net = AgentNet(4, 2)
agent_net = torch.load('./model.pt').to(device)
agent_net.eval()

scenario_n = 10

for episode in range(scenario_n):

    observation = env.reset()
    env.render()
    reward_per_episode = 0 
    done = 0

    while not done:
        action_scores = agent_net(preprocess(observation)).to(device)
        action = score_to_action(action_scores)
        observation, reward, done, info = env.step(action)
        env.render()
        reward_per_episode += reward

    print('model episode %d: reward %f' % (episode, reward_per_episode))

agent_net = torch.load('./best_model.pt').to(device)
agent_net.eval()

for episode in range(scenario_n):

    observation = env.reset()
    env.render()
    reward_per_episode = 0 
    done = 0

    while not done:
        action_scores = agent_net(preprocess(observation)).to(device)
        action = score_to_action(action_scores)
        observation, reward, done, info = env.step(action)
        env.render()
        reward_per_episode += reward

    print('best_model episode %d: reward %f' % (episode, reward_per_episode))

env.close()
