import gym
import torch
from network import AgentNet
from train import train
from collections import deque
import random
import copy
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess(x):
    return torch.Tensor(x).to(dtype=torch.float32, device=device)

def score_to_action(score):
    return torch.argmax(score, dim=-1).item()

env = gym.make("CartPole-v1")
EPISODES = 3000
epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.01
demonstrations = deque(maxlen=2000)

record = []
score_avg = 0.0
best_score = 300
loss = 0.
total_steps = 0

model = AgentNet(observation_space=4, action_space=2)
model = torch.load('./best_model.pt')
target_model = copy.deepcopy(model)

for iter in range(EPISODES):
    # init state
    state = env.reset()
    score = 0.0
    steps = 0
    while True:
        # get action
        if random.random() <= epsilon:
            # explore
            action = random.randint(0, 1)
        else:
            # exploit
            action_score = model(preprocess(state))
            action = score_to_action(action_score)
        
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # action
        next_state, reward, done, info = env.step(action)
        score += reward
        if done and score >= 500:
            reward = 1
        elif done:
            reward = -1
        else:
            reward = 0.1

        # record
        experience = (state, action, reward, done, next_state)
        demonstrations.append(experience)
        steps += 1

        # next
        state = next_state

        # train
        if total_steps > 1000:
            loss = train(model, target_model, random.sample(demonstrations, 64))
        
        if done:
            break
    
    total_steps += steps
    print("iter: {:4d} / {:4d}, score_avg: {:7.4f}".format(iter, EPISODES, score_avg))

    if score >= best_score:
        best_score = score
        torch.save(model, './best_model.pt')
    
    target_model = copy.deepcopy(model)

    score_avg = (score_avg * len(record) + score) / (len(record) + 1)
    record.append(score_avg)

torch.save(model, './model.pt')
env.close()

plt.plot(record)
plt.savefig('./cartpole_graph.png')
plt.show()