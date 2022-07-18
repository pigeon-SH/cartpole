import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, target_model, sample_batch):
    discount = 0.99
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_function = nn.MSELoss()

    samples = np.array(sample_batch)
    states = torch.Tensor(np.array([state for state in samples[:, 0]])).to(device)
    actions = torch.Tensor(np.array([action for action in samples[:, 1]])).to(device=device, dtype=torch.int64)
    rewards = torch.Tensor(np.array([action for action in samples[:, 2]])).to(device)
    dones = torch.Tensor(np.array([action for action in samples[:, 3]])).to(device)
    next_states = torch.Tensor(np.array([action for action in samples[:, 4]])).to(device)


    predict = model(states)
    # only q(s,a) update, qval for other actions should be same as predict
    one_hot_actions = F.one_hot(actions, 2)
    predict = torch.sum(predict * one_hot_actions, dim=1)

    with torch.no_grad():
        q_val = target_model(next_states)
    
    # bellman eqn
    target_val = rewards + (1 - dones) * discount * q_val.amax(axis=1)
    
    loss = loss_function(predict, target_val)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()