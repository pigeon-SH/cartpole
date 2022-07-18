import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(obs, obs_index):
    """obs: numpy vector
        (position, velocity, angle, angular_velocity)

        convert numpy vector to tensor
    """
    return torch.Tensor(obs[obs_index]).to(device=device)

def get_state(frames):
    """convert frames(list of frames) to tensor
    """
    return torch.cat(frames).to(device=device, dtype=torch.float32)

def score_to_action(action_score):
    """convert action_score to action_id
    """
    return torch.argmax(action_score, dim=-1).item()