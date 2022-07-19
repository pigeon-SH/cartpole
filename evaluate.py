import numpy as np
import torch
import gym
import utils
import argparse
from agent import AgentNet
import gym.wrappers
from gym.wrappers.monitoring.video_recorder import VideoRecorder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(env_name, scenario_n, obs_select, frame_num, load_best, video):

    obs_index = [domain == 'T' for domain in obs_select]
    env = gym.make(env_name, render_mode='human')

    policy_net = AgentNet(sum(obs_index) * frame_num, 2)
    if load_best:
        policy_net = torch.load('learn/best_policy_net_' + obs_select + '_' + str(frame_num) + '.pt')
        print("-----Best Network Evaluation-----")
    else:
        policy_net = torch.load('learn/best_policy_net_' + obs_select + '_' + str(frame_num) + '.pt')
        print("-----Final Network Evaluation-----")
    policy_net.eval()

    result_f = open('evaluate/best_policy_net_' + obs_select + '_' + str(frame_num) + '.txt', 'w')
    rewards = []

    for episode in range(scenario_n):
        if video:
            video_path = 'evaluate/video_' + obs_select + '_' + str(frame_num) + '_' + str(episode) + '.mp4'
            video_recorder = VideoRecorder(env, video_path, enabled=video_path is not None)
        obs = env.reset()
        frame = utils.preprocess(obs, obs_index)
        frames = [frame for _ in range(frame_num)]
        state = utils.get_state(frames)
        env.render()
        reward_per_episode = 0 
        done = 0

        while not done:
            if video:
                video_recorder.capture_frame()
            action_score = policy_net(state)
            action = utils.score_to_action(action_score)
            obs, reward, done, info = env.step(action)
            env.render()
            reward_per_episode += reward

            frame = utils.preprocess(obs, obs_index)
            frames.pop(0)
            frames.append(frame)
            state = utils.get_state(frames)

        print('episode %d: reward %f' % (episode, reward_per_episode))
        result_f.write('episode %d: reward %f\n' % (episode, reward_per_episode))
        rewards.append(reward_per_episode)
        result_f.write('reward average: %f\n' % (float(sum(rewards)) / len(rewards)))
        if video:
            video_recorder.close()
            video_recorder.enabled = False

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--obs', type=str, default="TTTT", help="You can restrict observation by giving F for that domain. Cart pole obs domain is (position, position_velocity, angle, angular_velocity) TFTF means only observe position and angle")
    parser.add_argument('--frame_num', type=int, default=4, help="how many frames you want to use for input of network")
    parser.add_argument('--episode_num', type=int, default=10)
    parser.add_argument('--load_best', type=bool, default=None)
    parser.add_argument('--video', type=bool, default=False)

    args = parser.parse_args()
    
    obs_select = args.obs
    frame_num = args.frame_num
    episode_num = args.episode_num
    load_best = args.load_best
    video = args.video

    if load_best == None:
        evaluate("CartPole-v1", episode_num, obs_select, frame_num, True, video)
        evaluate("CartPole-v1", episode_num, obs_select, frame_num, False, video)
    else:
        evaluate("CartPole-v1", episode_num, obs_select, frame_num, load_best, video)
    