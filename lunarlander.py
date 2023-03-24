import numpy as np
import gym
from ll_model import PolicyGradientAgent
import matplotlib.pyplot as plt
from ll_utils import plotLearning
from gym import wrappers

if __name__ == '__main__':
  agent = PolicyGradientAgent(lr=0.0005, input_dims=8, gamma=0.99, n_actions=4, l1_size=64, l2_size=64, chkpt_dir='tmp/lunar-lander-ckpt')
  #agent.load_checkpoint()
  env = gym.make('LunarLander-v2')
  score_history = []
  score = 0
  n_episodes = 2500

  env = wrappers.Monitor(env, 'tmp/lunar-lander', video_callable=lambda episode_id: True, force=True)

  for i in range(n_episodes):
    print('episode ', i, 'score ', score)
    done = False
    score = 0
    observation = env.reset()
    while not done:
      action = agent.choose_action(observation)
      observation_, reward, done, info = env.step(action)
      agent.store_transition(observation, action, reward)
      observation = observation_
      score += reward
    score_history.append(score)
    agent.learn()
    agent.save_checkpoint()
  filename = 'lunar-lander.png'
  plotLearning(score_history, filename=filename, window=25)