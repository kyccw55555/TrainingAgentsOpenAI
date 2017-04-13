### Project 4 by Yunchuan Kong
### The trained model is contained in trained.ckpt (based on 7700 episodes)
### The code is ready to load trained.ckpt and perform testing, i.e.
### trained = True, resume = False
### only the second while loop will run

import gym
import numpy as np
from rl_agent import neuralAgent


resume=False
# resume=True
trained=False
trained=True

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

if __name__ == "__main__":
    env = gym.make('Pong-v0')
    agent = neuralAgent( n_obs=80*80,
                            h=250,
	                        n_actions=3,
                            learning_rate=0.001,
                            gamma=0.99,
                            decay=0.99,
                            # save_path='training.ckpt',
							save_path='trained.ckpt'
                        )
    if resume or trained:
        agent.load()
    episode_number=1
    running_reward = None
    reward_sum = 0
    observation = env.reset()
    prevX = None

    while True:
        if trained:
            print 'The agent has been trained.'
            break
        observation = prepro(observation)
        if prevX is not None:
            features = observation - prevX
        else:
            features = np.zeros(80*80)
        prevX=observation
        action, label = agent.act(features)
        observation, reward, done, info = env.step(action+1)
        agent.record(reward)
        reward_sum += reward
        
        if done:
            if running_reward is None:
                running_reward = reward_sum
            else:
                running_reward = (running_reward  * (episode_number-1) + reward_sum) / episode_number
            agent.update()
            if episode_number % 25 == 0:
                print 'episode {}: reward: {}, mean reward: {:3f}'.format(episode_number, reward_sum, running_reward)
            else:
                print '\tepisode {}: reward: {}'.format(episode_number, reward_sum)
            if (episode_number) % 100 == 0:
                agent.save()
            reward_sum = 0
            observation = env.reset()
            episode_number += 1

    while True:
        if not trained:
            break
        elif resume:
            print 'Warning: training not finished.'
            break
        if episode_number>20:
            break
        observation = prepro(observation)
        if prevX is not None:
            features = observation - prevX
        else:
            features = np.zeros(80 * 80)
        prevX = observation
        action, label = agent.act(features)
        observation, reward, done, info = env.step(action + 1)
        reward_sum += reward
        if done:
            if running_reward is None:
                running_reward = reward_sum
            else:
                running_reward = (running_reward  * (episode_number-1) + reward_sum) / episode_number
            print 'game {}: reward: {}, mean reward: {:3f}'.format(episode_number, reward_sum, running_reward)
            reward_sum = 0
            observation = env.reset()
            episode_number += 1
