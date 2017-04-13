import gym
import numpy as np
from neurolab_agent import NeuroLabLearningAgent

EPISODES = 100000
resume=False
# resume=True
# Trained=True
Trained=False

# def prepro(I):
#     """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
#     I = I[35:195] # crop
#     I = I[::2,::2,0] # downsample by factor of 2
#     I[I == 144] = 0 # erase background (background type 1)
#     I[I == 109] = 0 # erase background (background type 2)
#     I[I != 0] = 1 # everything else (paddles, ball) just set to 1
#     return I.astype(np.float).ravel()

def prepro(observation,prev):
    features = []
    # features.extend(observation)
    diff = observation-prev
    features.extend(diff)
    return features

if __name__ == "__main__":
    env = gym.make('BipedalWalker-v2')
    outdir="/Dropbox/final"
    agent = NeuroLabLearningAgent(24,
                                     env.action_space,
                                     learning_rate=0.3,
                                     discount=0.99,
                                     exploration_rate=0.5,
                                     exploration_decay_rate=0.99,
                                    batch_size=40,
                                    hidden1 = 16,
                                    hidden2 = 8,
                                    maxInput=1,
                                    minInput=-1,
                                    nnType="newff"
                                  )
    if resume:
        agent.load("neurolab.model")
        print("model loaded.")
    total_rewards = []
    # total_lengths = []
    for i_episode in range(EPISODES):
        # env.monitor.start(outdir, force=True)
        observation = env.reset()
        agent.reset()
        total_reward = 0
        #length = 0
        moving_average = 0
        prevX=np.zeros(len(observation))
        for j in xrange(250):
            env.render()
            # discretize
            observation=agent.discretize(observation)
            features=prepro(observation, prevX)
            action = agent.act(features)
            new_observation, reward, done, info = env.step(action)
            new_features = prepro(new_observation, observation)
            prevX=observation
            observation = new_observation
            # print(observation)
            reward=reward
            total_reward += reward
            agent.update(features, action, new_features, reward, done)
            #length += 1
            if done:
                break
                
        total_rewards.append(total_reward)
        #total_lengths.append(length)
        moving_average = sum(total_rewards)/len(total_rewards)
        print "== episode: ",i_episode+1," reward: ",total_reward,"average: ",moving_average
        # env.monitor.close()
        if (i_episode+1) % 50 == 0:
            agent.save("neurolab.model")
            print("model saved.")


    # if Trained:
    #     agent.load("neurolab.model")
    #     print("model loaded.")
    #
    # for i in xrange(5):
    #     observation = env.reset()
    #     observation = prepro(observation)
    #     agent._exploration_rate = 0.0
    #     total_reward = 0
    #     prevX = np.zeros(len(observation))
    #     for j in range(10000):
    #         #env.render()
    #         features = prepro2(observation, prevX)
    #         action = agent.act(features)
    #         prevX = observation
    #         new_observation, reward, done, info = env.step(action)
    #         new_observation = prepro(new_observation)
    #         observation = new_observation
    #
    #         total_reward += reward
    #         if done:
    #             break
    #     print(total_reward)