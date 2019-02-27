import gym 
import random 
import torch 
import numpy as np
from utils.decay_schedule import LinearDecaySchedule
from catpole import DQNAgent
from collections import deque
from keras.callbacks import TensorBoard
from time import time




env = gym.make("CartPole-v0")
MAX_NUM_EPISODES = 100000
MAX_STEPS_PER_EPISODE = 300
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))


class Shallow_Q_Learner(object):
    def __init__(self, state_shape, action_shape, learning_rate=0.005,gamma=0.98):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.gamma = gamma # Agent's discount factor
        self.learning_rate = learning_rate # Agent's Q-learning rate
        # self.Q is the Action-Value function. This agent represents Q using a
        # Neural Network.
        print(self.state_shape, self.action_shape)
        self.Q = DQNAgent().build_model(self.state_shape[0], self.action_shape, 0.01,0.01 )
        self.tQ = DQNAgent().build_model(self.state_shape[0], self.action_shape, 0.01,0.01 )
        # self.policy is the policy followed by the agent. This agents follows
        # an epsilon-greedy policy w.r.t it's Q estimate.
        self.policy = self.epsilon_greedy_Q
        self.epsilon_max = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = LinearDecaySchedule(initial_value=self.epsilon_max,final_value=self.epsilon_min,max_steps= 0.5 * MAX_NUM_EPISODES *MAX_STEPS_PER_EPISODE)
        self.step_num = 0
        #self.memory =deque(maxlen=2000)

    def get_action(self, observation):
        return self.policy(observation)

    def epsilon_greedy_Q(self, observation):
        # Decay Epsilion/exploratin as per schedule
        if random.random() < self.epsilon_decay(self.step_num):
            return random.choice([i for i in range(self.action_shape)])
        return np.argmax(self.model.predict(state))


    def remember(self, state, action, reward, next_state, done):      
        self.memory.append((state, action, reward, next_state, done))
 
    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.Q.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.tQ.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])
            
        self.Q.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0, callbacks=[tensorboard])
    
    def target_train(self):
        weights = self.Q.get_weights()
        target_weights = self.tQ.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.learning_rate + target_weights[i] * (1 - self.learning_rate)
        self.tQ.set_weights(target_weights)

if __name__ == "__main__":
    sample_batch_size = 32
    observation_shape = env.observation_space.shape
    action_shape = env.action_space.n
    agent = Shallow_Q_Learner(observation_shape, action_shape)
    first_episode = True
    episode_rewards = list()
    steps= 0
    for episode in range(MAX_NUM_EPISODES):
        obs = env.reset()
        obs = np.reshape(obs, [1, observation_shape[0]])
        cum_reward = 0.0 # Cumulative reward
        for step in range(MAX_STEPS_PER_EPISODE):
            env.render()
            action = agent.get_action(obs)

            next_obs, reward, done, info = env.step(action)
            next_obs = np.reshape(next_obs, [1, observation_shape[0]])

            agent.remember(obs, action,reward,next_obs, done)
            agent.replay(sample_batch_size)
            steps += 1
                
            if steps % agent.update_steps == 0:
                agent.target_train()
            
            #agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            cum_reward += reward
            if done is True:
                if first_episode: # Initialize max_reward at the end of first episode
                    max_reward = cum_reward
                    first_episode = False
                episode_rewards.append(cum_reward)
                if cum_reward > max_reward:
                    max_reward = cum_reward
                    print("\nEpisode#{} ended in {} steps. reward ={} ; mean_reward={} best_reward={}".
                format(episode, step+1, cum_reward, np.mean(episode_rewards),
                max_reward))
                break
    env.close()