import gym 
import random 
import torch 
import numpy as np
from utils.decay_schedule import LinearDecaySchedule
from catpole import DQNAgent
from collections import deque
from keras.callbacks import TensorBoard
from time import time
from memory_is import Memory
from random_agent import RandomAgent



env = gym.make("CartPole-v0")
MAX_NUM_EPISODES = 100000
MAX_STEPS_PER_EPISODE = 300
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))


class Shallow_Q_Learner(object):
    def __init__(self, state_shape, action_shape, learning_rate=0.005,gamma=0.98, memory=Memory(capacity=2000)):
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
        self.update_steps = 64
        #self.memory =deque(maxlen=2000)
        self.memory = memory

    def get_action(self, observation):
        return self.policy(observation)

    def epsilon_greedy_Q(self, observation):
        # Decay Epsilion/exploratin as per schedule
        if random.random() < self.epsilon_decay(self.step_num):
            return random.choice([i for i in range(self.action_shape)])
        return np.argmax(self.model.predict(state))

    def compute_td_error(self, next_state, reward):
        Q_next = self.Q.predict(next_state)[0]
        current = reward + self.learning_rate * np.max(self.tQ.predict(next_state)[0])
        td_error = abs(current - Q_next)
        return td_error


    def remember(self, experience):
        self.memory.add(experience)

    def replay(self, batch_size):
        batch = self.memory.sample(batch_size)
        x_batch, y_batch, errors = [], [], []
        for i in range(len(batch)):
            minibatch = batch[i][1]
            state = minibatch[0]; action = minibatch[1]; reward = minibatch[2]; next_state = minibatch[3]; done = minibatch[4]

            y_target = self.Q.predict(state)
            target_init= y_target[0][action]
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.tQ.predict(next_state)[0])

            x_batch.append(state[0])
            y_batch.append(y_target[0])

            td_error = abs(target_init - y_target[0][action])
            errors.append(td_error*batch[i][2])

        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

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
    
    first_episode = True
    episode_rewards = list()
    steps = 0
    PRE_TRAINLENGTH=3000

    ## Random Agent
    ragent = RandomAgent(action_shape)
    obs = env.reset()
    for step in range(PRE_TRAINLENGTH):
            #env.render()
        action = ragent.act(obs)

        next_obs, reward, done, info = env.step(action)
        next_obs = np.reshape(next_obs, [1, observation_shape[0]])
        ragent.observe((obs,action, reward, next_obs, done))
        obs = next_obs

    agent = Shallow_Q_Learner(observation_shape, action_shape, memory=ragent.get_memory())

    for episode in range(MAX_NUM_EPISODES):
        obs = env.reset()
        obs = np.reshape(obs, [1, observation_shape[0]])
        cum_reward = 0.0 # Cumulative reward
        for step in range(MAX_STEPS_PER_EPISODE):
            #env.render()
            action = agent.get_action(obs)

            next_obs, reward, done, info = env.step(action)
            next_obs = np.reshape(next_obs, [1, observation_shape[0]])

            td_error = agent.compute_td_error(next_obs, reward)
            #agent.remember(obs, action,reward,next_obs, done)
            agent.remember((obs,action, reward, next_obs, done))
            agent.replay(sample_batch_size)
            steps += 1         
            if steps % agent.update_steps == 0:
                agent.target_train()
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