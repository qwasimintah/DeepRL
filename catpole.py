import gym
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random
from keras import backend as K
from keras.models import Model
from keras.layers import Lambda, merge, Input
 
class DQNAgent():
   
    def build_model(self, state_size, action_size, alpha, alpha_decay):
        print(state_size, action_size)
        model = Sequential()
        model.add(Dense(40, input_dim=state_size, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=alpha, decay=alpha_decay))
        return model

    def build_dueling_model(self, state_size, action_size, alpha, alpha_decay):
        input = Input(shape=(state_size,))
        x = Dense(32, input_shape=(state_size,), activation='relu', kernel_initializer='he_uniform')(input)
        x = Dense(16, activation='relu', kernel_initializer='he_uniform')(x)

        x = Dense(action_size + 1, activation='linear')(x)
        x = Lambda(lambda i: K.expand_dims(i[:,0],-1) + i[:,1:] - K.mean(i[:,1:], keepdims=True), output_shape=(action_size,))(x)

        model = Model(input=input, output=x)
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=alpha, decay=alpha_decay))
        return model


    def build_distribution_model(self,state_size, num_atoms, action_size, learning_rate):
        state_input = Input(shape=(state_size,))
        x = Dense(32, input_shape=(state_size,), activation='relu')(state_input)
        x = Dense(16, activation='relu')(x)
        distribution_list = []
        for i in range(action_size):
            distribution_list.append(Dense(num_atoms, activation='softmax')(x))

        model = Model(input=state_input, output=distribution_list)

        adam = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy',optimizer=adam)

        return model