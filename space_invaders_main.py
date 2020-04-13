import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(30)

import numpy as np
import random
import math
import glob
import io
import os
import cv2
import base64
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
from collections import deque
from datetime import datetime
import keras

from itertools import islice
from IPython.display import HTML
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf


class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.learning_rate = wandb.config.learning_rate
        self.epsilon = 1.0
        self.epsilon_decay = 0.9
        self.epsilon_min = 0.05
        self.gamma = 0.95
        self.epochs = 1000
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.randCount = 0
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(88, 80, 1)))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='softmax'))

        model.compile(loss='mse', optimizer=tf.keras.optimizers.Nadam(lr=self.learning_rate))
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def getSamples(self, minibatch):
          states = []
          actions = []
          rewards = []
          for state, action, reward, next_state, done in minibatch:
              states.append(state)
              actions.append(action)
              rewards.append(reward)
          states = np.asarray(states)
          actions = np.asarray(actions)
          rewards = np.asarray(rewards)
          discounted_rewards = self.discount_reward(rewards)
          return states, actions, discounted_rewards

    def train(self, batch_size):
#        minibatch = random.sample(self.memory, batch_size)
        total_samples = len(self.memory)
        batchPerEpoch = int (total_samples/batch_size)
        start = end = 0
        for batch in range(batchPerEpoch):
          start = batch * batch_size
          end = start + batch_size

          minibatch = deque(islice(self.memory, start, end))
          states, actions, discounted_rewards = self.getSamples(minibatch)
#          self.step(states, actions, discounted_rewards)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state, i):
        curr_state = np.expand_dims(state, axis=-1)
        curr_state = np.expand_dims(curr_state, axis=0)
        logits = self.model.predict(curr_state)
        prob_weights = tf.nn.softmax(logits).numpy()
        if True in np.isnan(prob_weights):
            self.randCount += 1
            print (self.randCount, prob_weights)
            return random.randrange(self.action_size)
        action = np.random.choice(self.action_size, size=1, p=prob_weights.flatten())[0]
        return action

    def save(self, name):
        self.model.save(name)

    def load(self, name):
        self.model.load_weights(name)

