from collections import deque
import numpy as np
import random
import abc

class QLearningAgent:
    def __init__(self, action_size):
        self.action_size = action_size

        # hyperparameters
        self.gamma = 0.95  # discount rate on future rewards
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.995  # the decay of epsilon after each training batch
        self.epsilon_min = 0.1  # the minimum exploration rate permissible
        self.batch_size = 32  # maximum size of the batches sampled from memory

        # agent state
        self.model = self.build_model()
        self.memory = deque(maxlen=2000)

    @abc.abstractmethod
    def build_model(self):
        return None

    def select_action(self, state, do_train=True):
        if do_train and np.random.rand() <= self.epsilon:
            return np.eye(self.action_size, dtype=int)[np.random.choice(self.action_size)]
        return np.eye(self.action_size, dtype=int)[
                np.argmax(self.model.predict(np.expand_dims(np.array(state), 0))[0])]

    def record(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.expand_dims(np.array(state), 0)
            next_state = np.expand_dims(np.array(next_state), 0)
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(np.eye(self.action_size, dtype=int)[np.argmax(self.model.predict(next_state)[0])]))
            target_f = self.model.predict(state)
            target_f[0][np.argmax(action)] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay





