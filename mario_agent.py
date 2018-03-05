from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import *

from runner import GymRunner
from qlearning_agent import QLearningAgent


class Agent(QLearningAgent):
    def __init__(self):
        super().__init__(6)

    def build_model(self):

        n_actions = 6
        n_frames = 3
        width_size = 80
        height_size = 80

        model = Sequential()
        model.add(
            Conv2D(16, (3, 3), activation='relu', input_shape=(n_frames, width_size, height_size), data_format='channels_first'))
        model.add(Dropout(0.4))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(n_actions))
        model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        # load the weights of the model if reusing previous training session
        # model.load_weights("models/mario-v0.h5")

        return model


if __name__ == '__main__':
    gym = GymRunner('Mario-v0')
    agent = Agent()

    gym.train(agent, 100000)
    gym.run(agent, 5000)
