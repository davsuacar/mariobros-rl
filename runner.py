from gym.envs.registration import register
from PIL import Image
import numpy as np

register(
    id='SuperMarioBros-1-1-v0',
    entry_point='gym.envs.ppaquette_gym_super_mario:MetaSuperMarioBrosEnv'
)

import gym

N_FRAMES = 3
WIDTH_SIZE = 224
HEIGHT_SIZE = 256


class GymRunner:
    def __init__(self, env_id, max_timesteps=100000):

        self.env = gym.make('SuperMarioBros-1-1-v0')
        self.max_timesteps = max_timesteps

    def train(self, agent, num_episodes):
        return self.run(agent, num_episodes, do_train=True)

    def run(self, agent, num_episodes, do_train=False):
        for episode in range(num_episodes):
            self.env.render()
            self.env.reset()

            # Reset does not return image until first action is set.
            frame = np.zeros((224, 256, 3), dtype=np.uint8)

            img = Image.fromarray(frame).convert('L').resize((WIDTH_SIZE, HEIGHT_SIZE))

            frame = np.array(img.getdata(), dtype=np.uint8).reshape(WIDTH_SIZE, HEIGHT_SIZE)

            frames = [frame] * N_FRAMES

            state = frames

            total_reward = 0

            for t in range(self.max_timesteps):

                action = agent.select_action(state, do_train)

                action_translated = self.translate_action(action)

                # execute the selected action
                next_state, reward, done, _ = self.env.step(action_translated)
                #print("Action...", action)
                #print("Reward: ", reward)
                img = Image.fromarray(next_state).convert('L').resize((WIDTH_SIZE, HEIGHT_SIZE))
                img = np.array(img.getdata(), dtype=np.uint8).reshape(WIDTH_SIZE, HEIGHT_SIZE)

                next_state = state
                next_state.append(img)
                next_state.pop(0)

                # record the results of the step
                if do_train:
                    agent.record(state, action, reward, next_state, done)

                total_reward += reward

                state.append(img)
                state.pop(0)

                if done:
                    break

            # train the agent based on a sample of past experiences
            if do_train:
                agent.replay()

            print("episode: {}/{} | score: {} | e: {:.3f}".format(
                episode + 1, num_episodes, total_reward, agent.epsilon))

    def translate_action(self, network_output):
        environment_input = np.zeros(6)
        extra_actions = network_output[7:14]
        if np.sum(extra_actions) == 1:
            action_key = np.argmax(extra_actions)

            # [Right + B]
            if action_key == 0:
                environment_input = [0, 0, 0, 1, 0, 1]
            # [Right + A]
            elif action_key == 1:
                environment_input = [0, 0, 0, 1, 1, 0]
            # [Left + B]
            elif action_key == 2:
                environment_input = [0, 1, 0, 0, 0, 1]
            # [Left + A]
            elif action_key == 3:
                environment_input = [0, 1, 0, 0, 1, 0]
            # [Up + A]
            elif action_key == 4:
                environment_input = [1, 0, 0, 0, 1, 0]
            # [Right + A + B]
            elif action_key == 5:
                environment_input = [0, 0, 0, 1, 1, 1]
            # [Left + A + B]
            elif action_key == 6:
                environment_input = [0, 1, 0, 0, 1, 1]

        else:

            environment_input[np.argmax(network_output[:6])] = 1

        return environment_input



