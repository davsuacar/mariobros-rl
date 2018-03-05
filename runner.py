from gym.envs.registration import register
from PIL import Image
import numpy as np

register(
    id='SuperMarioBros-1-1-v0',
    entry_point='gym.envs.ppaquette_gym_super_mario:MetaSuperMarioBrosEnv'
)

import gym

N_FRAMES = 3
WIDTH_SIZE = 80
HEIGHT_SIZE = 80


class GymRunner:
    def __init__(self, env_id, max_timesteps=100000):

        self.env = gym.make('SuperMarioBros-1-1-v0')
        self.max_timesteps = max_timesteps

    def calc_reward(self, state, action, gym_reward, next_state, done):
        return gym_reward

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

                # execute the selected action
                next_state, reward, done, _ = self.env.step(action)
                print("Action...", action)
                print("Reward: ", reward)
                img = Image.fromarray(next_state).convert('L').resize((WIDTH_SIZE, HEIGHT_SIZE))
                img = np.array(img.getdata(), dtype=np.uint8).reshape(WIDTH_SIZE, HEIGHT_SIZE)

                # TODO: Change next_states to three frames
                next_state = img

                reward = self.calc_reward(state, action, reward, next_state, done)

                # record the results of the step
                if do_train:
                    agent.record(state, action, reward, next_state, done)

                total_reward += reward

                state.append(next_state)
                state.pop(0)

                if done:
                    break

            # train the agent based on a sample of past experiences
            if do_train:
                agent.replay()

            print("episode: {}/{} | score: {} | e: {:.3f}".format(
                episode + 1, num_episodes, total_reward, agent.epsilon))





