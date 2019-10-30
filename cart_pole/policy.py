import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tqdm import tqdm


class BasePolicy(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def next(self, state):
        return self.action_space.sample()


class DQN(BasePolicy):
    def __init__(self, action_space, model_path=None):
        super().__init__(action_space)

        if model_path is not None:
            self.model = load_model(model_path)
            print('DQN model is loaded from {}'.format(model_path))

    def next(self, state):
        q_value = self.model.predict(np.array(state).reshape(1, -1))
        return np.argmax(q_value[0])

    def train(self, env):
        num_state = env.observation_space.shape[0]
        num_action = env.action_space.next

        self.model = Sequential()
        self.model.add(Dense(32, input_dim=num_state, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(num_action, activation='linear'))
        self.model.compile(loss='mse', optimizer='adam')

        num_iteration = 500
        min_timesteps_per_batch = 2500
        epsilon = .3
        gamma = .95

        for i in tqdm(range(num_iteration)):
            timesteps_this_batch = 0
            memory = []

            while True:
                state = env.reset()
                done = False

                while not done:
                    if np.random.uniform() < epsilon:
                        action = env.action_space.sample()
                    else:
                        q_value = self.model.predict(state.reshape(1, -1))
                        action = np.argmax(q_value[0])

                    next_state, reward, done, info = env.step(action)

                    memory.append((state, action, reward, next_state, done))
                    state = next_state

                timesteps_this_batch += len(memory)
                if timesteps_this_batch > min_timesteps_per_batch:
                    break

            # Replay
            for state, action, reward, next_state, done in memory:
                if done:
                    target = reward
                else:
                    target = reward + gamma * (np.max(self.model.predict(next_state.reshape(1, -1))[0]))

                q_value = self.model.predict(state.reshape(1, -1))
                q_value[0][action] = target
                self.model.fit(state.reshape(1, num_state), q_value, epochs=1, verbose=0)

        env.close()

    def save(self, model_path):
        self.model.save(model_path)
