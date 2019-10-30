import gym
from time import sleep
from const import ENV_NAME


with gym.make(ENV_NAME) as env:
    action_space = env.action_space

def run(pi, n_episode=1):
    with gym.make(ENV_NAME) as env:
        for i_episode in range(n_episode):
            state = env.reset()
            done = False
            t = 0

            while not done:
                env.render()

                action = pi.next(state)
                print(state, action)

                state, reward, done, info = env.step(action)
                if done:
                    print('Episode finished after {} steps'.format(t + 1))
                    break

                t += 1

            sleep(1)
