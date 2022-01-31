from environment import *
import numpy as np
import gym
import tensorflow as tf

if __name__ == '__main__':
    env = CubeEnv()
    learning_rate = 0.0005
    n_scrambles = 500
    agent = Agent(learning_rate, gamma = 1.0, action_space_size = env.action_space.n,
                    epsilon = 1.0, input_shape = env.observation_space.shape[0], batch_size = 64)
    scores = []
    action_counter = 0

    for i in range(n_scrambles):
        done = False
        score = 0
        # TODO: if the average score is -1 env.reset(2) etc
        state = env.reset(1)
        while not done and score > -2:  # change -2 based on number of scrambles
            action = agent.choose_action(state)
            new_state, reward, done, info = env.step(action)
            action_counter += 1
            score += reward
            agent.store_transition(state, action, new_state, done)
            state = new_state
            if action_counter % 4 == 0:
                agent.learn()

        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print('episode: ', i, 'score %.2f' %score,
                'average_score %.2f' % avg_score)

