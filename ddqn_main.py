from environment import *
import numpy as np
import gym
import tensorflow as tf

if __name__ == '__main__':
    env = CubeEnv()
    learning_rate = 0.0005
    n_scrambles = 500
    agent = Agent(learning_rate, gamma = 1.0, action_space_size = env.action_space.n,
                    epsilon = 1.0, input_shape = env.observation_space.shape, batch_size = 64)
    # Can put agent = load_model later on
    scores = []
    action_counter = 0
    episode_counter = 0
    avg_score = -2

    while avg_score < -1.1:
        episode_counter += 1
        done = False
        score = 0  # how many rotations
        # TODO: if the average score is -1 env.reset(2) etc
        state = env.reset(1)
        first = True
        while not done and score > -2:  # change -2 based on number of scrambles
            action = agent.choose_action(state)
            new_state, reward, done, info = env.step(action)
            score += reward
            if (first):
                agent.store_transition(state, new_state, action, done)
                action_counter += 1
            first = False
            state = new_state
            if action_counter % 4 == 0:
                agent.learn()

        scores.append(score)
        if episode_counter % 20 == 0:
            avg_score = np.mean(scores[-100:])
            print('episode: ', episode_counter, 'score %.2f' %score,
                'average_score %.2f' % avg_score)

