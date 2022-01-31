import gym
import numpy as np
import random
from tkinter import *
from Cube import *
from RenderEnv import *
from tensorflow import keras
import tensorflow as tf


class CubeEnv(gym.Env):  # child of gym.Env
    def __init__(self):
        self.dim = 2
        self.cube = Cube(self.dim)
        self.action_space = gym.spaces.Discrete(12)
        self.action_dict = {0: lambda: self.cube.clockwise_right(),  
        # doesn't work without lambda (but would work if the dictionary is written in step def) 
        # so i think something to do with return value
                        1: lambda: self.cube.clockwise_left(),
                        2: lambda: self.cube.clockwise_front(),
                        3: lambda: self.cube.clockwise_back(),
                        4: lambda: self.cube.clockwise_up(),
                        5: lambda: self.cube.clockwise_down(),
                        6: lambda: self.cube.anticlockwise_right(),
                        7: lambda: self.cube.anticlockwise_left(),
                        8: lambda: self.cube.anticlockwise_front(),
                        9: lambda: self.cube.anticlockwise_back(),
                        10: lambda: self.cube.anticlockwise_up(),
                        11: lambda: self.cube.anticlockwise_down()
                        }
        # each rotation is a number from 0 to 11
        self.observation_space = gym.spaces.Box(low = 0, high = 5, 
                                                shape = (6, self.dim, self.dim), dtype = np.int32)
        # Cube represented as a 6x3x3 tensor (or 6x2x2 for 2x2 cube)
        self.state = self.get_state()
        self.solved_states = []
        # for 2x2 there are 24 solved states (6 colours on top and 4 rotations of this)
        self.solved_states.append(self.state)
        if self.dim == 2:
            self.solved_states.append(np.array([self.cube.up_face, self.cube.down_face, self.cube.right_face, self.cube.left_face, self.cube.back_face, self.cube.front_face], dtype = np.uint8))
            self.solved_states.append(np.array([self.cube.up_face, self.cube.down_face, self.cube.left_face, self.cube.right_face, self.cube.front_face, self.cube.back_face], dtype = np.uint8))
            self.solved_states.append(np.array([self.cube.up_face, self.cube.down_face, self.cube.back_face, self.cube.front_face, self.cube.left_face, self.cube.right_face], dtype = np.uint8))
        
            self.solved_states.append(np.array([self.cube.down_face, self.cube.up_face, self.cube.front_face, self.cube.back_face, self.cube.left_face, self.cube.right_face], dtype = np.uint8))
            self.solved_states.append(np.array([self.cube.down_face, self.cube.up_face, self.cube.back_face, self.cube.front_face, self.cube.right_face, self.cube.left_face], dtype = np.uint8))
            self.solved_states.append(np.array([self.cube.down_face, self.cube.up_face, self.cube.left_face, self.cube.right_face, self.cube.back_face, self.cube.front_face], dtype = np.uint8))
            self.solved_states.append(np.array([self.cube.down_face, self.cube.up_face, self.cube.right_face, self.cube.left_face, self.cube.front_face, self.cube.back_face], dtype = np.uint8))

            self.solved_states.append(np.array([self.cube.front_face, self.cube.back_face, self.cube.up_face, self.cube.down_face, self.cube.left_face, self.cube.right_face], dtype = np.uint8))
            self.solved_states.append(np.array([self.cube.front_face, self.cube.back_face, self.cube.down_face, self.cube.up_face, self.cube.right_face, self.cube.left_face], dtype = np.uint8))
            self.solved_states.append(np.array([self.cube.front_face, self.cube.back_face, self.cube.right_face, self.cube.left_face, self.cube.up_face, self.cube.down_face], dtype = np.uint8))
            self.solved_states.append(np.array([self.cube.front_face, self.cube.back_face, self.cube.left_face, self.cube.right_face, self.cube.down_face, self.cube.up_face], dtype = np.uint8))

            self.solved_states.append(np.array([self.cube.back_face, self.cube.front_face, self.cube.up_face, self.cube.down_face, self.cube.right_face, self.cube.left_face], dtype = np.uint8))
            self.solved_states.append(np.array([self.cube.back_face, self.cube.front_face, self.cube.down_face, self.cube.up_face, self.cube.left_face, self.cube.right_face], dtype = np.uint8))
            self.solved_states.append(np.array([self.cube.back_face, self.cube.front_face, self.cube.right_face, self.cube.left_face, self.cube.down_face, self.cube.up_face], dtype = np.uint8))
            self.solved_states.append(np.array([self.cube.back_face, self.cube.front_face, self.cube.left_face, self.cube.right_face, self.cube.up_face, self.cube.down_face], dtype = np.uint8))

            self.solved_states.append(np.array([self.cube.right_face, self.cube.left_face, self.cube.up_face, self.cube.down_face, self.cube.front_face, self.cube.back_face], dtype = np.uint8))
            self.solved_states.append(np.array([self.cube.right_face, self.cube.left_face, self.cube.down_face, self.cube.up_face, self.cube.back_face, self.cube.front_face], dtype = np.uint8))
            self.solved_states.append(np.array([self.cube.right_face, self.cube.left_face, self.cube.front_face, self.cube.back_face, self.cube.up_face, self.cube.down_face], dtype = np.uint8))
            self.solved_states.append(np.array([self.cube.right_face, self.cube.left_face, self.cube.back_face, self.cube.front_face, self.cube.down_face, self.cube.up_face], dtype = np.uint8))

            self.solved_states.append(np.array([self.cube.left_face, self.cube.right_face, self.cube.up_face, self.cube.down_face, self.cube.back_face, self.cube.front_face], dtype = np.uint8))
            self.solved_states.append(np.array([self.cube.left_face, self.cube.right_face, self.cube.down_face, self.cube.up_face, self.cube.front_face, self.cube.back_face], dtype = np.uint8))
            self.solved_states.append(np.array([self.cube.left_face, self.cube.right_face, self.cube.front_face, self.cube.back_face, self.cube.up_face, self.cube.down_face], dtype = np.uint8))
            self.solved_states.append(np.array([self.cube.left_face, self.cube.right_face, self.cube.back_face, self.cube.front_face, self.cube.down_face, self.cube.up_face], dtype = np.uint8))
        # self.render()

    def get_state(self):
        return np.array([self.cube.up_face, self.cube.down_face, self.cube.front_face,
                        self.cube.back_face, self.cube.right_face, self.cube.left_face], 
                        dtype = np.int32)
        # represent 3 dimensional tensor with numpy -  since documentation uses np
        # Can we use tensorflow tensor instead?

    def step(self, action):
        self.action_dict[action]()  # () is needed
        self.state = self.get_state()
        reward = -1  # any rotation has a negative reward value
        done = False
        for solved_state in self.solved_states:
            if np.array_equal(self.state, solved_state): 
                done = True
        info = {}
        # self.render()
        return self.state, reward, done, info

    def render(self):  
        # Note that rendering will result in freezing the rest of the program until the window is closed since it has a mainloop
        # Don't want to run on a separate thread as tkinter is not thread safe
        window = RenderEnv(self.cube)
        window.mainloop()

    def reset(self, number_of_scrambles):  # number_of_scrambles is how many rotations to mess up the cube with
        self.cube = Cube(self.dim)
        for i in range(number_of_scrambles):
            action = np.random.choice(self.action_space.n)
            self.action_dict[action]()
        self.state = self.get_state()
        return self.state


def build_dqn(learning_rate, actions_output, input_shape):
    # learning_rate for Adam optimizer
    # action_output for the 12 possible actions which will be the output of the network
    # input shape is 6x3x3 tensor for 3x3 cube. This will be flattened so that it can be inputted into a dense layer

    # Sequential model is appropriate since no layer has multiple inputs or outputs and there are no residual connections
    model = keras.Sequential()
    model.add(keras.Input(shape = input_shape))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(25, activation = 'relu'))
    model.add(keras.layers.Dense(20, activation = 'relu'))
    model.add(keras.layers.Dense(actions_output))  # no activation needed here
    # Add batch normalisation?

    model.compile(optimizer = keras.optimizers.Adam(learning_rate), loss = 'mse')  # mean squared error for loss

    return model


class ExperienceReplay():
    def __init__(self, max_memory_size, input_shape, n_outputs):
        # n_outputs is number of outputs i.e.e the 12 rotations
        self.max_memory_size = max_memory_size
        self.input_shape = input_shape
        self.n_outputs = n_outputs

        self.state_memory = []  # could use numpy array but I don't think it is a limiting factor
        self.new_state_memory = []
        self.action_memory = []
        self.done_memory = []
        # the above 4 lists which are the information we need to store about any transition
        # we could have a 5th for the reward but every entry would be set to -1

    def store_transition(self, start_state, end_state, action, done):
        self.state_memory.append(start_state)
        self.new_state_memory.append(end_state)
        self.action_memory.append(action)
        self.done_memory.append(done)
    
        if len(self.done_memory) > self.max_memory_size:
            del self.state_memory[0]
            del self.new_state_memory[0]
            del self.action_memory[0]
            del self.done_memory[0]

    def get_sample(self, batch_size):
        indices = np.random.choice(range(len(self.done_memory)), size = batch_size)
        state_sample = np.array([self.state_memory[i] for i in indices], dtype = np.int32)
        new_state_sample = np.array([self.new_state_memory[i] for i in indices], dtype = np.int32)
        # two arrays above are numpy arrays since the will be passed into the dqn save_model
        action_sample = [self.action_memory[i] for i in indices]
        done_sample = np.array([self.done_memory[i] for i in indices], np.int32)  # actually needs to be np array for calculation later
        return state_sample, new_state_sample, action_sample, done_sample


# Could have just used DQNAgent from keras-rl module which has the inbuilt train method!
class Agent():
    def __init__(self, learning_rate, gamma, action_space_size, epsilon, input_shape, batch_size, 
                min_epsilon = 0.05, max_memory_size = 100000, update_target_network = 200):
        # gamma is the reward loss in the formula for the Q value
        # action_space_size is 12 for 12 quarter rotations defined in the environment
        # epsilon for epsilon greedy choice of actions. won't go below min_epsilon when later decrementing it
        # input_shape same as for the build dqn
        # update_after_actions is the number of actions taken before the learn function is called again
        # update_target_network is the number of actions taken before the target network is set to the current network
        self.gamma = gamma
        self.action_space = [i for i in range(action_space_size)]
        self.epsilon = epsilon
        self.epsilon_min = min_epsilon
        self.batch_size = batch_size
        self.update_target_network = update_target_network
        self.reward_sample = [-1 for i in range(self.batch_size)]  # put it here so it doesn't have to be loaded up every time we learn
        self.target_updater = 0  # used to determine when the target network should be updated

        self.memory = ExperienceReplay(max_memory_size, input_shape, action_space_size)
        self.q_val_net = build_dqn(learning_rate, action_space_size, input_shape)
        self.q_target_net = build_dqn(learning_rate, action_space_size, input_shape)
    
    def choose_action(self, state):
        if self.epsilon > np.random.rand(1)[0]:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_val_net.predict(state)
            action = np.argmax(actions)
        return action

    def store_transition(self, state, new_state, action, done):
        self.memory.store_transition(state, new_state, action, done)

    def learn(self):
        self.target_updater += 1
        if len(self.memory.done_memory) > self.batch_size:
            state_sample, new_state_sample, action_sample, done_sample = self.memory.get_sample(self.batch_size)
            new_state_target_eval = self.q_target_net.predict(new_state_sample, batch_size = self.batch_size)  # need batch_size to format data for model
            
            current_state_eval = self.q_val_net.predict(state_sample, batch_size = self.batch_size)
            
            target_eval = current_state_eval

            batch_index = np.arange(self.batch_size, dtype = np.int32)
            target_eval[batch_index, action_sample] = self.reward_sample + \
                                            self.gamma * np.max(new_state_target_eval, axis = 1) * (done_sample - 1)  # if done then target is larger
            # applying formula for Q values - updating reward for action the agent actually took
            # [batch_index, action_sample] are the indexes that we want to change. They are the actions the target network takes
            # we only change these ones so that the network only optimizes in their direction
            
            self.q_val_net.fit(state_sample, target_eval, verbose = 0)
            # takes loss from q_val_net.predict(state_sample) against updated_q_vals

            self.epsilon -= self.epsilon / 1000000  # one million
            self.epsilon = max(self.epsilon, self.epsilon_min)
            # updating epsilon

            if self.target_updater % self.update_target_network == 0:
                self.q_target_net.set_weights(self.q_val_net.get_weights())
                self.target_updater = 0
                # setting target network equal to value network

    def save_model(self):
        pass

    def load_model(self):
        pass

