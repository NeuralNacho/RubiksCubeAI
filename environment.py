import gym
import numpy as np
import random
from tkinter import *
from Cube import *
from RenderEnv import *
from tensorflow import *

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
                                                shape = (6, self.dim, self.dim), dtype = np.uint8)
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
                        dtype = np.uint8)
        # represent 3 dimensional tensor with numpy -  since documentation uses np
        # Can we use tensorflow tensor instead?

    def step(self, action):
        self.action_dict[action]()
        self.state = self.get_state()
        reward = -1  # any rotation has a negative reward value
        done = False
        for solved_state in self.solved_states:
            if np.array_equal(self.state, solved_state):  # WHAT ABOUT 2x2 WHERE CENTRE's NOT FIXED
                done = True
        info = {}
        # self.render()
        return self.state, reward, done, info

    def render(self):  
        # Note that rendering will result in freezing the rest of the program until the window is closed since it has a mainloop
        # Don't want to run on a separate thread as tkinter is not thread safe
        window = RenderEnv(self.cube)
        window.mainloop()

    def reset(self):
        self.cube = Cube(self.dim)
        self.state = self.solved_states[0]


class Agent()

