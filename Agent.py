from tensorflow import keras
import numpy as np


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
        state_sample = np.array([self.state_memory[i] for i in indices])
        new_state_sample = np.array([self.new_state_memory[i] for i in indices])
        # two arrays above are numpy arrays since the will be passed into the dqn save_model
        action_sample = [self.action_memory[i] for i in indices]
        done_sample = [self.done_memory[i] for i in indices]
        return state_sample, new_state_sample, action_sample, done_sample


class Agent():
    def __init__(self, learning_rate, gamma, action_space_size, epsilon, input_shape, batch_size, 
                min_epsilon = 0.05, max_memory_size = 100000, update_target_network = 1000):
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

    def store_transition(self, state, action, new_state, done):
        self.memory.store_transition(state, new_state, action, done)  # TODO: change order

    def learn(self):
        if len(self.memory.done_memory) > self.batch_size:
            state_sample, new_state_sample, action_sample, done_sample = self.memory.get_sample(self.batch_size)

            new_state_target_eval = self.q_target_net.predict(new_state_sample)

            current_state_eval = self.q_val_net.predict(state_sample)
            target_eval = current_state_eval

            batch_index = np.arange(self.batch_size, dtype = np.int32)
            target_eval[batch_index, action_sample] = self.rewards_sample + \
                                            self.gamma * np.max(new_state_target_eval, axis = 1) * (1- done_sample) 
            # applying formula for Q values
            # [batch_index, action_sample] are the indexes that we want to change. They are the actions the target network takes
            # we only change these ones so that the network only optimizes in their direction
            
            self.q_val_net.fit(state_sample, target_eval, verbose = 0)
            # takes loss from q_val_net.predict(state_sample) against updated_q_vals

            self.epsilon -= self.epsilon / 1000000  # one million
            self.epsilon = max(self.epsilon, self.epsilon_min)
            # updating epsilon

            if self.action_counter % self.update_target_network == 0:
                self.q_target_net.set_weights(self.q_val_net.model.get_weights())
                # setting target network equal to value network

    def save_model(self):
        pass

    def load_model(self):
        pass