import sys
import pylab
import random
import numpy as np
import os
import time, datetime
from collections import deque
from keras.layers import *
from keras.models import Sequential,Model
import keras
from keras import backend as K_back
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D, MaxPooling2D

state_size = 64
action_size = 5

model_path = "save_model/"
graph_path = "save_graph/"

if not os.path.isdir(model_path):
    os.mkdir(model_path)

if not os.path.isdir(graph_path):
    os.mkdir(graph_path)
    
load_model = True
n_ticks = 11

class Predator_prey:
    
    def __init__(self):
        self.n_ticks = n_ticks
        
    def reset_env(self):

        n_rows = self.n_ticks+2
        n_cols = self.n_ticks+2

        state_prey      = np.zeros((self.n_ticks,self.n_ticks))
        state_predators = np.zeros((self.n_ticks,self.n_ticks))

        self.prey_row = 0
        self.prey_col = 0
        self.predator_rows = np.zeros((1,4), dtype=int)[0]
        self.predator_cols = np.zeros((1,4), dtype=int)[0]
        
        """
        len_unique = 0
        while len_unique != 5:
            init_state = np.random.randint(low=0, high=64, size=5)
            len_unique = len(np.unique(init_state))
                        
        row_idx, col_idx = divmod(init_state[0],n_ticks)
        """
        row_idx, col_idx = 5,5 
        state_prey[row_idx][col_idx] = 1
        self.prey_row = row_idx
        self.prey_col = col_idx
        
        """
        for idx in range(4):
            row_idx,col_idx = divmod(init_state[idx+1],n_ticks)
            self.predator_rows[idx] = row_idx
            self.predator_cols[idx] = col_idx
        """
        
        self.game_prey = np.zeros((n_rows,n_cols))
        self.game_prey[1:n_ticks+1,1:n_ticks+1] = state_prey

        self.predator_0 = np.zeros((n_rows,n_cols))
        self.predator_1 = np.zeros((n_rows,n_cols))
        self.predator_2 = np.zeros((n_rows,n_cols))
        self.predator_3 = np.zeros((n_rows,n_cols))

        self.prey_row += 1
        self.prey_col += 1
        self.predator_rows = [1,1,n_ticks,n_ticks]
        self.predator_cols = [1,n_ticks,1,n_ticks]
        
        self.predator_0[self.predator_rows[0]][self.predator_cols[0]] = 2
        self.predator_1[self.predator_rows[1]][self.predator_cols[1]] = 2
        self.predator_2[self.predator_rows[2]][self.predator_cols[2]] = 2
        self.predator_3[self.predator_rows[3]][self.predator_cols[3]] = 2

        self.game_arr_frame = np.full((n_rows, n_cols), 8)
        self.game_arr_frame[1:n_ticks+1,1:n_ticks+1] = np.zeros((n_ticks,n_ticks))
        self.game_arr = self.game_arr_frame + self.game_prey + self.predator_0 + self.predator_1 + self.predator_2 + self.predator_3
        
        return self.game_arr
        
    def frame_step(self, act_arr):
        
        done = False
        
        # Assume that below action is correct
        for idx in range(4):
            if act_arr[idx] == 0:
                if self.game_arr[self.predator_rows[idx]+1][self.predator_cols[idx]] == 0:
                    self.predator_rows[idx] += 1
            if act_arr[idx] == 1:
                if self.game_arr[self.predator_rows[idx]-1][self.predator_cols[idx]] == 0:
                    self.predator_rows[idx] -= 1
            if act_arr[idx] == 2:
                if self.game_arr[self.predator_rows[idx]][self.predator_cols[idx]-1] == 0:
                    self.predator_cols[idx] -= 1
            if act_arr[idx] == 3:
                if self.game_arr[self.predator_rows[idx]][self.predator_cols[idx]+1] == 0:
                    self.predator_cols[idx] += 1

        self.predator_0 = np.zeros((n_ticks+2,n_ticks+2))
        self.predator_1 = np.zeros((n_ticks+2,n_ticks+2))
        self.predator_2 = np.zeros((n_ticks+2,n_ticks+2))
        self.predator_3 = np.zeros((n_ticks+2,n_ticks+2))            
        self.predator_0[self.predator_rows[0]][self.predator_cols[0]] = 2
        self.predator_1[self.predator_rows[1]][self.predator_cols[1]] = 2
        self.predator_2[self.predator_rows[2]][self.predator_cols[2]] = 2
        self.predator_3[self.predator_rows[3]][self.predator_cols[3]] = 2

        self.game_arr = self.game_arr_frame + self.game_prey + self.predator_0 + self.predator_1 + self.predator_2 + self.predator_3
        
        distances = np.zeros((1,4))[0]
            
        for idx in range(4):
            temp_dis = np.abs(self.predator_rows[idx] - self.prey_row) \
                    + np.abs(self.prey_col-self.predator_cols[idx])
            distances[idx] = temp_dis

        if self.game_arr[self.prey_row+1][self.prey_col] != 0:
            if self.game_arr[self.prey_row-1][self.prey_col] != 0:
                if self.game_arr[self.prey_row][self.prey_col-1] != 0:
                    if self.game_arr[self.prey_row][self.prey_col+1] != 0:
                        done = True
                        
        reward = np.zeros((1,4))[0]
        if done:
            reward[0] = 0
            reward[1] = 0
            reward[2] = 0
            reward[3] = 0
        else:
            reward[0] = -1 - np.sum(distances) - distances[0] + 4
            reward[1] = -1 - np.sum(distances) - distances[1] + 4
            reward[2] = -1 - np.sum(distances) - distances[2] + 4
            reward[3] = -1 - np.sum(distances) - distances[3] + 4
            
        return self.game_arr, reward, done
    
class Duelingdqn_agnt:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        # get size of state and action
        self.progress = " "
        self.action_size = action_size
        self.state_size = state_size
        
        self.episode = 0
        
        # These are hyper parameters for the Duelingdqn_agnt
        self.learning_rate = 0.001
        self.discount_factor = 0.99
                
        self.hidden1, self.hidden2 = 251, 251
        
        self.ep_trial_step = 1000
        
        # Parameter for Experience Replay
        self.size_replay_memory = 10000
        self.batch_size = 32
        
        self.input_shape = (n_ticks,n_ticks,1)
        
        # Experience Replay 
        self.memory = deque(maxlen=self.size_replay_memory)
        
        # create main model and target model
        self.model = self.build_model()

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        
        state = Input(shape=self.input_shape)        
        
        net1 = Convolution2D(32, kernel_size=(3, 3),activation='relu', \
                             padding = 'valid', input_shape=self.input_shape)(state)
        net2 = Convolution2D(64, kernel_size=(3, 3), activation='relu', padding = 'valid')(net1)
        net3 = MaxPooling2D(pool_size=(2, 2))(net2)
        net4 = Flatten()(net3)
        lay_2 = Dense(units=self.hidden2,activation='relu',kernel_initializer='he_uniform',\
                  name='hidden_layer_1')(net4)
        value_= Dense(units=1,activation='linear',kernel_initializer='he_uniform',\
                      name='Value_func')(lay_2)
        ac_activation = Dense(units=self.action_size,activation='linear',\
                              kernel_initializer='he_uniform',name='action')(lay_2)
        
        #Compute average of advantage function
        avg_ac_activation = Lambda(lambda x: K_back.mean(x,axis=1,keepdims=True))(ac_activation)
        
        #Concatenate value function to add it to the advantage function
        concat_value = Concatenate(axis=-1,name='concat_0')([value_,value_])
        concat_avg_ac = Concatenate(axis=-1,name='concat_ac_{}'.format(0))([avg_ac_activation,avg_ac_activation])

        for i in range(1,self.action_size-1):
            concat_value = Concatenate(axis=-1,name='concat_{}'.format(i))([concat_value,value_])
            concat_avg_ac = Concatenate(axis=-1,name='concat_ac_{}'.format(i))([concat_avg_ac,avg_ac_activation])

        #Subtract concatenated average advantage tensor with original advantage function
        ac_activation = Subtract()([ac_activation,concat_avg_ac])
        
        #Add the two (Value Function and modified advantage function)
        merged_layers = Add(name='final_layer')([concat_value,ac_activation])
        model = Model(inputs = state,outputs=merged_layers)
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        q_value = self.model.predict(state)
        return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        #in every action put in the memory
        self.memory.append((state, action, reward, next_state, done))
                        
def main():
    
    # Duelingdqn_agnt 에이전트의 생성
    agnt_0 = Duelingdqn_agnt(state_size, action_size)
    agnt_1 = Duelingdqn_agnt(state_size, action_size)
    agnt_2 = Duelingdqn_agnt(state_size, action_size)
    agnt_3 = Duelingdqn_agnt(state_size, action_size)
    
    game = Predator_prey()
    
    if load_model:
        agnt_0.model.load_weights(model_path + "/Preda_prey_dueling_0.h5")
        agnt_1.model.load_weights(model_path + "/Preda_prey_dueling_1.h5")
        agnt_2.model.load_weights(model_path + "/Preda_prey_dueling_2.h5")
        agnt_3.model.load_weights(model_path + "/Preda_prey_dueling_3.h5")

    last_n_game_score = deque(maxlen=20)
    last_n_game_score.append(agnt_0.ep_trial_step)
    avg_ep_step = np.mean(last_n_game_score)
    
    display_time = datetime.datetime.now()
    print("\n\n Game start at :",display_time)
    
    start_time = time.time()
    agnt_0.episode = 0
    time_step = 0
    
    while agnt_0.episode < 20:
        
        # reset environment
        game.game_arr = game.reset_env()
        
        done = False
        ep_step = 0
        
        act_arr = np.zeros((1,4), dtype=int)[0]
        
        while not done and ep_step < agnt_0.ep_trial_step:
            
            ep_step += 1
            time_step += 1
            
            # prey action define
            prey_action = random.randrange(5)
            if prey_action == 0:
                if game.game_arr[game.prey_row+1][game.prey_col] == 0:
                    game.prey_row += 1
            if prey_action == 1:
                if game.game_arr[game.prey_row-1][game.prey_col] == 0:
                    game.prey_row -= 1
            if prey_action == 2:
                if game.game_arr[game.prey_row][game.prey_col-1] == 0:
                    game.prey_col -= 1
            if prey_action == 3:
                if game.game_arr[game.prey_row][game.prey_col+1] == 0:
                    game.prey_col += 1
            # print("Prey Action :",prey_action)
            
            game.game_prey = np.zeros((n_ticks+2,n_ticks+2))
            game.game_prey[game.prey_row][game.prey_col] = 1
            
            game.game_arr = game.game_arr_frame + game.game_prey + game.predator_0 + game.predator_1 + game.predator_2 + game.predator_3
            # print(game_arr.astype(int))
            # sys.exit()
            
            # print(game_arr)
            
            state_t = game.game_arr[1:n_ticks+1,1:n_ticks+1]
            state = copy.deepcopy(state_t)
            state = state.reshape(1,n_ticks,n_ticks,1)
            
            # agent action define
            act_arr[0] = agnt_0.get_action(state)
            act_arr[1] = agnt_1.get_action(state)
            act_arr[2] = agnt_2.get_action(state)
            act_arr[3] = agnt_3.get_action(state)
            
            # print("Action :",act_arr)
            
            next_game_arr, reward, done = game.frame_step(act_arr)
            
            next_state_t = next_game_arr[1:n_ticks+1,1:n_ticks+1]
            next_state = copy.deepcopy(next_state_t)
            
            # print(next_state.astype(int))
            
            # sys.exit()
            
            next_state = next_state.reshape(1,n_ticks,n_ticks,1)
            
            agnt_0.append_sample(state, act_arr[0], reward[0], next_state, done)
            agnt_1.append_sample(state, act_arr[1], reward[1], next_state, done)
            agnt_2.append_sample(state, act_arr[2], reward[2], next_state, done)
            agnt_3.append_sample(state, act_arr[3], reward[3], next_state, done)
          
            state = next_state
            
            if done or ep_step == agnt_0.ep_trial_step:
                agnt_0.episode += 1
                last_n_game_score.append(ep_step)
                avg_ep_step = np.mean(last_n_game_score)
                print("episode :{:>5d} / ep_step :{:>5d} / last 20 game avg :{:>4.1f}".format(agnt_0.episode, ep_step, avg_ep_step))
                break
    
    e = int(time.time() - start_time)
    print(' Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
    sys.exit()
                    
if __name__ == "__main__":
    main()
