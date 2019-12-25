import numpy as np
import tensorflow as tf
from replay_memory import ReplayMemory
from model import Model


class Agent:
    def __init__(self, input_shape, num_actions, hidden_units, lr, gamma, replace, epsilon=1.0, epsilon_min=0.01, epsilon_dec=0.999, mem_size=int(1e6), batch_size=20, q_eval_fname='q_eval.h5', q_target_fname='q_next.h5'):
        self.action_space = [i for i in range(num_actions)]
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_min
        self.batch_size = batch_size
        self.replace = replace
        self.q_target_model_file = q_target_fname
        self.q_eval_model_file = q_eval_fname
        self.learn_step = 0
        self.memory = ReplayMemory(mem_size, input_shape)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss_object=tf.keras.losses.mean_squared_error()
        self.loss_metric=tf.keras.metrics.Mean(name='Loss')
        self.q_eval = Model(input_shape, hidden_units, num_actions)
        self.q_next = Model(input_shape, hidden_units, num_actions)

    def replace_target_network(self):
        if self.replace is not None and self.learn_step % self.replace == 0:
            self.q_next.copy_weights(self.q_eval)

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def get_targets(self, state):
        return self.q_next(np.atleast_2d(state.astype(np.float32)))
    
    def get_action(self, state):
        return self.q_eval(np.atleast_2d(state.astype(np.float32)))

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.get_action(state)
            action = np.argmax(actions)

        return action

    @tf.function
    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            pass
        
        states, actions, rewards, new_states, dones = self.memory.sample_memory(self.batch_size)

        value_next = np.max(self.get_targets(new_states), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        with tf.GradientTape as tape:
            one_hot_actions = tf.one_hot(actions, self.num_actions) 
            selected_actions = tf.math.reduce_sum(self.get_action(states) * one_hot_actions, axis=1)
            loss = self.loss_object(actual_values, selected_actions)
        gradients = tape.gradients(loss, self.q_eval.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_eval.trainable_variables))
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        self.learn_step += 1

    def save_models(self):
        print("... Saving models ...")
        self.q_eval.save(self.q_eval_model_file)
        self.q_next.save(self.q_target_model_file)

    def load_models(self):
        print("... Loading models ...")
        self.q_eval = tf.keras.models.load_model(self.q_eval_model_file)
        self.q_next = tf.keras.models.load_model(self.q_target_model_file)