import gym
import tensorflow as tf
from datetime import datetime
from dqn import Agent

TRAIN = True

def train(env, agent):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/dqn/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)


def test(env, agent):
    pass

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    lr = 0.0005
    n_games = 500
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n
    hidden_units = [32, 32]
    gamma = 0.99
    replace = 1000
    agent = Agent(input_shape, num_actions, hidden_units, lr, gamma, replace)
    train(env, agent) if TRAIN else test(env, agent)
