import gym
import numpy as np
import tensorflow as tf
from datetime import datetime
from dqn import Agent

TRAIN = True
NEGATIVE_REWARD = 0

def play_game(env: gym.Env, agent: Agent):
    rewards = 0
    done = False
    state = env.reset()
    while not done:
        action = agent.choose_action(state)
        new_state, reward, done, _ = env.step(action)
        rewards += reward
        if done:
            reward -= NEGATIVE_REWARD
            env.reset()
        
        agent.store_transition(state, action, reward, new_state, done)
        if agent.memory.mem_counter > agent.batch_size:
            agent.learn()
    
    return rewards


def train(env: gym.Env, agent: Agent):
    best = 0
    N = 50000
    total_rewards = np.empty(N)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/dqn/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)
    for n in range(N):
        total_reward = play_game(env, agent)
        total_rewards[n] = total_reward
        avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
        with summary_writer.as_default():
            tf.summary.scalar('episode reward', total_reward, step=n)
            tf.summary.scalar('running avg reward(100)', avg_rewards, step=n)
        if n % 100 == 0:
            print("episode:", n, "episode reward:", total_reward, "epsilon:", agent.epsilon, "avg reward (last 100):", avg_rewards)
        if avg_rewards > best:
            print(f"New best by {avg_rewards-best:.3f}")
            best = avg_rewards
            agent.save_models()
            
    print("avg reward for last 100 episodes:", avg_rewards)

def test(env: gym.Env, agent: Agent):
    pass

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    batch_size = 32
    lr = 0.001
    n_games = 500
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n
    hidden_units = [32, 64]
    gamma = 0.9
    replace = 100
    print(f"lr:{lr}\thidden:{hidden_units}\tgamme:{gamma}\treplace:{replace}\tbatch:{batch_size}")
    agent = Agent(input_shape, num_actions, hidden_units, lr, gamma, replace, batch_size=batch_size)
    train(env, agent) if TRAIN else test(env, agent)
