import gym
import numpy as np

class ReplayMemory:
    def __init__(self, mem_size, input_shape):
        self.mem_size = mem_size
        self.mem_counter = 0

        self.state = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state =  np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action = np.zeros(self.mem_size, dtype=np.int32)
        self.reward = np.zeros(self.mem_size, dtype=np.float32)
        self.done = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_counter % self.mem_size
        self.state[index] = state
        self.action[index] = action
        self.reward[index] = reward
        self.new_state[index] = new_state
        self.done[index] = int(1-done)
        self.mem_counter += 1

    def sample_memory(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state[batch]
        actions = self.action[batch]
        rewards = self.reward[batch]
        new_states = self.new_state[batch]
        dones = self.done[batch]

        return states, actions, rewards, new_states, dones

    
if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    memory = ReplayMemory(int(1e6), (env.observation_space.shape))

    for i in range(5):
        s = env.reset()
        ep_reward = 0
        for j in range(500):
            a = env.action_space.sample()
            s_, r, d, i = env.step(a)
            memory.store_transition(s, a, r, s_, d)

    print(memory.mem_counter)
    for i in range(1):
        print(memory.sample_memory(i))