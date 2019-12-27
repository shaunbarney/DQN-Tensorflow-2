import gym
import numpy as np
import tensorflow as tf
from model import Model

def trace(x):
    return model(tf.expand_dims(x, 0))

env = gym.make('CartPole-v0')
input_shape = env.observation_space.shape
model = Model([200, 200], 4)

state = env.reset()
state = state.astype(np.float32)

logdir = 'logdir'
writer = tf.summary.create_file_writer(logdir)
tf.summary.trace_on(graph=True, profiler=True)
trace(state)
with writer.as_default():
    tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=logdir)