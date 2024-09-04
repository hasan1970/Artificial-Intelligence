#!/usr/bin/env python
# coding: utf-8

# In[2]:
from tf_agents.environments import suite_gym
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.tf_py_environment import TFPyEnvironment
import tensorflow as tf
from tensorflow import keras
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics
import logging
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.utils.common import function
from tf_agents.utils.common import Checkpointer
import os
from tf_agents.policies import policy_saver
import numpy as np
# In[6]:



class AtariPreprocessingWithAutoFire(AtariPreprocessing):
    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        super().step(5) # FIRE
        super().step(6)
        return obs
    
    def step(self, action):
        lives_before_action = self.ale.lives()
        obs, rewards, done, info = super().step(action)
        if self.ale.lives() < lives_before_action and not done:
            super().step(5) # FIRE
            super().step(6) # FIRE to start after life lost
        return obs, rewards, done, info
    
class ShowProgress:
    def __init__(self, total):
            self.counter = 0
            self.total = total
    def __call__(self, trajectory):
            if not trajectory.is_boundary():
                self.counter += 1
            if self.counter % 100 == 0:
                print("\r{}/{}".format(self.counter, self.total), end="")

def main():
    max_episode_steps = 27000 # <=> 108k ALE frames since 1 step = 4 frames
    environment_name = "VideoPinball-v4"
    env = suite_atari.load(
        environment_name,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=[AtariPreprocessingWithAutoFire, FrameStack4])

    env.seed(54)
    env.reset()

    for i in range(2,9):
        action = i
        #print(action, env.unwrapped.get_action_meanings()[action])
        time_step = env.step(action)

    
    tf_env = TFPyEnvironment(env)


    preprocessing_layer = tf.keras.layers.Lambda(lambda obs: tf.cast(obs, np.float32) / 255.)
    conv_layer_params=[(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
    fc_layer_params=[512]

    q_net = QNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        preprocessing_layers=preprocessing_layer,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params)


    train_step = tf.Variable(0)
    update_period = 4 # run a training step every 4 collect steps
    optimizer = keras.optimizers.RMSprop(learning_rate=2.5e-4, rho=0.95, momentum=0.0,
                                     epsilon=0.00001, centered=True)

    #PARAMETERS TO CHANGE FOR DIFF GAMES
    epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=1.0, # initial ε
        #TO CHANGE
        decay_steps= 250000 // update_period, # <=> 1,000,000 ALE frames
        end_learning_rate=0.01) # final ε

    agent = DqnAgent(tf_env.time_step_spec(),
                 tf_env.action_spec(),
                 q_network=q_net,
                 optimizer=optimizer,
                 target_update_period=2000, # <=> 32,000 ALE frames
                 td_errors_loss_fn=keras.losses.Huber(reduction="none"),
                 gamma=0.99, # discount factor
                 train_step_counter=train_step,
                 epsilon_greedy=lambda: epsilon_fn(train_step))

    agent.initialize()


    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=1000000)
    replay_buffer_observer = replay_buffer.add_batch

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    #log these
    logging.getLogger().setLevel(logging.INFO)
    log_metrics(train_metrics)

    collect_driver = DynamicStepDriver(
        tf_env,
        agent.collect_policy,
        observers=[replay_buffer_observer] + train_metrics,
        num_steps=update_period) # collect 4 steps for each training iteration





    initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
                                        tf_env.action_spec())
    init_driver = DynamicStepDriver(
        tf_env,
        initial_collect_policy,
        observers=[replay_buffer.add_batch, ShowProgress(4000)],
        num_steps=4000) # <=> 80,000 ALE frames
    final_time_step, final_policy_state = init_driver.run()


    dataset = replay_buffer.as_dataset(
        sample_batch_size=64,
        num_steps=2,
        num_parallel_calls=3).prefetch(3)

    collect_driver.run = function(collect_driver.run)
    agent.train = function(agent.train)


    ckpt_dir ='./lastModelCheckpoint/VIDEOPINBALL'

    global_step = tf.Variable(0, trainable=False, name='global_step')

    checkpoint_dir = os.path.join(ckpt_dir, 'checkpoint')

    train_checkpointer = Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=global_step
    )   

    pol_dir = './savedPolicy/VIDEOPINBALL'
    policy_dir = os.path.join(pol_dir, 'policy')
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)


# In[19]:


    def train_agent(n_iterations):
        c = 0
        num_points = 10
        save_period = n_iterations // num_points
        time_step = None
        policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
        iterator = iter(dataset)
    
        for iteration in range(n_iterations):
            time_step, policy_state = collect_driver.run(time_step, policy_state)
            trajectories, buffer_info = next(iterator)
            train_loss = agent.train(trajectories)
            print("\r{} loss:{:.5f}".format(iteration, train_loss.loss.numpy()), end="")
        
            if iteration % 5000 == 0:
                print(f'{iteration*100/n_iterations:.2f}%')
                log_metrics(train_metrics)
            
            global_step.assign_add(1)
            if iteration % save_period == 0 and iteration:
                c+=1
                #if iteration > 120000:
                itr_policy_dir = os.path.join(policy_dir, f'{iteration}')
                tf_policy_saver.save(itr_policy_dir)
                print(f'Saved Policy for {iteration}, global step {global_step.numpy()}')
            
                if c in [5,9]:
                    train_checkpointer.save(global_step)
                    print(f'Checkpoint saved for {iteration}, global step {global_step.numpy()}')


            

    train_agent(n_iterations=150001)


    train_checkpointer.save(global_step)


if __name__ == "__main__":
    main()
    print("Done!")
