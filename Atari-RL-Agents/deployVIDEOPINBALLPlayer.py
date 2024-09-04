#!/usr/bin/env python
# coding: utf-8

# In[1]:
import re
import os
import fnmatch
import tensorflow_probability as tfp
import tensorflow as tf
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.tf_py_environment import TFPyEnvironment
import tensorflow_probability as tfp
import imageio
import io


# In[2]:

def run_episodes_and_create_video(policy, eval_tf_env, eval_py_env, num_itr):
    num_episodes = 3
    frames = []
    for K in range(num_episodes):
        print(K)
        time_step = eval_tf_env.reset()
        frames.append(eval_py_env.render())
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = eval_tf_env.step(action_step.action)
            frames.append(eval_py_env.render())
    print('Done')
    gif_file = io.BytesIO()
    imageio.mimsave(gif_file, frames, format='gif', fps=60)
    
    gif_file_name = f'VideoPinball_{num_itr}.gif'
    with open(gif_file_name, 'wb') as f:
        gif_file.seek(0)  # Go to the beginning of the BytesIO buffer
        f.write(gif_file.read())  # Write the buffer content to file

    print(f'GIF saved to {gif_file_name}')


def extract_number(path):
    match = re.search(r'/(\d+)/$', path)
    return int(match.group(1)) if match else 0


def get_policy_paths(parent_dir):

    # List all items in the 'policy' directory that match the pattern
    policyList = fnmatch.filter(os.listdir(os.path.join(parent_dir, 'policy')), '*')

    # Filter out any non-directory files that may be present and sort numerically
    policy_dirs = [d for d in policyList if os.path.isdir(os.path.join(parent_dir, 'policy', d))]
    policy_dirs.sort()  # Assuming directories are named with just numbers

    policy_paths = []
    for elem in policy_dirs:
        if elem.isdigit():
            policy_path = os.path.join(f'{parent_dir}/policy/',f'{elem}/')
            policy_paths.append(policy_path)
    policy_paths = sorted(policy_paths, key=extract_number)
    return policy_paths


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

def main():
    policy_paths = get_policy_paths('./savedPolicy/VIDEOPINBALL')
    policy_dir = policy_paths[-1]
    print(policy_dir)

    max_episode_steps = 27000 # <=> 108k ALE frames since 1 step = 4 frames
    environment_name = "VideoPinball-v4"

    env = suite_atari.load(
        environment_name,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=[AtariPreprocessingWithAutoFire, FrameStack4])


    tf_env = TFPyEnvironment(env)

    saved_policy = tf.saved_model.load(policy_dir)

    dir_split = policy_dir.split('/')
    dir_str = '_'.join(dir_split[-3:-1])
    
    run_episodes_and_create_video(saved_policy, tf_env, env, dir_str)

# In[3]:
if __name__ == "__main__":
    main()