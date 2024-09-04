from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
import tensorflow as tf
import time
import os
import fnmatch
import json
import re
import matplotlib.pyplot as plt
from tf_agents.environments.tf_py_environment import TFPyEnvironment


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

import tf_agents
import numpy as np
import imageio
import io

def run_episodes_and_create_video(policy, eval_tf_env, eval_py_env, label):
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
    
    gif_file_name = f'myAgentPlays-VIDEOPINBALL-{label.upper()}.gif'
    with open(gif_file_name, 'wb') as f:
        gif_file.seek(0)  # Go to the beginning of the BytesIO buffer
        f.write(gif_file.read())  # Write the buffer content to file

    print(f'GIF saved to {gif_file_name}')

def extract_number(path):
    match = re.search(r'/(\d+)/$', path)
    return int(match.group(1)) if match else 0

# Sort the paths based on the extracted number

def get_sorted_pol_list(parent_dir):
    # List all items in the 'policy' directory that match the pattern
    policyList = fnmatch.filter(os.listdir(os.path.join(parent_dir, 'policy')), '*')

    # Filter out any non-directory files that may be present and sort numerically
    policy_dirs = [d for d in policyList if os.path.isdir(os.path.join(parent_dir, 'policy', d))]

    policyList = []
    for elem in policy_dirs:
        if elem.isdigit():
            policy_path = os.path.join(f'{parent_dir}/policy',f'{elem}/')
            policyList.append(policy_path)    
     # Sort the paths based on the extracted number
    policyList = sorted(policyList, key=extract_number)
      
    policyList = policyList[-10:]
    
    return policyList

#ADD ARGV BACK AS PARAMETER
def main(argv):
    numEpisodes=500
    #numEpisodes=10
    update_period=4
    epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=1.0, # initial ε
        decay_steps=250000 // update_period, # <=> 1,000,000 ALE frames
        end_learning_rate=0.01) # final ε
    discount_factor=.99
    max_episode_steps = 27000 # <=> 108k ALE frames since 1 step = 4 frames
    environment_name = "VideoPinball-v4"
    num_parallel=128
    #num_parallel = 2
    start=time.perf_counter_ns()

    env = suite_atari.load(
        environment_name,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=[AtariPreprocessingWithAutoFire, FrameStack4])
    
    tf_env = TFPyEnvironment(env)
    
    parEnvs=tf_agents.environments.ParallelPyEnvironment([lambda: suite_atari.load(
        environment_name,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=[AtariPreprocessingWithAutoFire, FrameStack4])]*num_parallel,
                                                         start_serially=False)
    minAct=parEnvs.action_spec().minimum
    maxAct=parEnvs.action_spec().maximum
    allResponses=[]    
    
    parent_dir = './savedPolicy/VIDEOPINBALL'
    policyList = get_sorted_pol_list(parent_dir)
    
    # print(policyList, len(policyList))
    
    cum_exp = []
    total_avg_disc = []
    
    for polname in policyList:
                
        pol_split = polname.split('/')
        trainIterNum = int(pol_split[-2])
        #print(trainIterNum)
        
        cum_exp.append(trainIterNum)
        
        saved_policy=tf.saved_model.load(polname)
        #trainIterNum=int(polname.rpartition('_')[2])
        
        epsilon=epsilon_fn(trainIterNum)
        rewAccum=np.zeros(num_parallel)
        discRewAccum=np.zeros(num_parallel)
        nStepsAccum=np.zeros(num_parallel)
        totRewL=[]
        totDiscRewL=[]
        rng=np.random.default_rng(1234)
        resp=parEnvs.reset()
        while len(totRewL)<numEpisodes:
            actionResps=saved_policy.action(resp).action
            actsToTake=np.where(rng.random(num_parallel)>epsilon,actionResps,rng.integers(low=minAct,high=maxAct,size=num_parallel))
            resp=parEnvs.step(actsToTake)
            rewAccum=rewAccum+resp.reward
            discRewAccum=discRewAccum+tf.math.multiply(resp.reward,
                                                       tf.math.pow(discount_factor,nStepsAccum)).numpy()
            nStepsAccum=nStepsAccum+1
            for el in np.argwhere(resp.step_type==2):#tf.where(resp.step_type==2): 
                totRewL.append(rewAccum[el])
                totDiscRewL.append(discRewAccum[el])
            nStepsAccum[resp.step_type==2]=np.zeros_like(nStepsAccum[resp.step_type==2])
            rewAccum[resp.step_type==2]=np.zeros_like(nStepsAccum[resp.step_type==2])
            discRewAccum[resp.step_type==2]=np.zeros_like(nStepsAccum[resp.step_type==2])
        totDiscRewL=np.array(totDiscRewL)
        totRewL=np.array(totRewL)
        print(polname)
        print(str(np.average(totDiscRewL))+' +/- '+str(np.std(totDiscRewL)/np.sqrt(len(totRewL))))
        print(str(np.average(totRewL))+' +/- '+str(np.std(totRewL)/np.sqrt(len(totRewL))))
        print(len(totRewL))
        total_avg_disc.append(np.average(totDiscRewL))
        allResponses.append({'trainIterNum':trainIterNum,'total_rewards':totRewL.tolist(),'total_discounted_rewards':totDiscRewL.tolist(),
                             'avg_total_rewards':np.average(totRewL),'avg_total_discounted_rewards':np.average(totDiscRewL),
                             'std_total_rewards':np.std(totRewL),'std_total_discounted_rewards':np.std(totDiscRewL)})
        with open('./policyRewardsInfo.json','w') as outFile:
            json.dump(allResponses,outFile)
        print(totDiscRewL)
        print(totRewL)
        
    plt.plot(cum_exp, total_avg_disc)    
    plt.xlabel('Cumulative Gameplay Experience')
    plt.ylabel('Long-Run Average Discounted Reward')
    plt.title('Training Curve')
    plt.savefig('TrainingCurve.png')
    
    poor, interm, best = policyList[0], policyList[5], policyList[-1]

    saved_policy = tf.saved_model.load(poor)
    run_episodes_and_create_video(saved_policy, tf_env, env, 'poor')

    saved_policy = tf.saved_model.load(interm)
    run_episodes_and_create_video(saved_policy, tf_env, env, 'intermediate')

    saved_policy = tf.saved_model.load(best)
    run_episodes_and_create_video(saved_policy, tf_env, env, 'best')
    
    end=time.perf_counter_ns()
    print(f'{end-start:.3f}')

    

    

if __name__ == '__main__':
    #main()
    tf_agents.system.multiprocessing.handle_main(main)
