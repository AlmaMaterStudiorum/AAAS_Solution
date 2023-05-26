from cmath import isnan
from inspect import Parameter
from locale import normalize
from pickle import FALSE
import sys
import gym
from gymnasium.wrappers import ResizeObservation
from gymnasium.wrappers import GrayScaleObservation
from gymnasium.wrappers import FlattenObservation
# import gymnasium as gym
import torch
import os

import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import grad
import matplotlib.pyplot as plt
import pandas as pd
from IPython import get_ipython
import math
from QValueFunctionApproximation import QValueFunctionApproximation
from Policy import Policy
from torch.utils.data import Dataset
import Hyperparameters
import Parameters

from json import JSONEncoder
import json




epsilon = 10*1e-2
class EncodeTensor(JSONEncoder,Dataset):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(NpEncoder, self).default(obj)


def normalizestate(state,frame):
    den = frame * 255
    somma = 0
    for x in state:
        somma += x
    
    somma = somma/den
    return somma

# Semi-gradient One-step Sarsa Control On Atari
def sgosscOnAtari(env,policytype,observation_space_size,action_space_size,frame):
     print("observation_space : ",observation_space_size , ",action_space :",action_space_size)

     policy = Policy


     list_of_steps_of_all_episodes = []
     list_of_average_steps_sampled_every_n_episodes = []
     list_of_rewards_of_all_episodes = []

     bestepisode=0
     rewardsbestepisode = 0  
     step_counter = 0
     isfreekill = False
     israndomaction = False

     # Instazia l'oggetto della classe della rete neurale 
     # Instazia l'optimizer
     # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
     funtionapproximations = []
     optimizers = [] 

     hiddensize = 1
     hiddenlayers = []
     for h in range(hiddensize -1,-1,-1):
         hiddenlayers.append(action_space_size*(2**h))

     for x in range(action_space_size):
        # hiddenlayers = [action_space_size*8,action_space_size*4,action_space_size*2]
        
        functionapproximation = QValueFunctionApproximation(observation_space_size,action_space_size,hiddenlayers,Hyperparameters.learning_rate)
        # functionapproximation.requires_grad=True
        funtionapproximations.append(functionapproximation)
        optimizator = optim.Adam(functionapproximation.parameters() , lr=Hyperparameters.learning_rate)
        optimizers.append(optimizator);


     for episode in range(Parameters.max_episodes):

        rewards_of_this_episode = []
        rewards_of_this_episode_me = []
        rewards_of_this_episode_opponent = []
        I = 1
        G = 0

        # Variable used in methods

        q_action_value_t = 0

        # https://medium.com/@ashish_fagna/understanding-openai-gym-25c79c06eccb
        # env.reset(): This command will reset the environment as shown below in screenshot. It returns an initial state
        # S,A <- initial state and action of episode
        state_t = env.reset()

        if Parameters.RunningEnvironment == 'VS':
                state_t = state_t[0]

        state_t = normalizestate(state_t,1)
        action_t = 0       
        list_of_actions = [0 for i in range(action_space_size)]

        for steps in range(Parameters.max_steps_for_episode):
            # actor_output is POLICY
            # pi(a|s) is the probability of taking action a in state s under policy pi (pag 74 , RLAI)
            # print(steps)
            q_action_value_t = funtionapproximations[action_t].forward(state_t)

            list_of_actions[action_t] += 1
            # -------------------------------------------------------------------------
            # Take action A , observe S' , R
            if Parameters.RunningEnvironment == 'VS':
                state_t_1, reward_t_1, done, truncated , info   = env.step(action_t)
            else:
                state_t_1, reward_t_1, done, _                  = env.step(action_t)

            state_t_1 = normalizestate(state_t_1,frame)

            env.render()



            rewards_of_this_episode.append(reward_t_1)
            if(reward_t_1 > 0):
                rewards_of_this_episode_me.append(reward_t_1)
            
            if(reward_t_1 < 0):           
                rewards_of_this_episode_opponent.append(abs(reward_t_1))

            #if(reward_t_1 == 0):    
            #    reward_t_1 = 1e-2
            crit = nn.MSELoss()
            if done == True :
                # If is terminal:
                loss =  Parameters.alpha*(reward_t_1 - q_action_value_t.item())*q_action_value_t
                # loss = crit(q_action_value_t, reward_t_1)
                pass
            else:


                # Choose A' as a function of (e.g., epsilon-greedy)
                array_action_value_t = []
                for x in range(action_space_size):
                    fa = funtionapproximations[x]
                    qv = fa.forward(state_t_1)
                    array_action_value_t.append( qv )
            
                if policytype == 0:
                    action_t_1,q_action_value_t_1 = policy.getActionAndValueFromStocasticPolicyV2(array_action_value_t)
                elif policytype == 1:
                    action_t_1,q_action_value_t_1 = policy.getActionAndValueFromGreedyPolicy(array_action_value_t)
                elif policytype == 2:
                    action_t_1,q_action_value_t_1 = policy.getActionAndValueFromEpsilonGreedyPolicy(array_action_value_t,epsilon)
                elif policytype == 3:
                    action_t_1,q_action_value_t_1 = policy.getActionAndValueFromSoftmaxPolicy(array_action_value_t)
                elif policytype == 4:
                    step_counter =  step_counter + 1 
                    action_t_1,q_action_value_t_1 = policy.getActionAndValueFromDecresingEpsilonGreedyPolicy(array_action_value_t,epsilon,step_counter)

                # t = q_action_value_t
                # q_action_value_t_1.requires_grad = False
                # t_1 =  q_action_value_t_1
                loss =  Parameters.alpha*(reward_t_1 + Parameters.GAMMA*q_action_value_t_1.item() - q_action_value_t.item() )*q_action_value_t
                
                # loss = crit(q_action_value_t, reward_t_1 + Parameters.GAMMA*q_action_value_t_1)
                # https://github.com/pytorch/pytorch/issues/39279
                # https://discuss.pytorch.org/t/implementing-sarsa-weight-updates/10324

                #  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # https://github.com/LxzGordon/Deep-Reinforcement-Learning-with-pytorch/blob/master/algorithm/value-based/Sarsa.py

            if isfreekill == True:
               action_t = 0

            if israndomaction == True:
               action_t = np.random.randint(action_space_size)
            
            # la parte di pytorch optimazer + backward , zero + step
            # How autograd works
            # https://pytorch.org/blog/overview-of-pytorch-autograd-engine/#:~:text=PyTorch%20computes%20the%20gradient%20of,ways%3B%20forward%20and%20reverse%20mode.
            a = list(funtionapproximations[action_t].parameters())[0].clone()
            #grad, = torch.autograd.grad(loss, q_action_value_t, allow_unused=True, create_graph=True)[0]
            loss.backward()
            # grad.backward()
            # print(q_action_value_t.grad)
            # print(q_action_value_t.is_leaf)
            optimizers[action_t].step()  
            b = list(funtionapproximations[action_t].parameters())[0].clone()
            # print(list(funtionapproximations[action_t].parameters())[0].grad)
            e = torch.equal(a.data, b.data)
            optimizers[action_t].zero_grad()

            state_t = state_t_1

            action_t = action_t_1

            last_n_episodes = 1
            if done or steps == Parameters.max_steps_for_episode-1:
                list_of_steps_of_all_episodes.append(steps)
                sum_of_rewards_of_all_steps_of_this_episode = 0
                if episode % last_n_episodes == 0:     

                    sum_of_rewards_of_all_steps_of_this_episode = np.sum(rewards_of_this_episode)

                    list_of_rewards_of_all_episodes.append(sum_of_rewards_of_all_steps_of_this_episode) 

                    list_of_steps_of_last_n_episodes = list_of_steps_of_all_episodes[-last_n_episodes:]

                    average_steps_of_last_n_episodes = np.mean(list_of_steps_of_last_n_episodes)

                    deviation_standard_of_steps_of_last_n_episodes = np.std(list_of_steps_of_last_n_episodes)

                    list_of_average_steps_sampled_every_n_episodes.append(average_steps_of_last_n_episodes)

                    devstd2 = "%.2f" % round(deviation_standard_of_steps_of_last_n_episodes, 2)

                    # sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} , dev.standard : {}\n".format(episode, sum_of_rewards_of_all_steps_of_this_episode, steps, list_of_average_steps_sampled_every_n_episodes[-1],devstd2))
                    
                    fireaction = 0
                    for i,x in enumerate(list_of_actions):
                        if i == 1 :
                            fireaction +=x
                        if i >= 10 :
                            fireaction +=x

                    sys.stdout.write("episode: {}, reward: {}, total length: {}, score : {}-{}  \n".format(episode, sum_of_rewards_of_all_steps_of_this_episode, steps, np.sum(rewards_of_this_episode_me),np.sum(rewards_of_this_episode_opponent)))
                    print("fireaction : ", fireaction ,", hit ",np.sum(rewards_of_this_episode_me) ,"ratio : ",np.sum(rewards_of_this_episode_me)/fireaction )
                    print(list_of_actions)
                sum_of_rewards_of_all_steps_of_this_episode = np.sum(rewards_of_this_episode)
                if sum_of_rewards_of_all_steps_of_this_episode > rewardsbestepisode:
                    rewardsbestepisode = sum_of_rewards_of_all_steps_of_this_episode
                    bestepisode = episode
                    print("#  bestepisode : ", bestepisode , ", rewardsbestepisode : ", rewardsbestepisode)

                break

        # end of episode
        if False :
            for x in range(action_space_size):

                filename = "Boxing" + str(episode) + "_" + str(x) + ".nn"
                result = os.path.join(r"D:\Data\Temp\NN", filename)
                torch.save(funtionapproximations[x].state_dict(), result)


        if True :
            for x in range(action_space_size):
                filename = "Boxing" + str(episode) + "_" + str(x) + ".json"
                result = os.path.join(Parameters.NNFolder, filename)
                with open(result, 'w') as json_file:
                    json.dump(funtionapproximations[x].state_dict(), json_file,cls=EncodeTensor)





          
     # Plot results
     smoothed_rewards = pd.Series.rolling(pd.Series(list_of_rewards_of_all_episodes), 10).mean()
     smoothed_rewards = [elem for elem in smoothed_rewards]
     plt.plot(list_of_rewards_of_all_episodes)
     plt.plot(smoothed_rewards)
     plt.plot()
     plt.xlabel('Episode')
     plt.ylabel('Reward')
     plt.show()

     plt.plot(list_of_steps_of_all_episodes)
     plt.plot(average_steps_of_last_n_episodes)
     plt.xlabel('Episode')
     plt.ylabel('Episode length')
     plt.show()




