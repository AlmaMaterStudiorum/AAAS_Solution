import numpy as np
from IPython import get_ipython
import gym
import random

resizeRatio = 10
frame = 1
RunningEnvironment = '???'

# higher GAMMA higher exploration lower exploitation
GAMMA = 0.90
alpha = 0.5
max_steps_for_episode = 10000
max_episodes = 1000000


epsilon      =  100000*1e-6
deltaepsilon =      10*1e-6


AtariName = "FrozenLake-v1"
fromzero = True
decrease = True
minimumepsilon = False

def Sarsa(env,observation_space_size,action_space_size,fromzero=True,decrease=False,slipper = True):
    global epsilon
    global minimumepsilon

    Q_VALUE = []
    if fromzero == True:
        Q_VALUE = [[0 for x in range(action_space_size)] for y in range(observation_space_size)] 
    else:
        Q_VALUE = [[random.uniform(0, 0.1) for x in range(action_space_size)] for y in range(observation_space_size)] 

    print("Start ")
    print(Q_VALUE)

    consecutiverewards = 0
    periodicrewards = 0
    # Loop episode
    for episode in range(max_episodes):

        rewards_of_this_episode = 0;
        state_t = env.reset()

        if RunningEnvironment == 'VS':
            state_t = state_t[0]

        action_t = 0       


        # Choose A' from A(s') using policy derived from Q (e.g., -greedy)
        optimal_action_index = np.argmax(Q_VALUE[state_t])

        probability_of_any_action_policy = []
        for i in range(0 ,action_space_size,1):
            probability_of_any_action_policy.append(0)
            if i == optimal_action_index:
                probability_of_any_action_policy[i] = 1 - epsilon    +    epsilon/action_space_size
            else:
                probability_of_any_action_policy[i] =                     epsilon/action_space_size


        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
        # prende un valore dall'array num_outputs con le probabilita probability_of_any_action_policy
        # baco di numpy , non calcola correttamente array a somma 1
        p = probability_of_any_action_policy / np.sum(probability_of_any_action_policy)

        action_t = np.random.choice(action_space_size, p=p)

        # Loop steps
        for steps in range(max_steps_for_episode):

            state_t_1 = 0
            reward_t_1 = 0

            # -------------------------------------------------------------------------
            # Take action A , observe S' , R
            if RunningEnvironment == 'VS':
                state_t_1, reward_t_1, done, truncated , info   = env.step(action_t)
            else:
                state_t_1, reward_t_1, done, _                  = env.step(action_t)

            rewards_of_this_episode += reward_t_1
            # env.render()

            # Choose A' from A(s') using policy derived from Q (e.g., -greedy)
            optimal_action_index = np.argmax(Q_VALUE[state_t_1])

            probability_of_any_action_policy = []
            for i in range(0 ,action_space_size,1):
                probability_of_any_action_policy.append(0)
                if i == optimal_action_index:
                    probability_of_any_action_policy[i] = 1 - epsilon    +    epsilon/action_space_size
                else:
                    probability_of_any_action_policy[i] =                     epsilon/action_space_size


            # https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
            # prende un valore dall'array num_outputs con le probabilita probability_of_any_action_policy
            # baco di numpy , non calcola correttamente array a somma 1
            p = probability_of_any_action_policy / np.sum(probability_of_any_action_policy)

            action_t_1 = np.random.choice(action_space_size, p=p)

            currentQValue =  Q_VALUE[state_t][action_t]
            nextQValue =  Q_VALUE[state_t_1][action_t_1]

            Q_VALUE[state_t][action_t] = currentQValue + alpha*(reward_t_1 + GAMMA*nextQValue  - currentQValue)

            if done == True:
              if reward_t_1 > 0: 
                # Goal
                if decrease == True:
                    if epsilon > deltaepsilon :
                      epsilon = epsilon - deltaepsilon
                    else:
                      epsilon = deltaepsilon
                      if minimumepsilon == False:
                          minimumepsilon = True
                          print("Q_VALUE (minimumepsilon): " , Q_VALUE)

                consecutiverewards += 1
                periodicrewards +=1
                if consecutiverewards > 40:
                  print("episode :",episode, ", Q :", Q_VALUE[state_t][action_t])
                  print("consecutiverewards : ", consecutiverewards)
                  print(Q_VALUE)
              else:
                # Hole
                consecutiverewards = 0
              break

            state_t = state_t_1
            action_t = action_t_1

        if episode % 10000 == 0 :
           print("episode :" , episode , "; periodicrewards : ", periodicrewards)
           periodicrewards = 0




if 'google.colab' in str(get_ipython()):
    print('Running on CoLab')
    RunningEnvironment = 'COLAB'
else:
    print('Not running on CoLab')
    RunningEnvironment = 'VS'
# hyperparameters
# if torch.cuda.is_available():
#    print('cuda.is_available')

#else:
#    print('cuda.is_not_available')

desc=["SFFF", "FHFH", "FFFH", "HFFG"]
# ,is_slippery=False
slipper = True
env = gym.make(AtariName,desc=desc,is_slippery=slipper)

observation_space = env.observation_space.n
action_space = env.action_space.n

Sarsa(env,observation_space,action_space,fromzero , decrease,slipper)