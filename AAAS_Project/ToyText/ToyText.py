import Parameters
import numpy as np
from IPython import get_ipython
import gym
import random


def Sarsa(env,observation_space_size,action_space_size):


    Q_VALUE = [[random.uniform(-0.5, 0.5) for x in range(action_space_size)] for y in range(observation_space_size)] 

    # Loop episode
    for episode in range(Parameters.max_episodes):

        state_t = env.reset()

        if Parameters.RunningEnvironment == 'VS':
            state_t = state_t[0]

        action_t = 0       
        optimal_action_index = np.argmax(Q_VALUE[state_t])

        probability_of_any_action_policy = []
        for i in range(0 ,action_space_size,1):
            probability_of_any_action_policy.append(0)
            if i == optimal_action_index:
                probability_of_any_action_policy[i] = 1 - Parameters.epsilon    +    Parameters.epsilon/action_space_size
            else:
                probability_of_any_action_policy[i] =                                Parameters.epsilon/action_space_size

        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
        # prende un valore dall'array num_outputs con le probabilita probability_of_any_action_policy
        # baco di numpy , non calcola correttamente array a somma 1
        p = probability_of_any_action_policy / np.sum(probability_of_any_action_policy)

        action_t = np.random.choice(action_space_size, p=p)

        # Loop steps
        for steps in range(Parameters.max_steps_for_episode):

            state_t_1 = 0
            reward_t_1 = 0

            # -------------------------------------------------------------------------
            # Take action A , observe S' , R
            if Parameters.RunningEnvironment == 'VS':
                state_t_1, reward_t_1, done, truncated , info   = env.step(action_t)
            else:
                state_t_1, reward_t_1, done, _                  = env.step(action_t)

            env.render()



            # Choose A' from A(s') using policy derived from Q (e.g., -greedy)
                    # Returns the indices of the maximum values along an axis.
            optimal_action_index = np.argmax(Q_VALUE[state_t_1])

            probability_of_any_action_policy = []
            for i in range(0 ,action_space_size,1):
                probability_of_any_action_policy.append(0)
                if i == optimal_action_index:
                    probability_of_any_action_policy[i] = 1 - Parameters.epsilon    +    Parameters.epsilon/action_space_size
                else:
                    probability_of_any_action_policy[i] =                                Parameters.epsilon/action_space_size


            # https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
            # prende un valore dall'array num_outputs con le probabilita probability_of_any_action_policy
            # baco di numpy , non calcola correttamente array a somma 1
            p = probability_of_any_action_policy / np.sum(probability_of_any_action_policy)

            action_t_1 = np.random.choice(action_space_size, p=p)

            currentQValue =  Q_VALUE[state_t][action_t]
            nextQValue =  Q_VALUE[state_t_1][action_t_1]

            Q_VALUE[state_t][action_t] = currentQValue + Parameters.alpha*(reward_t_1 + Parameters.GAMMA*nextQValue - currentQValue)

            state_t = state_t_1
            action_t = action_t_1

            if done == True:
                break





if 'google.colab' in str(get_ipython()):
    print('Running on CoLab')
    Parameters.RunningEnvironment = 'COLAB'
else:
    print('Not running on CoLab')
    Parameters.RunningEnvironment = 'VS'
# hyperparameters
# if torch.cuda.is_available():
#    print('cuda.is_available')

#else:
#    print('cuda.is_not_available')


env = gym.make(Parameters.AtariName,render_mode="human")

# https://gymnasium.farama.org/api/wrappers/observation_wrappers/
print(env.observation_space.shape)

observation_space = env.observation_space.n
action_space = env.action_space.n

Sarsa(env,observation_space,action_space)