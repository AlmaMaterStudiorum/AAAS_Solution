import gym
from gymnasium.wrappers import ResizeObservation
from gymnasium.wrappers import GrayScaleObservation
from gymnasium.wrappers import FlattenObservation
from SemiGradientOneStepSarsaControl import sgosscOnAtari
from IPython import get_ipython
import torch
if __name__ == "__main__":


    RunningEnvironment = '???'
    if 'google.colab' in str(get_ipython()):
      print('Running on CoLab')
      RunningEnvironment = 'COLAB'
    else:
      print('Not running on CoLab')
      RunningEnvironment = 'VS'
    # hyperparameters
    if torch.cuda.is_available():
      print('cuda.is_available')

    else:
      print('cuda.is_not_available')
    

    env = gym.make("ALE/Alien-v5",difficulty =3,render_mode="human")

    # https://gymnasium.farama.org/api/wrappers/observation_wrappers/
    print(env.observation_space.shape)
    resizeRatio = 5
    frame = 4

    dim0 = int(env.observation_space.shape[0]/resizeRatio)
    dim1 = int(env.observation_space.shape[1]/resizeRatio)
    shape = (dim0,dim1)
    env = gym.wrappers.ResizeObservation(env, shape)
    print(env.observation_space.shape)

    env = gym.wrappers.GrayScaleObservation(env)
    print(env.observation_space.shape)

    env = gym.wrappers.FlattenObservation(env)
    print(env.observation_space.shape)

    env = gym.wrappers.FrameStack(env, frame)


    observation_space = env.observation_space.shape[1]
    action_space = env.action_space.n
        


    # policytype
    # 0 : as output of actor , so softmax/stocastic
    # 1 : greedy policy (deterministic)
    # 2 : epsilon greedy policy
    # 3 : softmax (on softmax of actor)
    # 4 : decreasing epsilon greedy policy
    policytype = 3

    sgosscOnAtari(env,policytype,observation_space,action_space,frame)




