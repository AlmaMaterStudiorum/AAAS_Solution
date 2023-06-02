resizeRatio = 10
frame = 1
RunningEnvironment = '???'

# higher GAMMA higher exploration lower exploitation
GAMMA = 0.99
alpha = 0.01
max_steps_for_episode = 10000
max_episodes = 3000

# Parameters: step sizes αθ > 0 and αw > 0
alphatheta = 1e-3
alphaw = 1e-3

NNFolder = r"D:\Data\Temp\NN"

epsilon = 10*1e-2

SaveNets = False

#AtariName = "Taxi-v3"
AtariName = "FrozenLake-v1"