'''
    File name: rlcard.examples.gin_rummy_dqn.py
    Author: William Hale
    Date created: 2/12/2020

    An example of learning a Deep-Q Agent on GinRummy
'''

import torch
import os

import rlcard

from rlcard.agents import DQNAgentPytorch as DQNAgent
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger

# Make environment
env = rlcard.make('gin-rummy', config={'seed': 0})
eval_env = rlcard.make('gin-rummy', config={'seed': 0})
env.game.settings.print_settings()

# Set the iterations numbers and how frequently we evaluate/save plot
evaluate_every = 2
evaluate_num = 2  # mahjong_dqn has 1000
episode_num = 10  # mahjong_dqn has 100000

# The initial memory size
memory_init_size = 1000

# Train the agent every X steps
train_every = 1

# The paths for saving the logs and learning curves
log_dir = './experiments/gin_rummy_dqn_result/'

# Set a global seed
set_global_seed(0)

agent = DQNAgent(scope='dqn',
                     action_num=env.action_num,
                     #replay_memory_size=20000,
                     replay_memory_size=1000,
                     #replay_memory_init_size=memory_init_size,
                     replay_memory_init_size=500,
                     train_every=train_every,
                     #state_shape=env.state_shape,
                     state_shape = [768],
                     mlp_layers=[512, 512],
                     device=torch.device('cpu'))

random_agent = RandomAgent(action_num=eval_env.action_num)
env.set_agents([agent, random_agent])
eval_env.set_agents([agent, random_agent])



# Init a Logger to plot the learning curve
logger = Logger(log_dir)

for episode in range(episode_num):
    print('epi: ', episode)

    # Generate data from the environment
    trajectories, _ = env.run(is_training=True)

    # Feed transitions into agent memory, and train the agent
    i = 0;
    for ts in trajectories[0][:3]:
        print("i = ",i)
        i += 1
        if i == 45:
            print("hi)")
        agent.feed(ts)

    # extra logging
    if episode % evaluate_every == 0:
        reward = 0
        reward2 = 0
        for eval_episode in range(evaluate_num):
            _, payoffs = eval_env.run(is_training=False)
            reward += payoffs[0]
            reward2 += payoffs[1]
        logger.log("\n\n########## Evaluation {} ##########".format(episode))
        reward_text = "{}".format(float(reward)/evaluate_num)
        reward2_text = "{}".format(float(reward2)/evaluate_num)
        info = "Timestep: {} Average reward is {}, reward2 is {}".format(env.timestep, reward_text, reward2_text)
        logger.log(info)

    # Evaluate the performance. Play with random agents.
    if episode % evaluate_every == 0:
        logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])

# Close files in the logger
logger.close_files()

# Plot the learning curve
logger.plot('DQN')

# Save model
save_dir = 'models/gin_rummy_dqn_pytorch'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
state_dict = agent.get_state_dict()
print(state_dict.keys())
torch.save(state_dict, os.path.join(save_dir, 'model.pth'))


