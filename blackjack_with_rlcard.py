''' An example of training a reinforcement learning agent on the environments in RLCard
'''
import os
import argparse
import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import get_device, set_seed, tournament, reorganize, Logger, plot_curve

def train(args):

    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(args.seed)
   
    # Make the environment woth seed and initialize the agent
    if args.algorithm == 'dqn':
        # Make the environment with seed
        env = rlcard.make(args.env, config={'seed': args.seed})
        
        # Initialize the agent
        from rlcard.agents import DQNAgent
        agent = DQNAgent(num_actions=env.num_actions,
                         state_shape=env.state_shape[0],
                         mlp_layers=[64, 64],
                         device=device)
        
        agents = [agent]
        for _ in range(env.num_players):
            agents.append(RandomAgent(num_actions=env.num_actions))
        env.set_agents(agents) # 将对应 agent 初始化到环境中
        
    elif args.algorithm == 'cfr': # ❗️
        # Make the environment with seed
        env = rlcard.make(args.env, config={'seed': args.seed, 'allow_step_back': True})
        eval_env = rlcard.make(args.env, config={'seed': args.seed})

        # Initialize the agent
        from rlcard.agents import CFRAgent
        agent = CFRAgent(env, os.path.join(args.log_dir, 'cfr_model'))
        agent.load() # If we have saved model, we first load the model
        
        # Evaluate CFR against random
        eval_env.set_agents([agent, RandomAgent(num_actions=env.num_actions)])

    elif args.algorithm == "ppo":
        pass
    
    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):

            if args.algorithm == 'cfr':
                agent.train()
                print('\rIteration {}'.format(episode), end='')
                
            elif args.algorithm == "dqn":
                # Generate data from the environment （玩完一局游戏后，将所有玩家最新状态和游戏结果存储起来）
                trajectories, payoffs = env.run(is_training=True)

                # Reorganaize the data to be state, action, reward, next_state, done （在 trajectories 的基础上，将每一步行动的奖励值和是否结束也标记出来）
                trajectories = reorganize(trajectories, payoffs)

                # Feed transitions into agent memory, and train the agent
                # Here, we assume that DQN always plays the first position
                # and the other players play randomly (if any)
                for ts in trajectories[0]: # 将每一个 trajectory 装入 memory，等待一定时机进行批量训练
                    agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0: # episode 每运行 evaluate_every 次，则输出本次 episode 内的 timestep 和 reward，并存储到 log.txt 文件内
                if args.algorithm == "cfr":
                    agent.save()
                    logger.log_performance(env.timestep, tournament(eval_env, args.num_eval_games)[0])
                elif args.algorithm == "dqn":
                    logger.log_performance(env.timestep, tournament(env, args.num_eval_games)[0])

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, args.algorithm)

    # Save model
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/CFR example in RLCard")
    parser.add_argument('--env', type=str, default='blackjack')
    parser.add_argument('--algorithm', type=str, default='dqn', choices=['dqn', 'cfr'])
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=5000)
    parser.add_argument('--num_eval_games', type=int, default=2000)
    parser.add_argument('--evaluate_every', type=int, default=100)
    parser.add_argument('--log_dir', type=str, default='experiments/blackjack/dqn/')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)

